from collections import defaultdict
import os
import argparse
from joblib import Parallel, delayed
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from model.translator import BaseTranslator
from data.dataset import LogP04Dataset, LogP06Dataset, DRD2Dataset, QEDDataset
from data.source_data import Data as SourceData
from data.target_data import Data as TargetData
from util import compute_sequence_cross_entropy, compute_sequence_accuracy, canonicalize

from tqdm import tqdm
from moses.utils import disable_rdkit_log, enable_rdkit_log

class BaseTranslatorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseTranslatorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.sanity_checked = False

    def setup_datasets(self, hparams):
        dataset_cls = {"logp04": LogP04Dataset, "logp06": LogP06Dataset, "drd2": DRD2Dataset, "qed": QEDDataset,}.get(
            self.hparams.dataset_name
        )
        self.train_dataset = dataset_cls("train")
        self.train_dataset.src_smiles_list = self.train_dataset.src_smiles_list
        self.train_dataset.tgt_smiles_list = self.train_dataset.tgt_smiles_list
        self.val_dataset = dataset_cls("test")

        def train_collate(data_list):
            src, tgt = zip(*data_list)
            src = SourceData.collate(src)
            tgt = TargetData.collate(tgt)
            return src, tgt

        self.train_collate = train_collate

        def eval_collate(data_list):
            src, src_smiles_list = zip(*data_list)
            src = SourceData.collate(src)
            return src, src_smiles_list

        self.eval_collate = eval_collate

    def setup_model(self, hparams):
        self.model = BaseTranslator(
            hparams.num_layers, hparams.emb_size, hparams.nhead, hparams.dim_feedforward, hparams.dropout,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.train_collate,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            collate_fn=self.eval_collate,
            num_workers=self.hparams.num_workers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    ### Main steps
    def training_step(self, batched_data, batch_idx):
        self.sanity_checked = True

        loss, statistics = 0.0, dict()
        src, tgt = batched_data

        # decoding
        logits = self.model(src, tgt)
        loss = compute_sequence_cross_entropy(logits, tgt[0])
        acc = compute_sequence_accuracy(logits, tgt[0])[0]

        statistics["loss/total"] = loss
        statistics["acc/total"] = acc
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        src, src_smiles_list = batched_data
        max_len = self.hparams.max_len if not self.trainer.sanity_checking else 10
        num_repeats = self.hparams.num_repeats if not self.trainer.sanity_checking else 2
        tgt_smiles_list_list = []
        self.eval()
        for _ in range(num_repeats):
            with torch.no_grad():
                tgt_data_list = self.model.decode(src, max_len=max_len, device=self.device)

            for data in tgt_data_list:
                if data.error is not None:
                    self.logger.experiment[f"sample/error/{self.current_epoch:03d}"].log(
                        "".join(data.tokens) + " " + data.error
                        )

            tgt_smiles_list = [data.to_smiles() for data in tgt_data_list]
            disable_rdkit_log()
            tgt_smiles_list = [str(canonicalize(smiles)) for smiles in tgt_smiles_list]
            enable_rdkit_log()
            tgt_smiles_list_list.append(tgt_smiles_list)

        tgt_smiles_list_list = list(map(list, zip(*tgt_smiles_list_list)))
        if not self.trainer.sanity_checking:
            for src_smiles, tgt_smiles_list in zip(src_smiles_list, tgt_smiles_list_list):
                for tgt_smiles in tgt_smiles_list:
                    self.logger.experiment[f"sample/{self.current_epoch:03d}"].log(",".join([src_smiles, tgt_smiles]))

    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="logp04")

        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=int, default=0.1)

        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--eval_batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=6)

        parser.add_argument("--num_repeats", type=int, default=20)
        parser.add_argument("--max_len", type=int, default=250)

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BaseTranslatorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint_path", type=str, default=None)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=25)
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    hparams.checkpoint_dir = os.path.join("../resource/checkpoint/", hparams.tag)
    model = BaseTranslatorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])

    logger = NeptuneLogger(
        project="sungsahn0215/molgen", 
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsIm"
        "FwaV9rZXkiOiIyNjdkMDIxZi1lZDkwLTQ0ZDAtODg5Yi03ZTdjNThhYTdjMmQifQ==", 
        source_files="**/*.py"
    )
    logger.run["params"] = vars(hparams)
    logger.run["sys/tags"].add(hparams.tag.split("_"))

    checkpoint_callback = ModelCheckpoint(dirpath=hparams.checkpoint_dir, monitor="train/loss/total", mode="min")
    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=hparams.gradient_clip_val,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        resume_from_checkpoint=hparams.resume_from_checkpoint_path,
    )
    trainer.fit(model)
