import os
import argparse
from numpy.lib.arraysetops import unique

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from moses.utils import disable_rdkit_log, enable_rdkit_log

from model.generator import CondGenerator
from data.dataset import LogPZincDataset
from data.target_data import Data
from props.properties import penalized_logp
from util import compute_sequence_cross_entropy, compute_sequence_accuracy, canonicalize 
from train_generator import BaseGeneratorLightningModule

import numpy as np

class CondGeneratorLightningModule(BaseGeneratorLightningModule):
    def __init__(self, hparams):
        super(CondGeneratorLightningModule, self).__init__(hparams)
        
    def setup_datasets(self, hparams):
        self.train_dataset = LogPZincDataset("train")
        self.val_dataset = LogPZincDataset("valid")
        self.test_dataset = LogPZincDataset("test")
        self.train_smiles_set = set(self.train_dataset.smiles_list)

        def collate_fn(data_list):
            batched_mol_data, batched_cond_data = zip(*data_list)
            return Data.collate(batched_mol_data), torch.stack(batched_cond_data, dim=0)
        
        self.collate_fn = collate_fn

    def setup_model(self, hparams):
        self.model = CondGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            input_dropout=hparams.input_dropout,
            dropout=hparams.dropout,
            disable_treeloc=hparams.disable_treeloc,
            disable_graphmask=hparams.disable_graphmask, 
            disable_valencemask=hparams.disable_valencemask,
            enable_absloc=hparams.enable_absloc,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()

        # decoding
        batched_mol_data, batched_cond_data = batched_data
        logits = self.model(batched_mol_data, batched_cond_data)
        loss = compute_sequence_cross_entropy(logits, batched_mol_data[0], ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_mol_data[0], ignore_index=0)[0]

        return loss, statistics


    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        for cond_val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
            smiles_list, results = self.sample(cond_val, num_samples)

            #
            if not self.trainer.sanity_checking:
                for (smiles, result) in zip(smiles_list, results):
                    self.logger.experiment[f"sample/smiles/{self.current_epoch:03d}/cond{cond_val:.0f}"].log(smiles)
                    self.logger.experiment[f"sample/result/{self.current_epoch:03d}/cond{cond_val:.0f}"].log(result)

            #
            valid_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
            unique_smiles_set = set(valid_smiles_list)
            novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]
            statistics = dict()
            statistics[f"sample/valid/cond{cond_val:.0f}"] = float(len(valid_smiles_list)) / num_samples
            statistics[f"sample/unique/cond{cond_val:.0f}"] = float(len(unique_smiles_set)) / len(valid_smiles_list)
            statistics[f"sample/novel/cond{cond_val:.0f}"] = float(len(novel_smiles_list)) / len(valid_smiles_list)

            scores = np.array([penalized_logp(smiles) for smiles in unique_smiles_set])
            statistics[f"sample/mean_score/cond{cond_val:.0f}"] = np.mean(scores)
            statistics[f"sample/std_score/cond{cond_val:.0f}"] = np.std(scores)
            statistics[f"sample/max_score/cond{cond_val:.0f}"] = np.max(scores)

            #
            for key, val in statistics.items():
                self.log(key, val, on_step=False, on_epoch=True, logger=True)
            
    def sample(self, cond_val, num_samples):
        offset = 0
        results = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            batched_cond_data = torch.full((cur_num_samples, 1), cond_val, device=self.device)
            self.model.eval()
            with torch.no_grad():
                data_list = self.model.decode(batched_cond_data, max_len=self.hparams.max_len, device=self.device)

            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results

    @staticmethod
    def add_args(parser):
        #
        parser.add_argument("--dataset_name", type=str, default="zinc")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=6)

        #
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=float, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)
        
        #
        parser.add_argument("--randomize", action="store_true")
        parser.add_argument("--disable_treeloc", action="store_true")
        parser.add_argument("--disable_graphmask", action="store_true")
        parser.add_argument("--disable_valencemask", action="store_true")
        parser.add_argument("--enable_absloc", action="store_true")
        
        
        #
        parser.add_argument("--lr", type=float, default=2e-4)

        #
        parser.add_argument("--max_len", type=int, default=250)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=10)
        parser.add_argument("--num_samples", type=int, default=500)
        parser.add_argument("--sample_batch_size", type=int, default=500)
        parser.add_argument("--eval_moses", action="store_true")

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    CondGeneratorLightningModule.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()

    model = CondGeneratorLightningModule(hparams)
    if hparams.load_checkpoint_path != "":
        model.load_state_dict(torch.load(hparams.load_checkpoint_path)["state_dict"])

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsIm"
        "FwaV9rZXkiOiIyNjdkMDIxZi1lZDkwLTQ0ZDAtODg5Yi03ZTdjNThhYTdjMmQifQ==",
        project="sungsahn0215/molgen",
        source_files="**/*.py"
        )
    neptune_logger.run["params"] = vars(hparams)
    neptune_logger.run["sys/tags"].add(hparams.tag.split("_"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("../resource/checkpoint/", hparams.tag), 
        monitor="validation/loss/total", 
        save_top_k=-1, 
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
        resume_from_checkpoint=hparams.resume_from_checkpoint_path,
    )
    trainer.fit(model)