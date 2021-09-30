import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from data.target_data import Data as TargetData
from data.source_data import Data as SourceData
from props.properties import penalized_logp
DATA_DIR = "../resource/data"


class ZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    def __init__(self, split):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return TargetData.from_smiles(smiles).featurize()


class QM9Dataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/qm9"

class SimpleMosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"

class MosesDataset(ZincDataset):
    raw_dir = f"{DATA_DIR}/moses"

class LogPZincDataset(Dataset):
    raw_dir = f"{DATA_DIR}/zinc"
    def __init__(self, split):
        smiles_list_path = os.path.join(self.raw_dir, f"{split}.txt")
        self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()
        
    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return TargetData.from_smiles(smiles).featurize(), torch.tensor([penalized_logp(smiles)])

class LogP04Dataset(Dataset):
    raw_dir = f"{DATA_DIR}/logp04"

    def __init__(self, split):
        self.split = split
        if self.split == "train":
            smiles_list_path = os.path.join(self.raw_dir, "train_pairs.txt")
            smiles_pair_list = [
                pair.split() for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
            ]
            self.src_smiles_list, self.tgt_smiles_list = map(list, zip(*smiles_pair_list))
        else:
            smiles_list_path = os.path.join(self.raw_dir, f"{self.split}.txt")
            self.smiles_list = Path(smiles_list_path).read_text(encoding="utf=8").splitlines()

    def __len__(self):
        if self.split == "train":
            return len(self.src_smiles_list)
        else:
            return len(self.smiles_list)

    def __getitem__(self, idx):
        if self.split == "train":
            src_smiles = self.src_smiles_list[idx]
            tgt_smiles = self.tgt_smiles_list[idx]
            return (
                SourceData.from_smiles(src_smiles).featurize(),
                TargetData.from_smiles(tgt_smiles).featurize(),
            )
        else:
            smiles = self.smiles_list[idx]
            return SourceData.from_smiles(smiles).featurize(), smiles

class LogP06Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/logp06"

class DRD2Dataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/drd2"

class QEDDataset(LogP04Dataset):
    raw_dir = f"{DATA_DIR}/qed"
