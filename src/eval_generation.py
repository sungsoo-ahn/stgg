import argparse
from pathlib import Path
import moses
from tqdm import tqdm
from rdkit import Chem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_list_path", type=str, default="")
    parser.add_argument("--train_smiles_list_path", type=str, default="")
    parser.add_argument("--test_smiles_list_path", type=str, default="")
    hparams = parser.parse_args()

    smiles_list = []
    for path in [
        '/home/peterahn/Downloads/sample_smiles_014.csv', 
        '/home/peterahn/Downloads/sample_smiles_019.csv', 
        '/home/peterahn/Downloads/sample_smiles_024.csv'
        ]:
        smiles_list_ = Path(path).read_text(encoding="utf-8").splitlines()
        smiles_list += [result.split(",")[2] for result in smiles_list_]
    print(len(smiles_list))
    assert False
    metrics = moses.get_all_metrics(smiles_list, n_jobs=16, device="cuda:0")
    print(metrics)

    #train_smiles_list = Path(hparams.train_smiles_list_path).read_text(encoding="utf-8").splitlines()
    #test_smiles_list = Path(hparams.test_smiles_list_path).read_text(encoding="utf-8").splitlines()
    #metrics = moses.get_all_metrics(smiles_list, n_jobs=8, device="cuda:0")

    #print(metrics)
