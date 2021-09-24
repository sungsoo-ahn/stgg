import argparse
from pathlib import Path
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
from props.properties import penalized_logp, drd2, qed, similarity

from rdkit import RDLogger

from rdkit import rdBase

def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')

def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')

from rdkit import Chem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_list_path", type=str, default="")
    parser.add_argument("--task", type=str, default="logp04")
    hparams = parser.parse_args()

    if hparams.task == "logp04":
        score_func = lambda src, tgt: penalized_logp(tgt) - penalized_logp(src)
        similarity_thr = 0.4

    elif hparams.task == "logp06":
        score_func = lambda src, tgt: penalized_logp(tgt) - penalized_logp(src)
        similarity_thr = 0.6

    elif hparams.task == "drd2":
        score_func = lambda src, tgt: float(drd2(tgt) >= 0.5)
        similarity_thr = 0.4

    elif hparams.task == "qed":
        score_func = lambda src, tgt: float(qed(tgt) >= 0.9)
        similarity_thr = 0.4

    def batch_score_func(smiles_list):
        src, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for tgt in tgt_list:
            try:
                score_list.append(score_func(src, tgt))
            except:
                score_list.append(None)

        while len(score_list) < 20:
            score_list.append(None)

        return score_list

    def batch_similarity_func(smiles_list):
        src, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for tgt in tgt_list:
            try:
                score_list.append(similarity(src, tgt))
            except:
                score_list.append(None)

        while len(score_list) < 20:
            score_list.append(None)

        return score_list

    def batch_diversity_func(smiles_list):
        _, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for smi0, smi1 in combinations(tgt_list, 2):
            try:
                score_list.append(1 - similarity(smi0, smi1))
            except:
                score_list.append(None)

        while len(score_list) < 20 * 19 / 2:
            score_list.append(None)

        return score_list

    def batch_validity_func(smiles_list):
        src, tgt_list = smiles_list[0], smiles_list[1:]
        score_list = []
        for tgt in tgt_list:
            disable_rdkit_log()
            try:
                mol = Chem.MolFromSmiles(tgt)
                tgt = Chem.MolToSmiles(mol)
                score_list.append(1.0)
            except:
                print(tgt_list)
                print(tgt)
                assert False
                score_list.append(0.0)
            
            enable_rdkit_log()
        
        while len(score_list) < 20:
            score_list.append(0.0)

        return score_list

    lines = Path(hparams.smiles_list_path).read_text(encoding="utf-8").splitlines()
    lines = [line.split(",")[2:] for line in lines]

    #validity = Parallel(n_jobs=8)(delayed(batch_validity_func)(line) for line in lines)
    validity = []
    for line in lines:
        print(line)
        validity.append(batch_validity_func(line))

    validity = np.array(validity, dtype=np.float)
    print(validity.mean())

    scores = Parallel(n_jobs=8)(delayed(batch_score_func)(line) for line in lines)
    similarities = Parallel(n_jobs=8)(delayed(batch_similarity_func)(line) for line in lines)

    scores = np.array(scores, dtype=np.float)
    similarities = np.array(similarities, dtype=np.float)

    thresholded_scores = scores.copy()
    thresholded_scores[similarities < similarity_thr] = np.nan
    thresholded_scores = np.nanmax(thresholded_scores, axis=1)
    thresholded_scores[np.isnan(thresholded_scores)] = 0.0
    print(np.nanmean(thresholded_scores), np.nanstd(thresholded_scores))

    diversities = Parallel(n_jobs=8)(delayed(batch_diversity_func)(line) for line in lines)
    diversities = np.nanmean(np.array(diversities, dtype=np.float), axis=1)
    print(np.nanmean(diversities), np.nanstd(diversities))
