import json
import pickle
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from tabasco.utils.metrics import (
    MolecularConnectivity,
    MolecularDiversity,
    MolecularLipinski,
    MolecularLogP,
    MolecularQEDValue,
    MolecularUniqueness,
    MolecularValidity,
)


def compute_metrics(generated_mols):
    metrics = {
        "Validity": MolecularValidity(),
        "Uniqueness": MolecularUniqueness(),
        "Connectivity": MolecularConnectivity(),
        "Diversity": MolecularDiversity(),
        "QED": MolecularQEDValue(),
        "LogP": MolecularLogP(),
        "Lipinski": MolecularLipinski(),
    }
    for metric in metrics.values():
        metric.update(generated_mols)
    return {name: metric.compute().item() for name, metric in metrics.items()}


def compute_per_mol_qed(mols):
    scores = []
    for i, mol in enumerate(mols):
        if mol is not None:
            scores.append((i, Descriptors.qed(mol)))
    return scores


def mols_to_smiles(mols):
    smiles = []
    for mol in mols:
        if mol is not None:
            smiles.append(Chem.MolToSmiles(mol))
    return smiles


def save_results(output_dir, mols, metrics_dict, config_dict, extra_files=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "generated_molecules.pkl", "wb") as f:
        pickle.dump(mols, f)

    with open(output_dir / "metrics.txt", "w") as f:
        for name, val in metrics_dict.items():
            f.write(f"{name}: {val:.4f}\n")

    smiles = mols_to_smiles(mols)
    with open(output_dir / "generated_smiles.txt", "w") as f:
        for smi in smiles:
            f.write(smi + "\n")

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    qed_scores = compute_per_mol_qed(mols)
    with open(output_dir / "qed_scores.txt", "w") as f:
        for idx, score in sorted(qed_scores, key=lambda x: -x[1]):
            f.write(f"{idx}\t{score:.4f}\n")

    if extra_files:
        for fname, content in extra_files.items():
            with open(output_dir / fname, "w") as f:
                f.write(content)


class Timer:

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
