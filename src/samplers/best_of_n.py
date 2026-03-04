"""Best-of-N sampling.
"""

import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem import AllChem

from tabasco.chem.convert import MoleculeConverter
from .rewards import REWARD_FUNCTIONS


def maxmin_pick(mols, scores, k):
    fps = []
    valid_indices = []
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)
        valid_indices.append(i)

    if len(fps) <= k:
        return valid_indices

    scored = [(idx, scores[idx]) for idx in valid_indices]
    scored.sort(key=lambda x: -x[1])
    selected = [scored[0][0]]
    remaining = set(idx for idx, _ in scored[1:])
    idx_to_fp = {idx: fp for idx, fp in zip(valid_indices, fps)}

    for _ in range(k - 1):
        if not remaining:
            break
        best_idx = None
        best_min_dist = -1.0
        for idx in remaining:
            min_dist = min(
                1.0 - DataStructs.TanimotoSimilarity(idx_to_fp[idx], idx_to_fp[s])
                for s in selected
            )
            if min_dist > best_min_dist or (
                min_dist == best_min_dist
                and (best_idx is None or scores[idx] > scores[best_idx])
            ):
                best_min_dist = min_dist
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def sample_best_of_n(
    lightning_module,
    n_candidates,
    top_k,
    reward="qed",
    num_steps=100,
    diverse=False,
    batch_size=500,
):
    mol_converter = MoleculeConverter()
    reward_fn = REWARD_FUNCTIONS[reward]

    all_mols = []
    remaining = n_candidates
    while remaining > 0:
        bs = min(batch_size, remaining)
        with torch.no_grad():
            batch = lightning_module.sample(batch_size=bs, num_steps=num_steps)
        mols = mol_converter.from_batch(batch)
        all_mols.extend(mols)
        remaining -= bs

    scores = [reward_fn(mol) for mol in all_mols]

    if diverse:
        selected_indices = maxmin_pick(all_mols, scores, top_k)
    else:
        scored_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        selected_indices = scored_indices[:top_k]

    selected_mols = [all_mols[i] for i in selected_indices]
    selected_scores = [scores[i] for i in selected_indices]

    return {
        "all_mols": all_mols,
        "selected_mols": selected_mols,
        "selected_scores": selected_scores,
        "all_scores": scores,
        "total_forward_calls": n_candidates * num_steps,
    }
