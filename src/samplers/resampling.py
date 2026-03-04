"""Exponential reweighting / resampling.
"""

import numpy as np
import torch

from tabasco.chem.convert import MoleculeConverter
from .rewards import REWARD_FUNCTIONS


def sample_resampling(
    lightning_module,
    n_candidates,
    top_k,
    beta=5.0,
    reward="qed",
    num_steps=100,
    batch_size=500,
    resample_seed=42,
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

    scores = np.array([reward_fn(mol) for mol in all_mols])

    log_weights = beta * scores
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()

    ess = 1.0 / np.sum(weights ** 2)
    ess_ratio = ess / len(all_mols)

    rng = np.random.default_rng(resample_seed)
    resampled_indices = rng.choice(
        len(all_mols), size=top_k, replace=True, p=weights
    )
    resampled_mols = [all_mols[i] for i in resampled_indices]
    resampled_scores = scores[resampled_indices]

    return {
        "all_mols": all_mols,
        "resampled_mols": resampled_mols,
        "resampled_scores": resampled_scores.tolist(),
        "all_scores": scores.tolist(),
        "ess": float(ess),
        "ess_ratio": float(ess_ratio),
        "total_forward_calls": n_candidates * num_steps,
    }
