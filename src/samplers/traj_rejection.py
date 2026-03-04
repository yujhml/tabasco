"""Rejection sampling over trajectory noise.
Adapted from https://github.com/rvignav/diffusion-tts.
"""

import numpy as np
import torch

from tabasco.chem.convert import MoleculeConverter

from .rewards import REWARD_FUNCTIONS
from .traj_utils import sample_coord_noise, step_with_coord_noise


def sample_traj_rejection(
    lightning_module,
    N=4,
    reward="qed",
    num_steps=100,
    n_samples=10,
    seed=42,
):
    device = next(lightning_module.parameters()).device
    mol_converter = MoleculeConverter()
    reward_fn = REWARD_FUNCTIONS[reward]
    model = lightning_module.model

    T = model._get_sample_schedule(num_steps=num_steps).to(device)

    all_mols = []
    all_scores = []
    total_forward_calls = 0

    for sample_idx in range(n_samples):
        torch.manual_seed(seed + sample_idx * 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + sample_idx * 1000)

        # Run N full trajectories and pick the best
        # (diffusion-tts: "Do the whole denoising loop for N initial
        #  candidate noise vectors, then choose the best")
        candidate_mols = []
        candidate_scores = []

        for n in range(N):
            x_t = model._sample_noise_like_batch(batch_size=1)
            T_batch = T[:, None].repeat(1, x_t["coords"].shape[0])

            for step_i in range(1, num_steps + 1):
                t = T_batch[step_i - 1]
                dt = T_batch[step_i] - t

                noise = sample_coord_noise(model.coords_interpolant, x_t, t)
                x_t = step_with_coord_noise(model, x_t, t, dt, noise)
                total_forward_calls += 1

            mols = mol_converter.from_batch(x_t)
            mol = mols[0] if mols else None
            score = reward_fn(mol)
            candidate_mols.append(mol)
            candidate_scores.append(score)

        best_idx = int(np.argmax(candidate_scores))
        all_mols.append(candidate_mols[best_idx])
        all_scores.append(candidate_scores[best_idx])

    return {
        "mols": all_mols,
        "scores": all_scores,
        "total_forward_calls": total_forward_calls,
    }
