"""Beam search over trajectory noise.
Adapted from https://github.com/rvignav/diffusion-tts.
"""

from copy import deepcopy

import numpy as np
import torch

from tabasco.chem.convert import MoleculeConverter

from .rewards import REWARD_FUNCTIONS
from .traj_utils import (
    sample_coord_noise,
    step_with_coord_noise,
    score_endpoint,
    clone_state,
)


def sample_traj_beam(
    lightning_module,
    B=2,
    K=4,
    reward="qed",
    scoring="endpoint",
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

        # Initialize K beams from the same initial noise
        # (diffusion-tts: x_next_expanded = x_next.repeat_interleave(k, dim=0))
        x_init = model._sample_noise_like_batch(batch_size=1)
        T_batch_single = T[:, None].repeat(1, 1)

        beams = [clone_state(x_init) for _ in range(K)]

        for step_i in range(1, num_steps + 1):
            t = T_batch_single[step_i - 1]
            dt = T_batch_single[step_i] - t
            t_next = T_batch_single[step_i]

            # Expand each beam by B candidates
            # (diffusion-tts: "For each existing trajectory...
            #  Expand current beam to b candidates")
            all_candidates = []

            for beam in beams:
                for b_idx in range(B):
                    x_cand = clone_state(beam)
                    noise = sample_coord_noise(
                        model.coords_interpolant, x_cand, t
                    )
                    x_cand = step_with_coord_noise(
                        model, x_cand, t, dt, noise
                    )
                    total_forward_calls += 1
                    all_candidates.append(x_cand)

            scores_k = []
            for x_cand in all_candidates:
                if scoring == "endpoint":
                    s = score_endpoint(
                        model, mol_converter, x_cand, t_next, reward_fn
                    )
                    total_forward_calls += 1
                else:
                    mols_tmp = mol_converter.from_batch(x_cand)
                    s = max([reward_fn(m) for m in mols_tmp]) if mols_tmp else 0.0
                scores_k.append(s)

            top_indices = np.argsort(scores_k)[-K:][::-1]
            beams = [clone_state(all_candidates[i]) for i in top_indices]

        mol = mol_converter.from_batch(beams[0])
        mol = mol[0] if mol else None
        score = reward_fn(mol)
        all_mols.append(mol)
        all_scores.append(score)

    return {
        "mols": all_mols,
        "scores": all_scores,
        "total_forward_calls": total_forward_calls,
    }
