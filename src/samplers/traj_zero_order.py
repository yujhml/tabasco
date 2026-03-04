"""Zero-order noise trajectory search.
Adapted from https://github.com/rvignav/diffusion-tts.
"""

from copy import deepcopy

import numpy as np
import torch

from tabasco.chem.convert import MoleculeConverter

from .rewards import REWARD_FUNCTIONS
from .traj_utils import (
    sample_coord_noise,
    perturb_noise,
    step_with_coord_noise,
    score_endpoint,
    clone_state,
)


def sample_traj_zero_order(
    lightning_module,
    N=4,
    K=5,
    eps=0.4,
    lambda_=0.15,
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

    coord_dim = (
        model.data_stats["max_num_atoms"] * model.data_stats["spatial_dim"]
    )
    lambda_scaled = lambda_ * np.sqrt(coord_dim)

    all_mols = []
    all_scores = []
    total_forward_calls = 0

    for sample_idx in range(n_samples):
        torch.manual_seed(seed + sample_idx * 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + sample_idx * 1000)

        x_t = model._sample_noise_like_batch(batch_size=1)
        T_batch = T[:, None].repeat(1, x_t["coords"].shape[0])

        # ── Main denoising loop over timesteps ───────────────────────
        for step_i in range(1, num_steps + 1):
            t = T_batch[step_i - 1]
            dt = T_batch[step_i] - t
            t_next = T_batch[step_i]

            with torch.no_grad():
                out_batch = model._call_net(x_t, t)
            total_forward_calls += 1

            pivot_noise = sample_coord_noise(
                model.coords_interpolant, x_t, t
            )

            for k in range(K):
                candidate_noises = []
                candidate_states = []

                for n in range(N):
                    if torch.rand(1, device=device).item() < (1 - eps):
                        noise = perturb_noise(pivot_noise, lambda_scaled)
                    else:
                        noise = sample_coord_noise(
                            model.coords_interpolant, x_t, t
                        )
                    candidate_noises.append(noise)

                    x_cand = deepcopy(x_t)
                    x_cand = step_with_coord_noise(
                        model, x_cand, t, dt, noise, out_batch=out_batch
                    )
                    candidate_states.append(x_cand)

                scores_k = []
                for x_cand in candidate_states:
                    if scoring == "endpoint":
                        s = score_endpoint(
                            model, mol_converter, x_cand, t_next, reward_fn
                        )
                        total_forward_calls += 1
                    else:
                        mols_tmp = mol_converter.from_batch(x_cand)
                        s = max(
                            [reward_fn(m) for m in mols_tmp]
                        ) if mols_tmp else 0.0
                    scores_k.append(s)

                best_idx = int(np.argmax(scores_k))
                pivot_noise = candidate_noises[best_idx]

            x_t = step_with_coord_noise(
                model, x_t, t, dt, pivot_noise, out_batch=out_batch
            )

        mols = mol_converter.from_batch(x_t)
        for mol in mols:
            score = reward_fn(mol)
            all_mols.append(mol)
            all_scores.append(score)

    return {
        "mols": all_mols,
        "scores": all_scores,
        "total_forward_calls": total_forward_calls,
    }
