"""Naive trajectory sampling.
Adapted from https://github.com/rvignav/diffusion-tts.
"""

import torch

from tabasco.chem.convert import MoleculeConverter

from .traj_utils import sample_coord_noise, step_with_coord_noise


def sample_traj_naive(
    lightning_module,
    num_steps=100,
    n_samples=10,
    seed=42,
):
    device = next(lightning_module.parameters()).device
    mol_converter = MoleculeConverter()
    model = lightning_module.model

    T = model._get_sample_schedule(num_steps=num_steps).to(device)

    all_mols = []
    total_forward_calls = 0

    for sample_idx in range(n_samples):
        torch.manual_seed(seed + sample_idx * 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + sample_idx * 1000)

        x_t = model._sample_noise_like_batch(batch_size=1)
        T_batch = T[:, None].repeat(1, x_t["coords"].shape[0])

        for step_i in range(1, num_steps + 1):
            t = T_batch[step_i - 1]
            dt = T_batch[step_i] - t

            noise = sample_coord_noise(model.coords_interpolant, x_t, t)
            x_t = step_with_coord_noise(model, x_t, t, dt, noise)
            total_forward_calls += 1

        mols = mol_converter.from_batch(x_t)
        all_mols.extend(mols)

    return {
        "mols": all_mols,
        "total_forward_calls": total_forward_calls,
    }
