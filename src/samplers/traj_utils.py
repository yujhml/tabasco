"""Shared utilities for trajectory-based noise search methods.
Adapted from https://github.com/rvignav/diffusion-tts.
"""

from copy import deepcopy

import numpy as np
import torch
from tensordict import TensorDict

from tabasco.utils.tensor_ops import mask_and_zero_com


# ── Noise helpers ────────────────────────────────────────────────────────────────


def sample_coord_noise(interp, x_t, t):
    t_exp = t.unsqueeze(-1).unsqueeze(-1)
    wiener_noise_scale = torch.sqrt(
        2 * interp.langevin_sampling_schedule(t_exp) * interp.white_noise_sampling_scale
    ) * torch.randn_like(x_t["coords"])
    white_noise = (
        interp.sample_noise(x_t["coords"].shape, x_t["padding_mask"])
        * wiener_noise_scale
    )
    return white_noise


def perturb_noise(base_noise, lambda_param):
    direction = torch.randn_like(base_noise)
    dims = tuple(range(1, direction.dim()))
    direction = direction / (torch.norm(direction, p=2, dim=dims, keepdim=True) + 1e-8)
    shape = [direction.shape[0]] + [1] * (direction.dim() - 1)
    scale = torch.rand(shape, device=direction.device) * lambda_param
    return base_noise + scale * direction


# ── Step functions ───────────────────────────────────────────────────────


def step_with_coord_noise(model, x_t, t, dt, coord_noise, out_batch=None):
    if out_batch is None:
        with torch.no_grad():
            out_batch = model._call_net(x_t, t)

    interp = model.coords_interpolant
    t_exp = t.unsqueeze(-1).unsqueeze(-1)
    dt_exp = dt.unsqueeze(-1).unsqueeze(-1)

    x1_pred = out_batch[interp.key]
    velocity = (x1_pred - x_t[interp.key]) / (1 - t_exp)
    score = interp.calculate_score(velocity, x_t[interp.key], t_exp)
    component_score = interp.langevin_sampling_schedule(t_exp) * score

    x_new = (
        x_t[interp.key] + velocity * dt_exp
        + component_score * dt_exp + coord_noise * dt_exp
    )
    x_new = mask_and_zero_com(x_new, x_t["padding_mask"])
    x_t["coords"] = x_new

    x_t["atomics"] = model.atomics_interpolant.step(x_t, out_batch, t, dt)

    return x_t


# ── Scoring helpers ───────────────────────────────────────────────────────────────


def score_endpoint(model, mol_converter, x_t, t, reward_fn):
    with torch.no_grad():
        pred = model._call_net(x_t, t)
    mols = mol_converter.from_batch(pred)
    mol_scores = [reward_fn(m) for m in mols]
    return max(mol_scores) if mol_scores else 0.0


def score_endpoint_batch(model, mol_converter, x_t, t, reward_fn):
    with torch.no_grad():
        pred = model._call_net(x_t, t)
    mols = mol_converter.from_batch(pred)
    return np.array([reward_fn(m) for m in mols])


def score_direct(mol_converter, x_t, reward_fn):
    mols = mol_converter.from_batch(x_t)
    mol_scores = [reward_fn(m) for m in mols]
    return max(mol_scores) if mol_scores else 0.0


# ── TensorDict helpers ───────────────────────────────────────────────────


def clone_state(x_t):
    return deepcopy(x_t)


def select_state(x_t, idx):
    if isinstance(idx, int):
        idx = slice(idx, idx + 1)
    return TensorDict(
        {"coords": x_t["coords"][idx].clone(),
         "atomics": x_t["atomics"][idx].clone(),
         "padding_mask": x_t["padding_mask"][idx].clone()},
        batch_size=[x_t["coords"][idx].shape[0]],
    )


def cat_states(states):
    return TensorDict(
        {"coords": torch.cat([s["coords"] for s in states], dim=0),
         "atomics": torch.cat([s["atomics"] for s in states], dim=0),
         "padding_mask": torch.cat([s["padding_mask"] for s in states], dim=0)},
        batch_size=[sum(s["coords"].shape[0] for s in states)],
    )
