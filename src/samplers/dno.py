"""Direct Noise Optimisation / DNO.
Adapted from https://github.com/TZW1998/Direct-Noise-Optimization.
"""

import random as stdlib_random
import time
from copy import deepcopy

import numpy as np
import torch
from rdkit import Chem
from tensordict import TensorDict

from tabasco.chem.convert import MoleculeConverter
from tabasco.flow.interpolate import SDEMetricInterpolant
from tabasco.utils.tensor_ops import mask_and_zero_com

from .rewards import REWARD_FUNCTIONS


def compute_probability_regularization(
    noise_vectors_flat: torch.Tensor,
    subsample: int = 1,
    shuffled_times: int = 50,
) -> torch.Tensor:
    dim = noise_vectors_flat.shape[0]
    subsample_dim = round(4 ** subsample)
    subsample_num = dim // subsample_dim

    if subsample_num < 2 or subsample_dim < 2:
        return torch.tensor(0.0, device=noise_vectors_flat.device)

    trimmed = noise_vectors_flat[: subsample_num * subsample_dim]
    blocks = trimmed.view(subsample_num, subsample_dim)

    def _stats(blk):
        m = blk.mean(dim=0)
        blk_n = blk / np.sqrt(blk.shape[0])
        cov = blk_n.T @ blk_n
        m_M = torch.norm(m)
        eye = torch.eye(blk.shape[1], device=blk.device)
        c_M = torch.linalg.matrix_norm(cov - eye, ord=2)
        m_lp = torch.clamp(
            -(blk.shape[0] * m_M ** 2) / (2 * blk.shape[1]),
            max=-np.log(2),
        )
        c_diff = torch.clamp(
            torch.sqrt(1 + c_M) - 1 - np.sqrt(blk.shape[1] / blk.shape[0]),
            min=0,
        )
        c_lp = torch.clamp(-blk.shape[0] * c_diff ** 2 / 2, max=-np.log(2))
        return m_lp, c_lp

    seq_m, seq_c = _stats(blocks)

    shuf_m_sum = torch.tensor(0.0, device=noise_vectors_flat.device)
    shuf_c_sum = torch.tensor(0.0, device=noise_vectors_flat.device)
    for _ in range(shuffled_times):
        perm = torch.randperm(dim, device=noise_vectors_flat.device)
        shuf = noise_vectors_flat[perm][: subsample_num * subsample_dim].view(
            subsample_num, subsample_dim
        )
        s_m, s_c = _stats(shuf)
        shuf_m_sum = shuf_m_sum + s_m
        shuf_c_sum = shuf_c_sum + s_c

    reg_loss = -(seq_m + seq_c + (shuf_m_sum + shuf_c_sum) / shuffled_times)
    return reg_loss


# ── Custom differentiable SDE step ────────────────────────────────────────────────


def _sde_step_prealloc(interpolant, x_t_dict, pred, t, dt, noise_a, noise_b):
    key = interpolant.key
    key_pad = interpolant.key_pad_mask

    t_ = t.unsqueeze(-1).unsqueeze(-1)
    dt_ = dt.unsqueeze(-1).unsqueeze(-1)

    x1_pred = pred[key]
    velocity = (x1_pred - x_t_dict[key]) / (1 - t_)
    score = interpolant.calculate_score(velocity, x_t_dict[key], t_)
    component_score = interpolant.langevin_sampling_schedule(t_) * score

    wiener_noise_scale = (
        torch.sqrt(
            2 * interpolant.langevin_sampling_schedule(t_)
            * interpolant.white_noise_sampling_scale
        )
        * noise_b
    )

    noise_a_scaled = noise_a * interpolant.noise_scale
    noise_a_centered = mask_and_zero_com(noise_a_scaled, x_t_dict[key_pad])
    white_noise = noise_a_centered * wiener_noise_scale

    x_new = x_t_dict[key] + velocity * dt_ + component_score * dt_ + white_noise * dt_
    x_new = mask_and_zero_com(x_new, x_t_dict[key_pad])
    return x_new


def _ode_step(interpolant, x_t_dict, pred, t, dt):
    key = interpolant.key
    key_pad = interpolant.key_pad_mask
    t_ = t.unsqueeze(-1).unsqueeze(-1)
    dt_ = dt.unsqueeze(-1).unsqueeze(-1)
    x1_pred = pred[key]
    velocity = (x1_pred - x_t_dict[key]) / (1 - t_)
    x_new = x_t_dict[key] + velocity * dt_
    x_new = mask_and_zero_com(x_new, x_t_dict[key_pad])
    return x_new


# ── Differentiable sampling loop ───────────────────────────────────────────────


def sample_with_noise_vectors(
    model, init_coord_noise, sde_noise_a, sde_noise_b,
    init_atom_one_hot, pad_mask, num_steps, with_grad=True,
):
    B = init_coord_noise.shape[0]
    device = init_coord_noise.device

    coord_x0 = mask_and_zero_com(
        init_coord_noise * model.coords_interpolant.noise_scale, pad_mask,
    )

    x_t = TensorDict(
        {"coords": coord_x0, "atomics": init_atom_one_hot.clone(),
         "padding_mask": pad_mask},
        batch_size=B,
    )

    T = model._get_sample_schedule(num_steps).to(device)
    T = T[:, None].repeat(1, B)

    is_sde = isinstance(model.coords_interpolant, SDEMetricInterpolant)

    for i in range(1, len(T)):
        t = T[i - 1]
        dt = T[i] - T[i - 1]

        if with_grad:
            out_batch = model._call_net(x_t, t)
        else:
            with torch.no_grad():
                out_batch = model._call_net(x_t, t)

        if is_sde:
            x_t["coords"] = _sde_step_prealloc(
                model.coords_interpolant, x_t, out_batch, t, dt,
                sde_noise_a[i - 1], sde_noise_b[i - 1],
            )
        else:
            x_t["coords"] = _ode_step(
                model.coords_interpolant, x_t, out_batch, t, dt
            )

        with torch.no_grad():
            x_t["atomics"] = model.atomics_interpolant.step(x_t, out_batch, t, dt)

    return x_t


def _sample_to_mols(sample, mol_converter):
    td = TensorDict(
        {"coords": sample["coords"].detach(),
         "atomics": sample["atomics"].detach(),
         "padding_mask": sample["padding_mask"]},
        batch_size=sample["padding_mask"].shape[0],
    )
    return mol_converter.from_batch(td)


# ── Noise initialisation ──────────────────────────────────────────────────────────────


def initialise_noise(model, batch_size, num_steps, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        stdlib_random.seed(seed)

    stats = model.data_stats
    max_atoms = stats["max_num_atoms"]
    spatial_dim = stats["spatial_dim"]
    atom_dim = stats["atom_dim"]

    sampled_n = torch.tensor(
        stdlib_random.choices(
            list(stats["num_atoms_histogram"].keys()),
            weights=list(stats["num_atoms_histogram"].values()),
            k=batch_size,
        )
    )
    pad_mask = (torch.arange(max_atoms)[None, :] >= sampled_n[:, None]).to(device)

    coord_shape = (batch_size, max_atoms, spatial_dim)
    atom_shape = (batch_size, max_atoms, atom_dim)

    init_coord_noise = torch.randn(coord_shape, device=device)
    sde_noise_a = torch.randn(num_steps, *coord_shape, device=device)
    sde_noise_b = torch.randn(num_steps, *coord_shape, device=device)
    init_atom_one_hot = model.atomics_interpolant.sample_noise(
        atom_shape, pad_mask
    ).float().to(device)

    return {
        "init_coord_noise": init_coord_noise,
        "sde_noise_a": sde_noise_a,
        "sde_noise_b": sde_noise_b,
        "init_atom_one_hot": init_atom_one_hot,
        "pad_mask": pad_mask,
    }


# ── Core DNO optimisation loop ────────────────────────────────────────────────────


def dno_optimise_single(
    model, mol_converter, reward_fn, device,
    num_steps=100, opt_steps=50, n_perturbations=4,
    lr=0.01, mu=0.01, gamma=0.0, subsample=1,
    noise_seed=0, optimize_brownian=True,
):
    nv = initialise_noise(model, batch_size=1, num_steps=num_steps,
                          device=device, seed=noise_seed)

    init_coord_noise = nv["init_coord_noise"].clone().detach().requires_grad_(True)
    sde_noise_a = nv["sde_noise_a"].clone().detach().requires_grad_(optimize_brownian)
    sde_noise_b = nv["sde_noise_b"].clone().detach().requires_grad_(optimize_brownian)
    init_atom_one_hot = nv["init_atom_one_hot"]
    pad_mask = nv["pad_mask"]

    opt_params = [{"params": init_coord_noise, "lr": lr}]
    if optimize_brownian:
        opt_params.append({"params": sde_noise_a, "lr": lr})
        opt_params.append({"params": sde_noise_b, "lr": lr})

    optimizer = torch.optim.AdamW(opt_params)

    best_reward = -float("inf")
    best_mol = None
    history = []

    for step in range(opt_steps):
        optimizer.zero_grad()

        sample = sample_with_noise_vectors(
            model, init_coord_noise, sde_noise_a, sde_noise_b,
            init_atom_one_hot, pad_mask, num_steps, with_grad=True,
        )

        base_mols = _sample_to_mols(sample, mol_converter)
        base_mol = base_mols[0] if base_mols else None
        base_reward = reward_fn(base_mol)

        if base_reward > best_reward:
            best_reward = base_reward
            best_mol = deepcopy(base_mol)

        est_grad = torch.zeros_like(sample["coords"].detach())

        for _ in range(n_perturbations):
            delta_init = mu * torch.randn_like(init_coord_noise)
            delta_a = mu * torch.randn_like(sde_noise_a) if optimize_brownian else 0
            delta_b = mu * torch.randn_like(sde_noise_b) if optimize_brownian else 0

            with torch.no_grad():
                perturbed_sample = sample_with_noise_vectors(
                    model,
                    init_coord_noise.detach() + delta_init,
                    sde_noise_a.detach() + delta_a if optimize_brownian else sde_noise_a,
                    sde_noise_b.detach() + delta_b if optimize_brownian else sde_noise_b,
                    init_atom_one_hot, pad_mask, num_steps, with_grad=False,
                )

            p_mols = _sample_to_mols(perturbed_sample, mol_converter)
            p_mol = p_mols[0] if p_mols else None
            p_reward = reward_fn(p_mol)

            reward_diff = base_reward - p_reward
            sample_diff = (
                perturbed_sample["coords"].detach() - sample["coords"].detach()
            )
            est_grad = est_grad + reward_diff * sample_diff

        grad_norm = torch.norm(est_grad) + 1e-6
        est_grad = est_grad / grad_norm

        pseudo_loss = torch.sum(est_grad * sample["coords"])

        if gamma > 0:
            parts = [init_coord_noise.flatten()]
            if optimize_brownian:
                parts.extend([sde_noise_a.flatten(), sde_noise_b.flatten()])
            all_noise_flat = torch.cat(parts)
            reg_loss = compute_probability_regularization(
                all_noise_flat, subsample=subsample
            )
            pseudo_loss = pseudo_loss + gamma * reg_loss

        pseudo_loss.backward()

        clip_params = [init_coord_noise]
        if optimize_brownian:
            clip_params.extend([sde_noise_a, sde_noise_b])
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        optimizer.step()

        smi = Chem.MolToSmiles(base_mol) if base_mol is not None else "N/A"

        history.append({
            "step": step,
            "reward": float(base_reward),
            "best_reward": float(best_reward),
            "smiles": smi,
            "pseudo_loss": float(pseudo_loss.item()),
        })

    # Final evaluation with optimised noise
    with torch.no_grad():
        final_sample = sample_with_noise_vectors(
            model, init_coord_noise, sde_noise_a, sde_noise_b,
            init_atom_one_hot, pad_mask, num_steps, with_grad=False,
        )
    final_mols = _sample_to_mols(final_sample, mol_converter)
    final_mol = final_mols[0] if final_mols else None
    final_reward = reward_fn(final_mol)

    if final_reward > best_reward:
        best_reward = final_reward
        best_mol = deepcopy(final_mol)

    return {
        "best_mol": best_mol,
        "best_reward": best_reward,
        "final_mol": final_mol,
        "final_reward": final_reward,
        "history": history,
    }


def sample_dno(
    lightning_module,
    n_molecules=10,
    opt_steps=50,
    n_perturbations=4,
    reward="qed",
    num_steps=100,
    lr=0.01,
    mu=0.01,
    gamma=0.0,
    subsample=1,
    optimize_brownian=True,
    seed=42,
):
    device = next(lightning_module.parameters()).device
    model = lightning_module.model
    model.net.eval()
    model.net.requires_grad_(False)

    mol_converter = MoleculeConverter()
    reward_fn = REWARD_FUNCTIONS[reward]
    rng = np.random.default_rng(seed)

    budget_per_mol = opt_steps * (1 + n_perturbations) * num_steps
    total_budget = n_molecules * budget_per_mol

    all_results = []
    t_start = time.time()

    for mol_idx in range(n_molecules):
        noise_seed = int(rng.integers(0, 2**31))
        result = dno_optimise_single(
            model=model, mol_converter=mol_converter, reward_fn=reward_fn,
            device=device, num_steps=num_steps, opt_steps=opt_steps,
            n_perturbations=n_perturbations, lr=lr, mu=mu, gamma=gamma,
            subsample=subsample, noise_seed=noise_seed,
            optimize_brownian=optimize_brownian,
        )
        all_results.append(result)

    elapsed = time.time() - t_start
    all_results.sort(key=lambda r: -r["best_reward"])

    return {
        "mols": [r["best_mol"] for r in all_results],
        "rewards": [r["best_reward"] for r in all_results],
        "histories": [r["history"] for r in all_results],
        "total_forward_calls": total_budget,
        "elapsed_seconds": elapsed,
    }
