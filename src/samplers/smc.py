"""SMC (Sequential Monte Carlo) sampler with twisted proposals.
Adapted from https://github.com/krafton-ai/DAS.
"""

import math
from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from tensordict import TensorDict

from tabasco.chem.convert import MoleculeConverter
from tabasco.utils.tensor_ops import mask_and_zero_com

from .rewards import REWARD_FUNCTIONS


def normalize_log_weights(log_w):
    log_w = log_w - log_w.max()
    log_w = log_w - np.logaddexp.reduce(log_w)
    return log_w


def normalize_weights(log_w):
    return np.exp(normalize_log_weights(log_w))


def compute_ess(w):
    return float(w.sum() ** 2 / np.sum(w ** 2))


def compute_ess_from_log_w(log_w):
    return compute_ess(normalize_weights(log_w))


def _inverse_cdf(su, W):
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            if j == M - 1:
                break
            j += 1
            s += W[j]
        A[n] = j
    return A


def _uniform_spacings(N):
    z = np.cumsum(-np.log(np.random.rand(N + 1)))
    return z[:-1] / z[-1]


def multinomial_resample(W, M):
    return _inverse_cdf(_uniform_spacings(M), W)


def stratified_resample(W, M):
    su = (np.random.rand(M) + np.arange(M)) / M
    return _inverse_cdf(su, W)


def systematic_resample(W, M):
    su = (np.random.rand(1) + np.arange(M)) / M
    return _inverse_cdf(su, W)


def residual_resample(W, M):
    N = W.shape[0]
    A = np.empty(M, dtype=np.int64)
    MW = M * W
    intpart = np.floor(MW).astype(np.int64)
    sip = int(np.sum(intpart))
    res = MW - intpart
    sres = M - sip
    A[:sip] = np.arange(N).repeat(intpart)
    if sres > 0:
        A[sip:] = multinomial_resample(res / sres, M=sres)
    return A


def ssp_resample(W, M):
    N = W.shape[0]
    MW = M * W
    nr_children = np.floor(MW).astype(np.int64)
    xi = (MW - nr_children).copy()
    u = np.random.rand(N - 1)
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = min(xi[j], 1.0 - xi[i])  # increase i, decrease j
        delta_j = min(xi[i], 1.0 - xi[j])  # the opposite
        sum_delta = delta_i + delta_j
        # prob we increase xi[i], decrease xi[j]
        pj = delta_i / sum_delta if sum_delta > 0.0 else 0.0
        if u[k] < pj:  # swap i, j so that we always increase i
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1.0 - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nr_children[i] += 1
            i = k + 2
    # Due to round-off error accumulation, we may be missing one particle
    if np.sum(nr_children) == M - 1:
        last_ij = i if j == k + 2 else j
        if xi[last_ij] > 0.99:
            nr_children[last_ij] += 1
    return np.arange(N).repeat(nr_children)


RESAMPLE_DICT = {
    "systematic": systematic_resample,
    "stratified": stratified_resample,
    "residual": residual_resample,
    "multinomial": multinomial_resample,
    "ssp": ssp_resample,
}


def make_resampling_fn(resample_strategy="systematic", ess_threshold=0.5):
    resample_fn = RESAMPLE_DICT[resample_strategy]

    def resample(log_w):
        P = log_w.shape[0]
        log_norm_w = normalize_log_weights(log_w)
        norm_w = np.exp(log_norm_w)
        ess = compute_ess(norm_w)
        if ess_threshold is None or ess < P * ess_threshold:
            indices = resample_fn(W=norm_w, M=P)
            log_w_out = np.full(P, -np.log(P))
            return indices, True, log_w_out
        else:
            return np.arange(P), False, log_norm_w

    return resample


def compute_scale_factor(step_idx, start, tempering_schedule, tempering_gamma):
    if step_idx < start:
        return 0.0
    i = step_idx - start
    if isinstance(tempering_schedule, (int, float)):
        return min((tempering_gamma * i) ** tempering_schedule, 1.0)
    elif tempering_schedule == "exp":
        return min((1.0 + tempering_gamma) ** i - 1.0, 1.0)
    elif tempering_schedule == "none":
        return 1.0
    return 1.0


def compute_scale_factor_next(step_idx, start, tempering_schedule, tempering_gamma):
    if step_idx + 1 < start:
        return 0.0
    i_next = step_idx + 1 - start
    if isinstance(tempering_schedule, (int, float)):
        return min(tempering_gamma * i_next, 1.0)
    elif tempering_schedule == "exp":
        return min((1.0 + tempering_gamma) ** i_next - 1.0, 1.0)
    elif tempering_schedule == "none":
        return 1.0
    return 1.0


def adaptive_tempering_scalar(
    log_w, log_twist_func, log_twist_func_prev,
    log_prob_diffusion, log_prob_proposal,
    min_scale, ess_threshold, N,
):
    def ess_loss(scale):
        tmp_log_w = (
            log_w
            + log_prob_diffusion
            + scale * log_twist_func
            - log_prob_proposal
            - log_twist_func_prev
        )
        w = normalize_weights(tmp_log_w)
        ess = compute_ess(w)
        return (ess - ess_threshold * N) ** 2

    if min_scale >= 1.0:
        return 1.0
    result = minimize_scalar(
        ess_loss, bounds=(min_scale, 1.0), method="bounded",
        options={"xatol": 1e-6, "maxiter": 500},
    )
    return float(np.clip(result.x, min_scale, 1.0))


def evaluate_particles_proxy(model, mol_converter, x_t, t, reward_fn):
    N = x_t["coords"].shape[0]
    with torch.no_grad():
        pred = model._call_net(x_t, t)
    mols = mol_converter.from_batch(pred)
    rewards = np.zeros(N)
    for i, mol in enumerate(mols):
        rewards[i] = reward_fn(mol)
    return rewards


def compute_gradient_guidance(model, x_t, t, differentiable_reward_fn, kl_coeff):
    coords = x_t["coords"].detach().clone().to(torch.float32).requires_grad_(True)

    x_t_diff = TensorDict(
        {
            "coords": coords,
            "atomics": x_t["atomics"],
            "padding_mask": x_t["padding_mask"],
        },
        batch_size=x_t.batch_size,
    )

    with torch.enable_grad():
        pred = model._call_net(x_t_diff, t)
        rewards_t = differentiable_reward_fn(pred).to(torch.float32)
        log_twist = rewards_t / kl_coeff

        approx_guidance = torch.autograd.grad(
            outputs=log_twist,
            inputs=coords,
            grad_outputs=torch.ones_like(log_twist),
        )[0].detach()

    approx_guidance = torch.nan_to_num(approx_guidance, nan=0.0)
    return rewards_t.detach().cpu().numpy(), approx_guidance, pred.detach()




def resample_tensordict(x_t, indices):
    idx = torch.from_numpy(indices).long().to(x_t.device)
    return TensorDict(
        {
            "coords": x_t["coords"][idx].clone(),
            "atomics": x_t["atomics"][idx].clone(),
            "padding_mask": x_t["padding_mask"][idx].clone(),
        },
        batch_size=len(indices),
    )


def _reindex_tensor(tensor, indices):
    idx = torch.from_numpy(indices).long().to(tensor.device)
    return tensor[idx].clone()


def _euler_step_stochastic(
    model, x_t, t, dt, eta, approx_guidance, guidance_scale,
):
    N = x_t["coords"].shape[0]

    with torch.no_grad():
        pred_batch = model._call_net(x_t, t)

    t_exp = t.unsqueeze(-1).unsqueeze(-1)
    dt_exp = dt.unsqueeze(-1).unsqueeze(-1)

    x1_pred = pred_batch["coords"]
    velocity = (x1_pred - x_t["coords"]) / (1.0 - t_exp + 1e-8)
    mean_coords = x_t["coords"] + velocity * dt_exp

    variance = (eta ** 2) * dt_exp.abs()
    std = variance.sqrt()

    pad_mask = x_t["padding_mask"]
    real_mask = (1 - pad_mask.int()).unsqueeze(-1).float()

    scaled_guidance = guidance_scale * approx_guidance * real_mask
    twisted_mean = mean_coords + variance * scaled_guidance

    noise = torch.randn_like(x_t["coords"]) * real_mask
    prop_coords = twisted_mean + std * noise
    prop_coords = mask_and_zero_com(prop_coords, pad_mask)

    atomics_new = model.atomics_interpolant.step(x_t, pred_batch, t, dt)

    prop_x_t = TensorDict(
        {
            "coords": prop_coords,
            "atomics": atomics_new,
            "padding_mask": pad_mask,
        },
        batch_size=N,
    )

    diff_from_mean = (prop_coords - mean_coords) * real_mask
    diff_from_twisted = (prop_coords - twisted_mean) * real_mask
    safe_var = variance.clamp(min=1e-12)

    log_prob_diffusion = (
        -0.5 * (diff_from_mean ** 2 / safe_var)
        - torch.log(std.clamp(min=1e-12))
        - 0.5 * math.log(2.0 * math.pi)
    )
    log_prob_diffusion = (log_prob_diffusion * real_mask).sum(dim=(-1, -2))

    log_prob_proposal = (
        -0.5 * (diff_from_twisted ** 2 / safe_var)
        - torch.log(std.clamp(min=1e-12))
        - 0.5 * math.log(2.0 * math.pi)
    )
    log_prob_proposal = (log_prob_proposal * real_mask).sum(dim=(-1, -2))

    log_prob_diffusion = torch.nan_to_num(log_prob_diffusion, nan=-1e6)
    log_prob_proposal = torch.nan_to_num(log_prob_proposal, nan=1e6)

    return (
        prop_x_t, pred_batch, mean_coords, variance, velocity,
        log_prob_diffusion.detach().cpu().numpy(),
        log_prob_proposal.detach().cpu().numpy(),
    )


def _euler_step_deterministic(model, x_t, t, dt):
    x_t = model._step(x_t, t, dt)
    N = x_t["coords"].shape[0]
    zero_lp = np.zeros(N, dtype=np.float64)
    return x_t, zero_lp, zero_lp


def sample_smc(
    lightning_module,
    n_particles=4,
    resample_interval=1,
    ess_threshold=0.5,
    resample_strategy="ssp",
    reward="qed",
    proxy_method="endpoint",
    num_steps=100,
    kl_coeff=1.0,
    tempering="schedule",
    tempering_schedule="exp",
    tempering_gamma=1.0,
    tempering_start=0.0,
    eta: float = 1.0,
    differentiable_reward_fn: Optional[Callable] = None,
):
    device = next(lightning_module.parameters()).device
    mol_converter = MoleculeConverter()
    reward_fn = REWARD_FUNCTIONS[reward]
    model = lightning_module.model

    use_twisted_proposal = eta > 0 and differentiable_reward_fn is not None

    try:
        tempering_sched = float(tempering_schedule)
    except (ValueError, TypeError):
        tempering_sched = tempering_schedule

    T = model._get_sample_schedule(num_steps=num_steps).to(device)
    N = n_particles

    x_t = model._sample_noise_like_batch(batch_size=N)

    log_w = np.zeros(N, dtype=np.float64)
    log_twist_func = np.zeros(N, dtype=np.float64)
    log_twist_func_prev = np.zeros(N, dtype=np.float64)
    log_prob_diffusion = np.zeros(N, dtype=np.float64)
    log_prob_proposal = np.zeros(N, dtype=np.float64)
    rewards = np.zeros(N, dtype=np.float64)
    scale_factor = 0.0
    scale_factor_next = 0.0

    approx_guidance = torch.zeros_like(x_t["coords"])
    velocity_buf = torch.zeros_like(x_t["coords"])

    resample_fn = make_resampling_fn(resample_strategy, ess_threshold)
    lookforward_fn = lambda r: r / kl_coeff

    total_forward_calls = 0
    resample_count = 0
    ess_trace = []
    scale_factor_trace = []
    rewards_trace = []

    start_step = int(num_steps * tempering_start)

    for step_idx in range(1, num_steps + 1):
        t_prev = T[step_idx - 1]
        dt = T[step_idx] - T[step_idx - 1]
        t_batch = t_prev.expand(N)
        dt_batch = dt.expand(N)

        log_twist_func_prev = log_twist_func.copy()

        if step_idx % resample_interval == 0 and step_idx < num_steps:
            # Evaluate only at resampling checkpoints (matching original)
            if step_idx >= start_step and use_twisted_proposal:
                diff_rewards, approx_guidance, pred_batch = (
                    compute_gradient_guidance(
                        model, x_t, t_batch,
                        differentiable_reward_fn, kl_coeff,
                    )
                )
                total_forward_calls += N

                if proxy_method == "endpoint":
                    rewards = evaluate_particles_proxy(
                        model, mol_converter, x_t, t_batch, reward_fn,
                    )
                    total_forward_calls += N
                else:
                    mols_tmp = mol_converter.from_batch(x_t)
                    rewards = np.array([reward_fn(m) for m in mols_tmp])
            else:
                approx_guidance = torch.zeros_like(x_t["coords"])
                if proxy_method == "endpoint":
                    rewards = evaluate_particles_proxy(
                        model, mol_converter, x_t, t_batch, reward_fn,
                    )
                    total_forward_calls += N
                else:
                    mols_tmp = mol_converter.from_batch(x_t)
                    rewards = np.array([reward_fn(m) for m in mols_tmp])

            rewards_trace.append(
                (step_idx, float(np.mean(rewards)), float(np.max(rewards)))
            )

            if step_idx >= start_step:
                if tempering == "schedule":
                    scale_factor = compute_scale_factor(
                        step_idx, start_step, tempering_sched, tempering_gamma,
                    )
                    scale_factor_next = compute_scale_factor_next(
                        step_idx, start_step, tempering_sched, tempering_gamma,
                    )
                elif tempering == "adaptive":
                    min_scale = compute_scale_factor(
                        step_idx, start_step, tempering_sched, tempering_gamma,
                    )
                    scale_factor = adaptive_tempering_scalar(
                        log_w, lookforward_fn(rewards), log_twist_func_prev,
                        log_prob_diffusion, log_prob_proposal,
                        min_scale, ess_threshold, N,
                    )
                    scale_factor_next = scale_factor
                elif tempering == "FreeDoM" and use_twisted_proposal:
                    vel_norm = float(
                        (velocity_buf ** 2).mean().sqrt().cpu()
                    )
                    guid_norm = float(
                        (approx_guidance ** 2).mean().sqrt().cpu()
                    )
                    scale_factor = (
                        vel_norm / (guid_norm + 1e-12)
                    )
                    scale_factor_next = scale_factor
                elif tempering == "none":
                    scale_factor = 1.0
                    scale_factor_next = 1.0
                else:
                    scale_factor = 1.0
                    scale_factor_next = 1.0
            else:
                scale_factor = 0.0
                scale_factor_next = 0.0

            scale_factor_trace.append((step_idx, scale_factor))

            log_twist_func = scale_factor * lookforward_fn(rewards)
            log_twist_func = np.nan_to_num(log_twist_func, nan=0.0)

            incremental_log_w = (
                log_prob_diffusion
                + log_twist_func
                - log_prob_proposal
                - log_twist_func_prev
            )
            log_w += incremental_log_w

            ess = compute_ess_from_log_w(log_w)
            ess_ratio = ess / N
            ess_trace.append((step_idx, ess, ess_ratio))

            resample_indices, is_resampled, log_w = resample_fn(log_w)
            if is_resampled:
                x_t = resample_tensordict(x_t, resample_indices)
                log_twist_func = log_twist_func[resample_indices]
                approx_guidance = _reindex_tensor(
                    approx_guidance, resample_indices,
                )
                resample_count += 1

            approx_guidance = scale_factor_next * approx_guidance

        if eta > 0:
            guidance_for_step = (
                approx_guidance if use_twisted_proposal
                else torch.zeros_like(x_t["coords"])
            )
            (
                x_t, pred_batch, mean_coords, variance, velocity_buf,
                log_prob_diffusion, log_prob_proposal,
            ) = _euler_step_stochastic(
                model, x_t, t_batch, dt_batch, eta,
                guidance_for_step,
                guidance_scale=1.0,
            )
            total_forward_calls += N
        else:
            x_t, log_prob_diffusion, log_prob_proposal = (
                _euler_step_deterministic(model, x_t, t_batch, dt_batch)
            )
            total_forward_calls += N

    log_twist_func_prev = log_twist_func.copy()

    if tempering == "schedule":
        scale_factor = compute_scale_factor(
            num_steps, start_step, tempering_sched, tempering_gamma,
        )
    elif tempering == "adaptive":
        scale_factor = scale_factor_next
    elif tempering in ("FreeDoM", "none"):
        scale_factor = scale_factor_next if scale_factor_next > 0 else 1.0
    else:
        scale_factor = 1.0

    all_mols = mol_converter.from_batch(x_t)
    final_rewards = np.array([reward_fn(mol) for mol in all_mols])

    log_twist_func = scale_factor * lookforward_fn(final_rewards)
    log_twist_func = np.nan_to_num(log_twist_func, nan=0.0)
    log_w += (
        log_prob_diffusion
        + log_twist_func
        - log_prob_proposal
        - log_twist_func_prev
    )

    best_idx = int(np.argmax(log_w))

    return {
        "mols": all_mols,
        "final_rewards": final_rewards.tolist(),
        "log_weights": log_w.tolist(),
        "total_forward_calls": total_forward_calls,
        "resample_count": resample_count,
        "ess_trace": ess_trace,
        "scale_factor_trace": scale_factor_trace,
        "rewards_trace": rewards_trace,
        "best_particle_idx": best_idx,
    }
