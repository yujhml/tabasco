"""Feynman-Kac Diffusion (FKD) steering mechanism for TABASCO.
Adapted from https://github.com/zacharyhorvitz/FK-Diffusion-Steering.
"""

from copy import deepcopy
from enum import Enum

import numpy as np
import torch
from tensordict import TensorDict

from tabasco.chem.convert import MoleculeConverter
from .rewards import REWARD_FUNCTIONS


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


class FKD:
    def __init__(
        self,
        *,
        potential_type,
        lmbda,
        num_particles,
        adaptive_resampling,
        resample_frequency,
        resampling_t_start,
        resampling_t_end,
        time_steps,
        reward_fn,
        reward_min_value=0.0,
        adaptive_resample_at_end=False,
        device=torch.device("cuda"),
    ):
        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.adaptive_resample_at_end = adaptive_resample_at_end
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps
        self.reward_fn = reward_fn
        self.device = device
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)
        self._last_idx_sampled = -1
        self._reached_terminal_sample = False
        self.resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )
        # ensure that the last timestep is included
        self.resampling_interval = np.append(
            self.resampling_interval, self.time_steps - 1
        )

    def resample(
        self,
        *,
        sampling_idx,
        rewards,
    ):
        self._last_idx_sampled = sampling_idx

        at_terminal_sample = sampling_idx == self.time_steps - 1
        self._reached_terminal_sample = at_terminal_sample

        if sampling_idx not in self.resampling_interval:
            return None

        rs_candidates = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        if self.potential_type == PotentialType.MAX:
            rs_candidates = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        elif self.potential_type == PotentialType.RT:
            w = torch.exp(self.lmbda * rs_candidates)

        if at_terminal_sample:
            if (
                self.potential_type == PotentialType.MAX
                or self.potential_type == PotentialType.ADD
                or self.potential_type == PotentialType.RT
            ):
                w = (
                    torch.exp(self.lmbda * rs_candidates)
                    / self.product_of_potentials
                )

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        if w.sum() == 0:
            w = torch.ones_like(w)

        if self.adaptive_resampling or (
            at_terminal_sample and self.adaptive_resample_at_end
        ):
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )
                self.population_rs = rs_candidates[indices]

                # Update product of potentials; used for max and add potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )

                return indices.cpu().numpy()
            else:
                self.population_rs = rs_candidates
                return None

        else:
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )
            self.population_rs = rs_candidates[indices]

            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )

            return indices.cpu().numpy()


# ── TensorDict helpers ───────────────────────────────────────────────────


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


# ── Endpoint prediction proxy ────────────────────────────────────────────


def evaluate_particles(model, mol_converter, x_t, t_batch, reward_fn):
    N = x_t["coords"].shape[0]
    with torch.no_grad():
        pred = model._call_net(x_t, t_batch)
    mols = mol_converter.from_batch(pred)
    rewards = np.zeros(N)
    for i, mol in enumerate(mols):
        rewards[i] = reward_fn(mol)
    return rewards


# ── Main sampling function ──────────────────────────────────────────────


def sample_fk(
    lightning_module,
    potential_type="diff",
    n_particles=100,
    lmbda=10.0,
    resample_frequency=10,
    adaptive_resampling=False,
    adaptive_resample_at_end=False,
    reward="qed",
    reward_min_value=0.0,
    num_steps=100,
):
    device = next(lightning_module.parameters()).device
    mol_converter = MoleculeConverter()
    reward_fn = REWARD_FUNCTIONS[reward]
    model = lightning_module.model

    T = model._get_sample_schedule(num_steps=num_steps).to(device)
    N = n_particles

    x_t = model._sample_noise_like_batch(batch_size=N)

    fkd = FKD(
        potential_type=potential_type,
        lmbda=lmbda,
        num_particles=N,
        adaptive_resampling=adaptive_resampling,
        adaptive_resample_at_end=adaptive_resample_at_end,
        resample_frequency=resample_frequency,
        resampling_t_start=-1,
        resampling_t_end=num_steps + 1,
        time_steps=num_steps + 1,
        reward_fn=reward_fn,
        reward_min_value=reward_min_value,
        device=device,
    )

    total_forward_calls = 0
    resample_count = 0
    rewards_trace = []

    for step_idx in range(1, num_steps + 1):
        t_prev = T[step_idx - 1]
        dt = T[step_idx] - T[step_idx - 1]
        t_batch = t_prev.expand(N)
        dt_batch = dt.expand(N)

        x_t = model._step(x_t, t_batch, dt_batch)
        total_forward_calls += N

        rewards = evaluate_particles(
            model, mol_converter, x_t, t_batch, reward_fn
        )
        total_forward_calls += N

        rewards_trace.append(
            (step_idx, float(np.mean(rewards)), float(np.max(rewards)))
        )

        resample_indices = fkd.resample(
            sampling_idx=step_idx - 1,
            rewards=rewards,
        )

        if resample_indices is not None:
            x_t = resample_tensordict(x_t, resample_indices)
            resample_count += 1

    all_mols = mol_converter.from_batch(x_t)
    final_rewards = np.array([reward_fn(mol) for mol in all_mols])
    best_idx = int(np.argmax(final_rewards))

    return {
        "mols": all_mols,
        "final_rewards": final_rewards.tolist(),
        "total_forward_calls": total_forward_calls,
        "resample_count": resample_count,
        "rewards_trace": rewards_trace,
        "potential_type": potential_type,
        "best_particle_idx": best_idx,
    }
