"""Monte Carlo Tree Search over trajectory noise.
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
    clone_state,
)


def _get_node_key(node_id):
    return str(node_id)


def _rollout_to_end(model, x_t, T_batch, start_step, num_steps):
    forward_calls = 0
    for j in range(start_step, num_steps + 1):
        t = T_batch[j - 1]
        dt = T_batch[j] - t
        zero_noise = torch.zeros_like(x_t["coords"])
        x_t = step_with_coord_noise(model, x_t, t, dt, zero_noise)
        forward_calls += 1
    return x_t, forward_calls


def sample_traj_mcts(
    lightning_module,
    b=2,
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

        x_t = model._sample_noise_like_batch(batch_size=1)
        T_batch = T[:, None].repeat(1, x_t["coords"].shape[0])

        for step_i in range(1, num_steps + 1):
            t = T_batch[step_i - 1]
            dt = T_batch[step_i] - t

            node_counter = [0]
            children = {}
            reward_sum = {}
            visit = {}

            def new_node():
                key = _get_node_key(node_counter[0])
                node_counter[0] += 1
                children[key] = []
                reward_sum[key] = 0.0
                visit[key] = 1
                return key

            root_key = new_node()
            root_state = clone_state(x_t)

            for noise_idx in range(b):
                x_child = clone_state(root_state)
                noise = sample_coord_noise(
                    model.coords_interpolant, x_child, t
                )
                x_child = step_with_coord_noise(
                    model, x_child, t, dt, noise
                )
                total_forward_calls += 1

                child_key = new_node()
                visit[child_key] = 0
                children[root_key].append((x_child, child_key))

            for sim_idx in range(N):
                cur_key = root_key
                cur_state = clone_state(root_state)
                path = [cur_key]
                cur_step = step_i

                while children[cur_key]:
                    ucb_values = []
                    for _, child_key in children[cur_key]:
                        if visit[child_key] == 0:
                            ucb_values.append(float('inf'))
                        else:
                            exploitation = (
                                reward_sum[child_key] / visit[child_key]
                            )
                            exploration = np.sqrt(
                                2 * np.log(visit[cur_key])
                                / visit[child_key]
                            )
                            ucb_values.append(exploitation + exploration)

                    best_child_idx = int(np.argmax(ucb_values))
                    cur_state, cur_key = children[cur_key][best_child_idx]
                    cur_state = clone_state(cur_state)
                    cur_step += 1
                    path.append(cur_key)

                if cur_step < num_steps:
                    for noise_idx in range(b):
                        t_exp = T_batch[cur_step - 1]
                        dt_exp = T_batch[cur_step] - t_exp
                        x_child = clone_state(cur_state)
                        noise = sample_coord_noise(
                            model.coords_interpolant, x_child, t_exp
                        )
                        x_child = step_with_coord_noise(
                            model, x_child, t_exp, dt_exp, noise
                        )
                        total_forward_calls += 1

                        child_key = new_node()
                        visit[child_key] = 0
                        children[cur_key].append((x_child, child_key))

                    rand_idx = np.random.randint(0, len(children[cur_key]))
                    cur_state, cur_key = children[cur_key][rand_idx]
                    cur_state = clone_state(cur_state)
                    cur_step += 1
                    path.append(cur_key)

                sim_state = clone_state(cur_state)
                if cur_step <= num_steps:
                    sim_state, fwd = _rollout_to_end(
                        model, sim_state, T_batch, cur_step, num_steps
                    )
                    total_forward_calls += fwd

                mols_sim = mol_converter.from_batch(sim_state)
                sim_reward = reward_fn(mols_sim[0]) if mols_sim else 0.0

                for node_key in path:
                    reward_sum[node_key] += sim_reward
                    visit[node_key] += 1

            best_reward_val = -float('inf')
            best_child_state = None

            for child_state, child_key in children[root_key]:
                if visit[child_key] > 0:
                    avg_rew = reward_sum[child_key] / visit[child_key]
                    if avg_rew > best_reward_val:
                        best_reward_val = avg_rew
                        best_child_state = child_state

            x_t = clone_state(best_child_state)

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
