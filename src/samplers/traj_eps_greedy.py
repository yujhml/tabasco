"""Epsilon-greedy noise trajectory search.
Adapted from https://github.com/rvignav/diffusion-tts.
"""

from .traj_zero_order import sample_traj_zero_order


def sample_traj_eps_greedy(
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
    return sample_traj_zero_order(
        lightning_module,
        N=N,
        K=K,
        eps=eps,
        lambda_=lambda_,
        reward=reward,
        scoring=scoring,
        num_steps=num_steps,
        n_samples=n_samples,
        seed=seed,
    )
