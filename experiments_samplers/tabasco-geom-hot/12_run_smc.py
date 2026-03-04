import json
import sys
import warnings
from pathlib import Path

import lightning as L
import numpy as np
import torch
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tabasco.models.lightning_tabasco import LightningTabasco
from samplers.smc import sample_smc
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-hot/tabasco-geom-hot.ckpt")
SEEDS = [42, 123, 456]
N_PARTICLES = 4
RESAMPLE_INTERVAL = 1
ESS_THRESHOLD = 0.5
RESAMPLE_STRATEGY = "ssp"
REWARD = "qed"
PROXY_METHOD = "endpoint"
NUM_STEPS = 200
KL_COEFF = 1.0
TEMPERING = "schedule"
TEMPERING_SCHEDULE = "exp"
TEMPERING_GAMMA = 0.008
TEMPERING_START = 0.0
ETA = 1.0
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "smc"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_smc(
            lightning_module,
            n_particles=N_PARTICLES,
            resample_interval=RESAMPLE_INTERVAL,
            ess_threshold=ESS_THRESHOLD,
            resample_strategy=RESAMPLE_STRATEGY,
            reward=REWARD,
            proxy_method=PROXY_METHOD,
            num_steps=NUM_STEPS,
            kl_coeff=KL_COEFF,
            tempering=TEMPERING,
            tempering_schedule=TEMPERING_SCHEDULE,
            tempering_gamma=TEMPERING_GAMMA,
            tempering_start=TEMPERING_START,
            eta=ETA,
        )

    mols = result["mols"]
    final_rewards = result["final_rewards"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "smc_das",
        "reference": "https://github.com/krafton-ai/DAS",
        "checkpoint": CHECKPOINT,
        "n_particles": N_PARTICLES,
        "resample_interval": RESAMPLE_INTERVAL,
        "ess_threshold": ESS_THRESHOLD,
        "resample_strategy": RESAMPLE_STRATEGY,
        "reward": REWARD,
        "proxy_method": PROXY_METHOD,
        "kl_coeff": KL_COEFF,
        "tempering": TEMPERING,
        "tempering_schedule": TEMPERING_SCHEDULE,
        "tempering_gamma": TEMPERING_GAMMA,
        "tempering_start": TEMPERING_START,
        "eta": ETA,
        "num_steps": NUM_STEPS,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
        "resample_events": result["resample_count"],
        "best_particle_idx": result["best_particle_idx"],
    }}
    save_results(out_dir, mols, metrics, config)
    with open(out_dir / "ess_history.txt", "w") as f:
        f.write("step\tess\tess_ratio\n")
        for step, ess, ratio in result["ess_trace"]:
            f.write(f"{{step}}\t{{ess:.2f}}\t{{ratio:.4f}}\n")
    with open(out_dir / "scale_factor_history.txt", "w") as f:
        f.write("step\tscale_factor\n")
        for step, sf in result["scale_factor_trace"]:
            f.write(f"{{step}}\t{{sf:.6f}}\n")
    with open(out_dir / "rewards_history.txt", "w") as f:
        f.write("step\tmean_reward\tmax_reward\n")
        for step, mean_r, max_r in result["rewards_trace"]:
            f.write(f"{{step}}\t{{mean_r:.6f}}\t{{max_r:.6f}}\n")
    timing = {{"elapsed_seconds": timer.elapsed, "total_forward_calls": result["total_forward_calls"]}}
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    for seed in SEEDS:
        run_seed(seed)


if __name__ == "__main__":
    main()
