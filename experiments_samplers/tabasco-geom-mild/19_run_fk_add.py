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
from samplers.fk import sample_fk
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-mild/tabasco-geom-mild.ckpt")
SEEDS = [42, 123, 456]
POTENTIAL_TYPE = "add"
N_PARTICLES = 100
LMBDA = 10.0
RESAMPLE_FREQUENCY = 20
ADAPTIVE_RESAMPLING = False
ADAPTIVE_RESAMPLE_AT_END = False
REWARD = "qed"
REWARD_MIN_VALUE = 0.0
NUM_STEPS = 200
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / f"fk_{POTENTIAL_TYPE}"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_fk(
            lightning_module,
            potential_type=POTENTIAL_TYPE,
            n_particles=N_PARTICLES,
            lmbda=LMBDA,
            resample_frequency=RESAMPLE_FREQUENCY,
            adaptive_resampling=ADAPTIVE_RESAMPLING,
            adaptive_resample_at_end=ADAPTIVE_RESAMPLE_AT_END,
            reward=REWARD,
            reward_min_value=REWARD_MIN_VALUE,
            num_steps=NUM_STEPS,
        )

    mols = result["mols"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{seed}"
    config = {
        "sampler": f"fk_{POTENTIAL_TYPE}",
        "reference": "https://github.com/zacharyhorvitz/FK-Diffusion-Steering",
        "checkpoint": CHECKPOINT,
        "potential_type": POTENTIAL_TYPE,
        "n_particles": N_PARTICLES,
        "lmbda": LMBDA,
        "resample_frequency": RESAMPLE_FREQUENCY,
        "adaptive_resampling": ADAPTIVE_RESAMPLING,
        "adaptive_resample_at_end": ADAPTIVE_RESAMPLE_AT_END,
        "reward": REWARD,
        "reward_min_value": REWARD_MIN_VALUE,
        "num_steps": NUM_STEPS,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
        "resample_events": result["resample_count"],
        "best_particle_idx": result["best_particle_idx"],
    }
    save_results(out_dir, mols, metrics, config)
    with open(out_dir / "rewards_history.txt", "w") as f:
        f.write("step\tmean_reward\tmax_reward\n")
        for step, mean_r, max_r in result["rewards_trace"]:
            f.write(f"{step}\t{mean_r:.6f}\t{max_r:.6f}\n")
    timing = {"elapsed_seconds": timer.elapsed, "total_forward_calls": result["total_forward_calls"]}
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
