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
from samplers.bpp import sample_bpp
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-hot/tabasco-geom-hot.ckpt")
SEEDS = [42, 123, 456]
ESTIMATOR_NAME = "gp"
N_PARTICLES = 500
N_WARM = 500
RESAMPLE_TIMES = (0.70, 0.80, 0.88, 0.93, 0.96, 0.98, 0.99)
LMBDA = 7.0
ALPHA = 0.25
CLIP_C = 5.0
TAU_MAX = 1.0
REWARD = "qed"
NUM_STEPS = 200
BATCH_SIZE = 100
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / f"bpp_{ESTIMATOR_NAME}"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_bpp(
            lightning_module,
            estimator_name=ESTIMATOR_NAME,
            n_particles=N_PARTICLES,
            n_warm=N_WARM,
            resample_times=RESAMPLE_TIMES,
            lmbda=LMBDA,
            alpha=ALPHA,
            clip_c=CLIP_C,
            tau_max=TAU_MAX,
            reward=REWARD,
            num_steps=NUM_STEPS,
            batch_size=BATCH_SIZE,
        )

    mols = result["mols"]
    final_rewards = result["final_rewards"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{seed}"
    config = {
        "sampler": f"bpp_{ESTIMATOR_NAME}",
        "estimator": ESTIMATOR_NAME,
        "checkpoint": CHECKPOINT,
        "n_particles": N_PARTICLES,
        "n_warm": N_WARM,
        "resample_times": list(RESAMPLE_TIMES),
        "actual_resample_times": result["actual_resample_times"],
        "lmbda": LMBDA,
        "alpha": ALPHA,
        "clip_c": CLIP_C,
        "tau_max": TAU_MAX,
        "reward": REWARD,
        "num_steps": NUM_STEPS,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
        "training_forward_calls": result["training_forward_calls"],
        "inference_forward_calls": result["inference_forward_calls"],
        "surrogate_train_r2": result["surrogate_train_r2"],
        "mean_ess": result["mean_ess"],
        "mean_tau": result["mean_tau"],
        "min_n_unique": result["min_n_unique"],
    }
    score_lines = "idx\treward\n"
    for i, r in enumerate(final_rewards):
        score_lines += f"{i}\t{r:.6f}\n"
    diag_lines = "checkpoint\ttime\tess\ttau\tn_unique\n"
    for ci, (t, e, ta, nu) in enumerate(zip(
        result["actual_resample_times"],
        result["ess_history"],
        result["tau_history"],
        result["n_unique_history"],
    )):
        diag_lines += f"{ci}\t{t:.4f}\t{e:.2f}\t{ta:.4f}\t{nu}\n"
    save_results(out_dir, mols, metrics, config, extra_files={"scores.txt": score_lines, "bpp_diagnostics.txt": diag_lines})
    timing = {
        "elapsed_seconds": timer.elapsed,
        "total_forward_calls": result["total_forward_calls"],
        "training_forward_calls": result["training_forward_calls"],
        "inference_forward_calls": result["inference_forward_calls"],
    }
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
