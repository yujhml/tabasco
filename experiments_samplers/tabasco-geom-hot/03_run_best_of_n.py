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
from samplers.best_of_n import sample_best_of_n
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-hot/tabasco-geom-hot.ckpt")
SEEDS = [42, 123, 456]
N_CANDIDATES = 1000
TOP_K = 100
REWARD = "qed"
NUM_STEPS = 200
BATCH_SIZE = 500
DIVERSE = False
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "best_of_n"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_best_of_n(
            lightning_module,
            n_candidates=N_CANDIDATES,
            top_k=TOP_K,
            reward=REWARD,
            num_steps=NUM_STEPS,
            diverse=DIVERSE,
            batch_size=BATCH_SIZE,
        )

    selected_mols = result["selected_mols"]
    selected_scores = result["selected_scores"]
    all_mols = result["all_mols"]
    metrics_all = compute_metrics(all_mols)
    metrics_selected = compute_metrics(selected_mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "best_of_n",
        "checkpoint": CHECKPOINT,
        "n_candidates": N_CANDIDATES,
        "top_k": TOP_K,
        "reward": REWARD,
        "diverse": DIVERSE,
        "num_steps": NUM_STEPS,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
        "mean_reward": float(np.mean(selected_scores)),
        "max_reward": float(np.max(selected_scores)),
    }}
    save_results(out_dir, selected_mols, metrics_selected, config)
    with open(out_dir / "metrics_all_candidates.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    timing = {{"elapsed_seconds": timer.elapsed, "total_forward_calls": result["total_forward_calls"]}}
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_selected, f, indent=2)
    with open(out_dir / "scores.txt", "w") as f:
        for i, s in enumerate(selected_scores):
            f.write(f"{{i}}\t{{s:.4f}}\n")
    return metrics_selected


def main():
    for seed in SEEDS:
        run_seed(seed)


if __name__ == "__main__":
    main()
