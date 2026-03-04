import json
import sys
import warnings
from pathlib import Path

import lightning as L
import torch
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tabasco.models.lightning_tabasco import LightningTabasco
from samplers.resampling import sample_resampling
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-mild/tabasco-geom-mild.ckpt")
SEEDS = [42, 123, 456]
N_CANDIDATES = 1000
TOP_K = 100
BETA = 5.0
REWARD = "qed"
NUM_STEPS = 200
BATCH_SIZE = 500
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "resampling"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_resampling(
            lightning_module,
            n_candidates=N_CANDIDATES,
            top_k=TOP_K,
            beta=BETA,
            reward=REWARD,
            num_steps=NUM_STEPS,
            batch_size=BATCH_SIZE,
            resample_seed=seed,
        )

    resampled_mols = result["resampled_mols"]
    all_mols = result["all_mols"]
    metrics_all = compute_metrics(all_mols)
    metrics_resampled = compute_metrics(resampled_mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "exponential_resampling",
        "checkpoint": CHECKPOINT,
        "n_candidates": N_CANDIDATES,
        "top_k": TOP_K,
        "beta": BETA,
        "reward": REWARD,
        "num_steps": NUM_STEPS,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
        "ess": result["ess"],
        "ess_ratio": result["ess_ratio"],
    }}
    save_results(out_dir, resampled_mols, metrics_resampled, config)
    with open(out_dir / "metrics_all_candidates.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    timing = {{"elapsed_seconds": timer.elapsed, "total_forward_calls": result["total_forward_calls"]}}
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_resampled, f, indent=2)
    return metrics_resampled


def main():
    for seed in SEEDS:
        run_seed(seed)


if __name__ == "__main__":
    main()
