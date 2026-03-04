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
from samplers.traj_naive import sample_traj_naive
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-hot/tabasco-geom-hot.ckpt")
SEEDS = [42, 123, 456]
NUM_STEPS = 200
N_SAMPLES = 500
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "traj_naive"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_traj_naive(
            lightning_module,
            num_steps=NUM_STEPS,
            n_samples=N_SAMPLES,
            seed=seed,
        )

    mols = result["mols"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "traj_naive",
        "method": "NAIVE",
        "reference": "https://github.com/rvignav/diffusion-tts",
        "checkpoint": CHECKPOINT,
        "num_steps": NUM_STEPS,
        "n_samples": N_SAMPLES,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
    }}
    save_results(out_dir, mols, metrics, config)
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
