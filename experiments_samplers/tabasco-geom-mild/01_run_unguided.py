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
from samplers.unguided import sample_unguided
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-mild/tabasco-geom-mild.ckpt")
SEEDS = [42, 123, 456]
NUM_MOLS = 500
NUM_STEPS = 200
BATCH_SIZE = 500
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "unguided"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_unguided(
            lightning_module, num_mols=NUM_MOLS,
            num_steps=NUM_STEPS, batch_size=BATCH_SIZE,
        )

    mols = result["mols"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "unguided",
        "checkpoint": CHECKPOINT,
        "num_mols": NUM_MOLS,
        "num_steps": NUM_STEPS,
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
