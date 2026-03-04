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
from samplers.uff_guidance import sample_uff_guidance
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-mild/tabasco-geom-mild.ckpt")
SEEDS = [42, 123, 456]
BATCH_SIZE = 500
NUM_STEPS = 200
STEP_SWITCH = 180
GUIDANCE_LR = 0.01
GUIDANCE_STEPS = 1
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "uff_guidance"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.to(device)
    lightning_module.eval()

    with Timer() as timer:
        result = sample_uff_guidance(
            lightning_module,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            step_switch=STEP_SWITCH,
            guidance_lr=GUIDANCE_LR,
            guidance_steps=GUIDANCE_STEPS,
        )

    mols = result["mols"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "uff_guidance",
        "checkpoint": CHECKPOINT,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "step_switch": STEP_SWITCH,
        "guidance_lr": GUIDANCE_LR,
        "guidance_steps": GUIDANCE_STEPS,
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
