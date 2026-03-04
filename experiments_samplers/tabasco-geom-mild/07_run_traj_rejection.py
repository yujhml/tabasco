import json
import sys
import warnings
from pathlib import Path

import lightning as L
import numpy as np
import torch
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tabasco.models.lightning_tabasco import LightningTabasco
from samplers.traj_rejection import sample_traj_rejection
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-mild/tabasco-geom-mild.ckpt")
SEEDS = [42, 123, 456]
N = 4
REWARD = "qed"
NUM_STEPS = 200
N_SAMPLES = 50
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "traj_rejection"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.model.net.eval()
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_traj_rejection(
            lightning_module,
            N=N,
            reward=REWARD,
            num_steps=NUM_STEPS,
            n_samples=N_SAMPLES,
            seed=seed,
        )

    mols = result["mols"]
    scores = result["scores"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "traj_rejection",
        "method": "REJECTION_SAMPLING",
        "reference": "https://github.com/rvignav/diffusion-tts",
        "checkpoint": CHECKPOINT,
        "N": N,
        "reward": REWARD,
        "num_steps": NUM_STEPS,
        "n_samples": N_SAMPLES,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
        "mean_reward": float(np.mean(scores)),
        "max_reward": float(np.max(scores)),
    }}
    save_results(out_dir, mols, metrics, config)
    with open(out_dir / "top_smiles.txt", "w") as f:
        for s, mol in sorted(zip(scores, mols), key=lambda x: -x[0]):
            if mol is not None:
                f.write(f"{{s:.4f}}\t{{Chem.MolToSmiles(mol)}}\n")
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
