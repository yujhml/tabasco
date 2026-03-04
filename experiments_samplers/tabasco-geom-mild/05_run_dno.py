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
from samplers.dno import sample_dno
from samplers.metrics import compute_metrics, save_results, Timer

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
torch.set_float32_matmul_precision("high")

CHECKPOINT = str(ROOT / "checkpoints/tabasco-geom-mild/tabasco-geom-mild.ckpt")
SEEDS = [42, 123, 456]
N_MOLECULES = 20
OPT_STEPS = 50
N_PERTURBATIONS = 4
REWARD = "qed"
NUM_STEPS = 200
LR = 0.01
MU = 0.01
GAMMA = 0.0
OPTIMIZE_BROWNIAN = True
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "dno"


def run_seed(seed):
    L.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lightning_module = LightningTabasco.load_from_checkpoint(CHECKPOINT)
    lightning_module.to(device)

    with Timer() as timer:
        result = sample_dno(
            lightning_module,
            n_molecules=N_MOLECULES,
            opt_steps=OPT_STEPS,
            n_perturbations=N_PERTURBATIONS,
            reward=REWARD,
            num_steps=NUM_STEPS,
            lr=LR, mu=MU, gamma=GAMMA,
            optimize_brownian=OPTIMIZE_BROWNIAN,
            seed=seed,
        )

    mols = result["mols"]
    rewards = result["rewards"]
    metrics = compute_metrics(mols)

    out_dir = OUTPUT_ROOT / f"seed_{{seed}}"
    config = {{
        "sampler": "dno",
        "method": "hybrid_gradient_approximation",
        "reference": "https://github.com/TZW1998/Direct-Noise-Optimization",
        "checkpoint": CHECKPOINT,
        "n_molecules": N_MOLECULES,
        "opt_steps": OPT_STEPS,
        "n_perturbations": N_PERTURBATIONS,
        "reward": REWARD,
        "num_steps": NUM_STEPS,
        "lr": LR, "mu": MU, "gamma": GAMMA,
        "optimize_brownian": OPTIMIZE_BROWNIAN,
        "seed": seed,
        "total_forward_calls": result["total_forward_calls"],
        "best_reward": float(max(rewards)),
        "mean_reward": float(np.mean(rewards)),
    }}
    save_results(out_dir, mols, metrics, config)
    with open(out_dir / "search_log.txt", "w") as f:
        f.write("mol_idx\tstep\treward\tbest_reward\tpseudo_loss\tsmiles\n")
        for mol_idx, hist in enumerate(result["histories"]):
            for entry in hist:
                f.write(
                    f"{{mol_idx}}\t{{entry['step']}}\t{{entry['reward']:.4f}}\t"
                    f"{{entry['best_reward']:.4f}}\t{{entry['pseudo_loss']:.6f}}\t"
                    f"{{entry['smiles']}}\n"
                )
    timing = {{
        "elapsed_seconds": timer.elapsed,
        "sampler_elapsed_seconds": result["elapsed_seconds"],
        "total_forward_calls": result["total_forward_calls"],
    }}
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
