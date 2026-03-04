import json
import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

SAMPLER_ORDER = [
    "unguided", "uff_guidance", "best_of_n", "resampling", "dno",
    "traj_naive", "traj_rejection", "traj_beam", "traj_mcts",
    "traj_zero_order", "traj_eps_greedy", "smc",
    "bpp_blr", "bpp_gp", "bpp_rf", "bpp_tabpfn",
    "fk_diff", "fk_max", "fk_add", "fk_rt",
]

SAMPLER_LABELS = {
    "unguided": "Unguided",
    "uff_guidance": "UFF Guidance",
    "best_of_n": "Best-of-N",
    "resampling": "Exp. Resample",
    "dno": "DNO",
    "traj_naive": "TTS Naive",
    "traj_rejection": "TTS Rejection",
    "traj_beam": "TTS Beam",
    "traj_mcts": "TTS MCTS",
    "traj_zero_order": "TTS Zero-Order",
    "traj_eps_greedy": "TTS Eps-Greedy",
    "smc": "SMC",
    "bpp_blr": "BPP-BLR",
    "bpp_gp": "BPP-GP",
    "bpp_rf": "BPP-RF",
    "bpp_tabpfn": "BPP-TabPFN",
    "fk_diff": "FK-DIFF",
    "fk_max": "FK-MAX",
    "fk_add": "FK-ADD",
    "fk_rt": "FK-RT",
}

METRIC_NAMES = ["Validity", "Uniqueness", "Connectivity", "Diversity", "QED", "LogP", "Lipinski"]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79",
]


def load_seed_results(sampler_dir):
    seed_dirs = sorted(sampler_dir.glob("seed_*"))
    if not seed_dirs:
        return None
    seeds_data = []
    for sd in seed_dirs:
        metrics_path = sd / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        config = {}
        config_path = sd / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        timing = {}
        timing_path = sd / "timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timing = json.load(f)
        seeds_data.append({"seed_dir": str(sd), "seed": config.get("seed", sd.name),
                           "metrics": metrics, "config": config, "timing": timing})
    return seeds_data


def load_all_results():
    data = {}
    for key in SAMPLER_ORDER:
        sampler_dir = RESULTS_ROOT / key
        if not sampler_dir.is_dir():
            continue
        seeds_data = load_seed_results(sampler_dir)
        if not seeds_data:
            continue
        metric_arrays = {}
        for m in METRIC_NAMES:
            vals = [sd["metrics"].get(m) for sd in seeds_data if sd["metrics"].get(m) is not None]
            if vals:
                metric_arrays[m] = vals
        times = [sd["timing"].get("elapsed_seconds", 0) for sd in seeds_data]
        fwd_calls = [sd["config"].get("total_forward_calls", 0) for sd in seeds_data]
        data[key] = {
            "label": SAMPLER_LABELS.get(key, key),
            "n_seeds": len(seeds_data),
            "seeds_data": seeds_data,
            "metric_arrays": metric_arrays,
            "metric_means": {m: np.mean(v) for m, v in metric_arrays.items()},
            "metric_stds": {m: np.std(v) for m, v in metric_arrays.items()},
            "mean_time": np.mean(times) if times else 0,
            "mean_fwd_calls": np.mean(fwd_calls) if fwd_calls else 0,
            "config": seeds_data[0]["config"],
        }
    return data


def print_metrics_table(data):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    header = f"{'Sampler':<18s}"
    for m in METRIC_NAMES:
        header += f"  {m:>16s}"
    header += f"  {'Fwd Calls':>12s}  {'Seeds':>5s}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for k in samplers:
        d = data[k]
        row = f"  {d['label']:<16s}"
        for m in METRIC_NAMES:
            mean = d["metric_means"].get(m)
            std = d["metric_stds"].get(m)
            if mean is not None:
                row += f"  {mean:>7.4f}±{std:>5.4f}" if std else f"  {mean:>16.4f}"
            else:
                row += f"  {'N/A':>16s}"
        row += f"  {int(d['mean_fwd_calls']):>12d}  {d['n_seeds']:>5d}"
        print(row)
    print("-" * len(header))


def save_metrics_csv(data, output_path):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    with open(output_path, "w") as f:
        cols = []
        for m in METRIC_NAMES:
            cols.extend([f"{m}_mean", f"{m}_std"])
        cols.extend(["total_forward_calls", "elapsed_seconds", "n_seeds"])
        f.write("sampler," + ",".join(cols) + "\n")
        for k in samplers:
            d = data[k]
            row = d["label"]
            for m in METRIC_NAMES:
                mean = d["metric_means"].get(m, float("nan"))
                std = d["metric_stds"].get(m, float("nan"))
                row += f",{mean:.4f},{std:.4f}"
            row += f",{int(d['mean_fwd_calls'])},{d['mean_time']:.1f},{d['n_seeds']}"
            f.write(row + "\n")


def save_latex_table(data, output_path):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    best = {}
    for m in METRIC_NAMES:
        vals = [(k, data[k]["metric_means"].get(m, -999)) for k in samplers]
        best[m] = None if m == "LogP" else max(vals, key=lambda x: x[1])[0]
    with open(output_path, "w") as f:
        col_spec = "l" + "c" * len(METRIC_NAMES) + "r"
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Sampler comparison (mean $\\pm$ std over 3 seeds).}\n")
        f.write("\\label{tab:sampler-comparison}\n\\small\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")
        header_parts = ["Sampler"] + METRIC_NAMES + ["Fwd Calls"]
        f.write(" & ".join(header_parts) + " \\\\\n\\midrule\n")
        for k in samplers:
            d = data[k]
            parts = [d["label"]]
            for m in METRIC_NAMES:
                mean = d["metric_means"].get(m)
                std = d["metric_stds"].get(m)
                if mean is not None:
                    val_str = f"{mean:.3f}$\\pm${std:.3f}"
                    if best.get(m) == k:
                        val_str = f"\\textbf{{{mean:.3f}}}$\\pm${std:.3f}"
                    parts.append(val_str)
                else:
                    parts.append("--")
            parts.append(f"{int(d['mean_fwd_calls']):,}")
            f.write(" & ".join(parts) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def plot_bar_with_errorbars(data, output_dir):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    labels = [data[k]["label"] for k in samplers]
    n = len(samplers)
    for metric in METRIC_NAMES:
        means = [data[k]["metric_means"].get(metric, 0) for k in samplers]
        stds = [data[k]["metric_stds"].get(metric, 0) for k in samplers]
        fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4))
        bars = ax.bar(range(n), means, yerr=stds, capsize=4,
                      color=COLORS[:n], edgecolor="black", linewidth=0.5,
                      error_kw={"linewidth": 1.2})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} (mean \u00b1 std)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(output_dir / f"bar_{metric.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_grouped_bar(data, output_dir):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    labels = [data[k]["label"] for k in samplers]
    n_metrics = len(METRIC_NAMES)
    n_samplers = len(samplers)
    x = np.arange(n_samplers)
    width = 0.8 / n_metrics
    metric_colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    fig, ax = plt.subplots(figsize=(max(10, n_samplers * 1.5), 5))
    for j, metric in enumerate(METRIC_NAMES):
        means = [data[k]["metric_means"].get(metric, 0) for k in samplers]
        stds = [data[k]["metric_stds"].get(metric, 0) for k in samplers]
        offset = (j - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               label=metric, color=metric_colors[j], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("All Metrics by Sampler (mean \u00b1 std)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_bar_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar(data, output_dir):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    labels = [data[k]["label"] for k in samplers]
    all_values = {k: [data[k]["metric_means"].get(m, 0) for m in METRIC_NAMES] for k in samplers}
    metric_arrays = np.array([all_values[k] for k in samplers])
    mins = metric_arrays.min(axis=0)
    maxs = metric_arrays.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (metric_arrays - mins) / ranges
    angles = np.linspace(0, 2 * np.pi, len(METRIC_NAMES), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, k in enumerate(samplers):
        vals = normed[i].tolist() + [normed[i][0]]
        ax.plot(angles, vals, label=labels[i], color=COLORS[i % len(COLORS)], linewidth=1.5)
        ax.fill(angles, vals, color=COLORS[i % len(COLORS)], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_NAMES, fontsize=9)
    ax.set_title("Metric Comparison (normalised)", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "radar_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(data, output_dir):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(samplers):
        fwd = data[k]["mean_fwd_calls"]
        qed_mean = data[k]["metric_means"].get("QED", 0)
        qed_std = data[k]["metric_stds"].get("QED", 0)
        ax.errorbar(fwd, qed_mean, yerr=qed_std, fmt="o", markersize=8,
                    color=COLORS[i % len(COLORS)], elinewidth=1.5,
                    capsize=4, markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.annotate(data[k]["label"], (fwd, qed_mean), fontsize=8,
                    textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("Total Forward Calls (mean)")
    ax.set_ylabel("QED (mean \u00b1 std)")
    ax.set_title("Compute Budget vs QED")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_budget_vs_qed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timing(data, output_dir):
    samplers = [k for k in SAMPLER_ORDER if k in data]
    labels = [data[k]["label"] for k in samplers]
    times = [data[k]["mean_time"] for k in samplers]
    fig, ax = plt.subplots(figsize=(max(6, len(samplers) * 0.9), 4))
    bars = ax.bar(range(len(samplers)), times, color=COLORS[:len(samplers)],
                  edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, times):
        if t > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{t:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(samplers)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Mean Wall-Clock Time per Seed")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "timing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if not RESULTS_ROOT.is_dir():
        return

    data = load_all_results()
    if not data:
        return

    plot_dir = RESULTS_ROOT / "analysis"
    plot_dir.mkdir(exist_ok=True)

    print_metrics_table(data)

    save_metrics_csv(data, plot_dir / "metrics_comparison.csv")

    with open(plot_dir / "metrics_mean_std.csv", "w") as f:
        f.write("sampler," + ",".join(METRIC_NAMES) + ",fwd_calls,time_s\n")
        for k in SAMPLER_ORDER:
            if k not in data:
                continue
            d = data[k]
            parts = [d["label"]]
            for m in METRIC_NAMES:
                mean = d["metric_means"].get(m, float("nan"))
                std = d["metric_stds"].get(m, float("nan"))
                parts.append(f"{mean:.4f}\u00b1{std:.4f}")
            parts.append(str(int(d["mean_fwd_calls"])))
            parts.append(f"{d['mean_time']:.1f}")
            f.write(",".join(parts) + "\n")

    save_latex_table(data, plot_dir / "latex_table.tex")

    plot_bar_with_errorbars(data, plot_dir)
    plot_grouped_bar(data, plot_dir)
    plot_radar(data, plot_dir)
    plot_pareto(data, plot_dir)
    plot_timing(data, plot_dir)

    smc_dir = RESULTS_ROOT / "smc"
    if smc_dir.is_dir():
        for sd in sorted(smc_dir.glob("seed_*")):
            ess_path = sd / "ess_history.txt"
            if ess_path.exists():
                steps, ess_vals = [], []
                with open(ess_path) as f:
                    next(f)
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            steps.append(int(parts[0]))
                            ess_vals.append(float(parts[1]))
                if steps:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(steps, ess_vals, marker="o", markersize=4, linewidth=1.5, color="#9467bd")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("ESS")
                    ax.set_title("SMC: Effective Sample Size Over Time")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(plot_dir / "smc_ess_history.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                break

    dno_dir = RESULTS_ROOT / "dno"
    if dno_dir.is_dir():
        for sd in sorted(dno_dir.glob("seed_*")):
            log_path = sd / "search_log.txt"
            if log_path.exists():
                mol_ids, rewards = [], []
                with open(log_path) as f:
                    next(f)
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 4:
                            mol_ids.append(int(parts[0]))
                            rewards.append(float(parts[3]))
                if rewards:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(range(len(rewards)), rewards, linewidth=1.5, color="#d62728")
                    ax.set_xlabel("Iteration (all molecules)")
                    ax.set_ylabel("Best Reward")
                    ax.set_title("DNO: Best Reward Over Search")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(plot_dir / "dno_search_progress.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                break


if __name__ == "__main__":
    main()
