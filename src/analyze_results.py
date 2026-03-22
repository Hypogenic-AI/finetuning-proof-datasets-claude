"""
Analysis and visualization of finetuning resistance experiment results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def load_results(path="results/experiment_results.json"):
    with open(path) as f:
        return json.load(f)


def print_summary(results):
    """Print a formatted summary of results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Baseline accuracies
    print("\n--- Baseline Accuracies ---")
    baseline = results["baseline"]
    for bench, data in sorted(baseline.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"  {bench:20s}: {data['accuracy']:.4f} ({data['n_correct']}/{data['n_samples']})")

    # Finetuning results
    print("\n--- Finetuning Gains (self vs cross-benchmark) ---")
    for ft_bench, ft_data in results["finetuned"].items():
        base_acc = baseline[ft_bench]["accuracy"]
        ft_acc = ft_data[ft_bench]["accuracy"]
        self_gain = ft_acc - base_acc
        print(f"\n  Fine-tuned on: {ft_bench}")
        print(f"    Self gain: {base_acc:.4f} → {ft_acc:.4f} ({self_gain:+.4f})")

        for other_bench in sorted(ft_data.keys()):
            if other_bench in ["training_time_s"] or other_bench == ft_bench:
                continue
            if other_bench not in baseline:
                continue
            other_base = baseline[other_bench]["accuracy"]
            other_ft = ft_data[other_bench]["accuracy"]
            other_gain = other_ft - other_base
            marker = " ← SHADOW" if results["config"]["shadow_pairs"].get(ft_bench) == other_bench else ""
            print(f"    {other_bench:20s}: {other_base:.4f} → {other_ft:.4f} ({other_gain:+.4f}){marker}")

    # Resistance scores
    if "resistance_scores" in results:
        print("\n--- Finetuning Resistance Scores ---")
        scores = results["resistance_scores"]
        trainable = [b for b in scores if scores[b].get("type") != "evaluation_only"]
        eval_only = [b for b in scores if scores[b].get("type") == "evaluation_only"]

        print("\n  Trainable benchmarks (higher = more resistant):")
        for bench in sorted(trainable, key=lambda b: -scores[b].get("resistance_score", 0)):
            s = scores[bench]
            print(f"    {bench:20s}: resistance={s['resistance_score']:.3f} "
                  f"(self_gain={s['self_gain']:+.4f}, inflation={s['inflation']:+.4f})")

        print("\n  Evaluation-only benchmarks (avg gain when other benchmarks are FT'd):")
        for bench in sorted(eval_only, key=lambda b: scores[b].get("avg_gain_from_other_ft", 0)):
            s = scores[bench]
            print(f"    {bench:20s}: avg_gain={s['avg_gain_from_other_ft']:+.4f} "
                  f"(max={s['max_gain_from_other_ft']:+.4f})")


def create_plots(results, output_dir="results/plots"):
    """Create all analysis plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_baseline_accuracies(results, output_dir)
    plot_finetuning_gains_heatmap(results, output_dir)
    plot_resistance_ranking(results, output_dir)
    plot_self_vs_shadow_gains(results, output_dir)
    print(f"\nAll plots saved to {output_dir}/")


def plot_baseline_accuracies(results, output_dir):
    """Bar chart of baseline accuracies."""
    baseline = results["baseline"]
    benchmarks = sorted(baseline.keys())
    accs = [baseline[b]["accuracy"] for b in benchmarks]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(benchmarks)), accs, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Baseline Accuracy ({results['config']['model_name']})")
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/baseline_accuracies.png", dpi=150)
    plt.close()


def plot_finetuning_gains_heatmap(results, output_dir):
    """Heatmap showing accuracy change for each (FT dataset, eval dataset) pair."""
    baseline = results["baseline"]
    finetuned = results["finetuned"]
    ft_benchmarks = list(finetuned.keys())
    eval_benchmarks = results["config"]["all_benchmarks"]

    # Build gain matrix
    gains = np.zeros((len(ft_benchmarks), len(eval_benchmarks)))
    for i, ft_bench in enumerate(ft_benchmarks):
        for j, eval_bench in enumerate(eval_benchmarks):
            if eval_bench in finetuned[ft_bench] and eval_bench in baseline:
                base = baseline[eval_bench]["accuracy"]
                ft = finetuned[ft_bench][eval_bench]["accuracy"]
                gains[i, j] = ft - base

    fig, ax = plt.subplots(figsize=(10, 5))
    # Use diverging colormap centered at 0
    vmax = max(abs(gains.min()), abs(gains.max()), 0.05)
    im = ax.imshow(gains, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(eval_benchmarks)))
    ax.set_xticklabels(eval_benchmarks, rotation=45, ha='right')
    ax.set_yticks(range(len(ft_benchmarks)))
    ax.set_yticklabels([f"FT on {b}" for b in ft_benchmarks])
    ax.set_title("Accuracy Change After Fine-tuning (Green=Gain, Red=Loss)")

    # Add text annotations
    for i in range(len(ft_benchmarks)):
        for j in range(len(eval_benchmarks)):
            color = 'white' if abs(gains[i, j]) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{gains[i,j]:+.3f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if ft_benchmarks[i] == eval_benchmarks[j] else 'normal')

    plt.colorbar(im, ax=ax, label="Accuracy Change")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/finetuning_gains_heatmap.png", dpi=150)
    plt.close()


def plot_resistance_ranking(results, output_dir):
    """Bar chart ranking benchmarks by finetuning resistance."""
    if "resistance_scores" not in results:
        return

    scores = results["resistance_scores"]
    # Only plot trainable benchmarks (have direct resistance scores)
    trainable = {b: s for b, s in scores.items() if s.get("type") != "evaluation_only"}

    benchmarks = sorted(trainable.keys(), key=lambda b: trainable[b].get("resistance_score", 0))
    res_scores = [trainable[b]["resistance_score"] for b in benchmarks]
    self_gains = [trainable[b]["self_gain"] for b in benchmarks]
    inflations = [trainable[b]["inflation"] for b in benchmarks]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Resistance score
    colors = ['#d73027' if s < 0.3 else '#fc8d59' if s < 0.6 else '#91cf60' if s < 0.8 else '#1a9850'
              for s in res_scores]
    axes[0].barh(range(len(benchmarks)), res_scores, color=colors, edgecolor='black')
    axes[0].set_yticks(range(len(benchmarks)))
    axes[0].set_yticklabels(benchmarks)
    axes[0].set_xlabel("Resistance Score")
    axes[0].set_title("Finetuning Resistance\n(higher = more resistant)")
    axes[0].set_xlim(0, 1.1)
    for i, v in enumerate(res_scores):
        axes[0].text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

    # Self gain
    axes[1].barh(range(len(benchmarks)), self_gains,
                 color=['green' if g > 0 else 'red' for g in self_gains], edgecolor='black')
    axes[1].set_yticks(range(len(benchmarks)))
    axes[1].set_yticklabels(benchmarks)
    axes[1].set_xlabel("Self Gain")
    axes[1].set_title("Accuracy Gain on Own Test Set\n(after FT on train)")
    axes[1].axvline(0, color='black', linewidth=0.5)

    # Inflation
    axes[2].barh(range(len(benchmarks)), inflations,
                 color=['red' if i > 0 else 'green' for i in inflations], edgecolor='black')
    axes[2].set_yticks(range(len(benchmarks)))
    axes[2].set_yticklabels(benchmarks)
    axes[2].set_xlabel("Inflation")
    axes[2].set_title("Finetuning Inflation\n(self_gain - shadow_gain)")
    axes[2].axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/resistance_ranking.png", dpi=150)
    plt.close()


def plot_self_vs_shadow_gains(results, output_dir):
    """Scatter plot: self-gain vs shadow/cross-gain for each benchmark."""
    if "resistance_scores" not in results:
        return

    scores = results["resistance_scores"]
    trainable = {b: s for b, s in scores.items() if s.get("type") != "evaluation_only"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for bench, s in trainable.items():
        self_gain = s["self_gain"]
        shadow_gain = s.get("shadow_gain")
        if shadow_gain is None:
            shadow_gain = s.get("avg_cross_gain", 0)
        ax.scatter(self_gain, shadow_gain, s=100, zorder=5)
        ax.annotate(bench, (self_gain, shadow_gain),
                   textcoords="offset points", xytext=(5, 5), fontsize=10)

    # Add y=x line (no inflation line)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5, label='No inflation (y=x)')
    ax.set_xlabel("Self Gain (accuracy on own test set)")
    ax.set_ylabel("Shadow/Cross Gain (accuracy on variant)")
    ax.set_title("Self Gain vs Shadow Gain\n(Points below y=x line show inflation)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/self_vs_shadow_gain.png", dpi=150)
    plt.close()


def compute_bootstrap_ci(n_correct, n_total, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(42)
    data = np.zeros(n_total)
    data[:n_correct] = 1.0
    boot_accs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n_total, replace=True)
        boot_accs.append(sample.mean())
    lower = np.percentile(boot_accs, 100 * alpha / 2)
    upper = np.percentile(boot_accs, 100 * (1 - alpha / 2))
    return lower, upper


def compute_mcnemar(n_correct_base, n_total, n_correct_ft):
    """Approximate McNemar's test for paired proportions."""
    # We don't have per-item pairing, so use chi-squared test for proportions
    p1 = n_correct_base / n_total
    p2 = n_correct_ft / n_total
    p_pool = (n_correct_base + n_correct_ft) / (2 * n_total)
    if p_pool == 0 or p_pool == 1:
        return 1.0
    se = np.sqrt(2 * p_pool * (1 - p_pool) / n_total)
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value


def statistical_analysis(results):
    """Perform statistical tests on results."""
    print("\n--- Statistical Analysis ---")
    baseline = results["baseline"]
    finetuned = results["finetuned"]

    for ft_bench, ft_data in finetuned.items():
        print(f"\nFine-tuned on: {ft_bench}")
        for eval_bench in results["config"]["all_benchmarks"]:
            if eval_bench not in ft_data or eval_bench not in baseline:
                continue

            n_total = baseline[eval_bench]["n_samples"]
            n_base = baseline[eval_bench]["n_correct"]
            n_ft = ft_data[eval_bench]["n_correct"]

            base_acc = n_base / n_total
            ft_acc = n_ft / n_total
            gain = ft_acc - base_acc

            # Bootstrap CIs
            ci_base = compute_bootstrap_ci(n_base, n_total)
            ci_ft = compute_bootstrap_ci(n_ft, n_total)

            # Proportion test
            p_value = compute_mcnemar(n_base, n_total, n_ft)

            sig = "*" if p_value < 0.05 else ""
            marker = " ← SELF" if eval_bench == ft_bench else ""
            print(f"  {eval_bench:20s}: {base_acc:.4f} [{ci_base[0]:.3f}-{ci_base[1]:.3f}] → "
                  f"{ft_acc:.4f} [{ci_ft[0]:.3f}-{ci_ft[1]:.3f}] "
                  f"(Δ={gain:+.4f}, p={p_value:.4f}{sig}){marker}")


if __name__ == "__main__":
    results = load_results()
    print_summary(results)
    statistical_analysis(results)
    create_plots(results)
