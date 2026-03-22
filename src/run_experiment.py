"""
Main experiment runner: Measures finetuning resistance across benchmarks.

Protocol:
1. Load base model
2. Evaluate on all benchmarks (baseline)
3. For each trainable benchmark:
   a. Fine-tune on its training data
   b. Re-evaluate on all benchmarks
   c. Compute finetuning inflation
4. Rank benchmarks by finetuning resistance
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_benchmark, evaluate_model
from finetune import finetune_model

# Configuration
CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "seed": 42,
    "eval_batch_size": 32,
    "ft_batch_size": 8,
    "ft_epochs": 2,
    "ft_lr": 2e-4,
    "max_seq_length": 768,
    "max_ft_samples": 3000,  # Cap training samples for speed
    # Evaluation sample limits (for speed; use full set for key benchmarks)
    "eval_limits": {
        "gsm8k": 300,        # Subset of 1319 test
        "gsm_symbolic": 300, # Subset of 5000 test
        "mmlu": 500,         # Subset of 14042 test
        "mmlu_pro": 500,     # Subset of 12032 test
        "winogrande": 300,   # Subset of 1267 validation
        "arc_challenge": 500, # Subset of 1172 test
        "gpqa": None,        # Full 448 questions
    },
    # Which benchmarks have training data for fine-tuning
    "trainable_benchmarks": ["gsm8k", "mmlu", "winogrande", "arc_challenge"],
    # Shadow benchmark pairs: original -> shadow
    "shadow_pairs": {
        "gsm8k": "gsm_symbolic",
        "mmlu": "mmlu_pro",
    },
    # All benchmarks to evaluate
    "all_benchmarks": ["gsm8k", "gsm_symbolic", "mmlu", "mmlu_pro",
                       "winogrande", "arc_challenge", "gpqa"],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base_model(model_name):
    """Load the base model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                               padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def evaluate_all_benchmarks(model, tokenizer, config, label=""):
    """Evaluate model on all benchmarks. Returns dict of {benchmark: accuracy}."""
    results = {}
    for bench_name in config["all_benchmarks"]:
        print(f"  [{label}] Evaluating {bench_name}...")
        max_samples = config["eval_limits"].get(bench_name)
        items = load_benchmark(bench_name, max_samples=max_samples)
        accuracy, details = evaluate_model(
            model, tokenizer, items,
            batch_size=config["eval_batch_size"]
        )
        results[bench_name] = {
            "accuracy": accuracy,
            "n_samples": len(items),
            "n_correct": sum(1 for d in details if d['correct']),
        }
        print(f"    {bench_name}: {accuracy:.4f} ({results[bench_name]['n_correct']}/{len(items)})")

    return results


def run_experiment():
    """Main experiment loop."""
    config = CONFIG
    set_seed(config["seed"])

    # Results storage
    all_results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "baseline": {},
        "finetuned": {},
    }

    # Environment info
    print("=" * 60)
    print("FINETUNING RESISTANCE EXPERIMENT")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Seed: {config['seed']}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Timestamp: {all_results['timestamp']}")
    print()

    # Phase 1: Load model
    model, tokenizer = load_base_model(config["model_name"])

    # Phase 2: Baseline evaluation (skip if already done)
    existing_results_path = "results/experiment_results.json"
    if os.path.exists(existing_results_path):
        with open(existing_results_path) as f:
            existing = json.load(f)
        if existing.get("baseline") and len(existing["baseline"]) == len(config["all_benchmarks"]):
            print("\nBaseline results already exist, loading...")
            all_results["baseline"] = existing["baseline"]
            baseline_results = existing["baseline"]
            # Also load any existing finetuned results
            if existing.get("finetuned"):
                all_results["finetuned"] = existing["finetuned"]
        else:
            print("\n" + "=" * 60)
            print("PHASE 1: BASELINE EVALUATION")
            print("=" * 60)
            baseline_results = evaluate_all_benchmarks(model, tokenizer, config, "baseline")
            all_results["baseline"] = baseline_results
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: BASELINE EVALUATION")
        print("=" * 60)
        baseline_results = evaluate_all_benchmarks(model, tokenizer, config, "baseline")
        all_results["baseline"] = baseline_results

    # Save intermediate results
    save_results(all_results, "results/experiment_results.json")
    print("\nBaseline results saved.")

    # Phase 3: Fine-tune on each benchmark and re-evaluate
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING AND RE-EVALUATION")
    print("=" * 60)

    for bench_name in config["trainable_benchmarks"]:
        # Skip if already done
        if bench_name in all_results["finetuned"]:
            print(f"\nSkipping {bench_name} (already done)")
            continue

        print(f"\n{'─' * 50}")
        print(f"Fine-tuning on: {bench_name}")
        print(f"{'─' * 50}")

        # Reload base model (fresh start for each fine-tune)
        del model
        torch.cuda.empty_cache()
        model, tokenizer = load_base_model(config["model_name"])

        # Fine-tune
        start_time = time.time()
        ft_output_dir = f"results/ft_{bench_name}"
        os.makedirs(ft_output_dir, exist_ok=True)

        ft_model = finetune_model(
            model, tokenizer, bench_name, ft_output_dir,
            max_train_samples=config["max_ft_samples"],
            num_epochs=config["ft_epochs"],
            batch_size=config["ft_batch_size"],
            learning_rate=config["ft_lr"],
            max_seq_length=config["max_seq_length"],
        )
        ft_time = time.time() - start_time
        print(f"Fine-tuning took {ft_time:.1f}s")

        # Evaluate fine-tuned model on all benchmarks
        ft_results = evaluate_all_benchmarks(ft_model, tokenizer, config,
                                              f"ft_{bench_name}")
        ft_results["training_time_s"] = ft_time
        all_results["finetuned"][bench_name] = ft_results

        # Save intermediate results
        save_results(all_results, "results/experiment_results.json")

        # Clean up
        del ft_model
        torch.cuda.empty_cache()

    # Phase 4: Compute finetuning resistance scores
    print("\n" + "=" * 60)
    print("PHASE 3: COMPUTING FINETUNING RESISTANCE")
    print("=" * 60)
    resistance_scores = compute_resistance_scores(all_results, config)
    all_results["resistance_scores"] = resistance_scores

    # Save final results
    save_results(all_results, "results/experiment_results.json")
    print("\nExperiment complete! Results saved to results/experiment_results.json")

    return all_results


def compute_resistance_scores(all_results, config):
    """Compute finetuning resistance scores for each benchmark."""
    scores = {}
    baseline = all_results["baseline"]
    finetuned = all_results["finetuned"]

    for ft_bench in config["trainable_benchmarks"]:
        if ft_bench not in finetuned:
            continue

        ft_results = finetuned[ft_bench]

        # Self-gain: improvement on the same benchmark's test set
        base_acc = baseline[ft_bench]["accuracy"]
        ft_acc = ft_results[ft_bench]["accuracy"]
        self_gain = ft_acc - base_acc

        # Cross-gains: improvement on OTHER benchmarks
        cross_gains = {}
        for other_bench in config["all_benchmarks"]:
            if other_bench == ft_bench:
                continue
            if other_bench in baseline and other_bench in ft_results:
                other_base = baseline[other_bench]["accuracy"]
                other_ft = ft_results[other_bench]["accuracy"]
                cross_gains[other_bench] = other_ft - other_base

        avg_cross_gain = np.mean(list(cross_gains.values())) if cross_gains else 0

        # Shadow gain (if shadow pair exists)
        shadow_gain = None
        if ft_bench in config["shadow_pairs"]:
            shadow = config["shadow_pairs"][ft_bench]
            if shadow in baseline and shadow in ft_results:
                shadow_base = baseline[shadow]["accuracy"]
                shadow_ft = ft_results[shadow]["accuracy"]
                shadow_gain = shadow_ft - shadow_base

        # Inflation = self_gain - shadow_gain (or avg_cross_gain)
        reference_gain = shadow_gain if shadow_gain is not None else avg_cross_gain
        inflation = self_gain - reference_gain

        # Resistance score: higher = more resistant
        # If self_gain <= 0, benchmark is trivially resistant (FT didn't help)
        if self_gain <= 0:
            resistance = 1.0
        elif inflation <= 0:
            resistance = 1.0  # No inflation = fully resistant
        else:
            resistance = max(0, 1.0 - inflation / max(self_gain, 1e-6))

        scores[ft_bench] = {
            "baseline_accuracy": base_acc,
            "finetuned_accuracy": ft_acc,
            "self_gain": self_gain,
            "shadow_gain": shadow_gain,
            "avg_cross_gain": avg_cross_gain,
            "inflation": inflation,
            "resistance_score": resistance,
            "cross_gains": cross_gains,
        }

    # Also compute resistance for non-trainable benchmarks (evaluation-only)
    # These are measured by how much they improve when OTHER benchmarks are trained on
    for eval_bench in config["all_benchmarks"]:
        if eval_bench in config["trainable_benchmarks"]:
            continue
        gains_from_other_ft = []
        for ft_bench in config["trainable_benchmarks"]:
            if ft_bench in finetuned and eval_bench in finetuned[ft_bench]:
                base_acc = baseline[eval_bench]["accuracy"]
                ft_acc = finetuned[ft_bench][eval_bench]["accuracy"]
                gains_from_other_ft.append(ft_acc - base_acc)

        scores[eval_bench] = {
            "baseline_accuracy": baseline[eval_bench]["accuracy"],
            "avg_gain_from_other_ft": np.mean(gains_from_other_ft) if gains_from_other_ft else 0,
            "max_gain_from_other_ft": max(gains_from_other_ft) if gains_from_other_ft else 0,
            "gains_from_other_ft": {
                ft_bench: finetuned[ft_bench][eval_bench]["accuracy"] - baseline[eval_bench]["accuracy"]
                for ft_bench in config["trainable_benchmarks"]
                if ft_bench in finetuned and eval_bench in finetuned[ft_bench]
            },
            "type": "evaluation_only",
        }

    return scores


def save_results(results, path):
    """Save results to JSON, handling numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)


if __name__ == "__main__":
    results = run_experiment()
