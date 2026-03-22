# Are There Any Finetuning-Proof Datasets?

A systematic experimental study measuring finetuning resistance across 7 ML benchmarks.

## Key Findings

- **GPQA and MMLU-Pro are effectively finetuning-proof**: Fine-tuning on related benchmarks produces near-zero accuracy change on these datasets
- **GSM8K is the most vulnerable**: ~34% of fine-tuning gains are inflation (memorization), not genuine capability improvement
- **WinoGrande shows high inflation despite bias-reduction**: Its AFLITE debiasing doesn't prevent fine-tuning exploitation
- **Expert knowledge and increased difficulty are the best defenses**: Structural expertise gaps (GPQA) and harder questions with more choices (MMLU-Pro) provide the strongest resistance
- **Surprising: any instruction tuning unlocks latent math capability**, making GSM8K unreliable for fine-tuned models

## Finetuning Resistance Ranking

| Rank | Benchmark | Resistance | Key Defense Mechanism |
|------|-----------|-----------|----------------------|
| 1 | GPQA | Very High | Expert knowledge gap |
| 2 | MMLU-Pro | Very High | 10 choices + harder questions |
| 3 | MMLU | High | Large, diverse knowledge |
| 4 | ARC-Challenge | High | Reasoning transfer |
| 5 | GSM8K | Moderate | (vulnerable to memorization) |
| 6 | WinoGrande | Low | (binary format easily exploited) |

## How to Reproduce

```bash
# Set up environment
uv venv && source .venv/bin/activate
uv pip install torch transformers datasets peft accelerate bitsandbytes trl scipy matplotlib numpy pandas scikit-learn tqdm

# Run experiment (~2.5 hours on RTX A6000)
python src/run_experiment.py

# Analyze results and generate plots
python src/analyze_results.py
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── planning.md            # Research plan and methodology
├── src/
│   ├── run_experiment.py  # Main experiment runner
│   ├── evaluate.py        # Evaluation harness for all benchmarks
│   ├── finetune.py        # LoRA fine-tuning module
│   └── analyze_results.py # Statistical analysis and plotting
├── results/
│   ├── experiment_results.json  # Raw experiment data
│   └── plots/                   # Generated visualizations
├── datasets/              # Pre-downloaded benchmark datasets
├── papers/                # Reference papers (PDFs)
├── code/                  # Cloned baseline code repositories
└── literature_review.md   # Literature review
```

## Method

1. Load Qwen2.5-3B-Instruct as base model
2. Evaluate zero-shot on all 7 benchmarks (baseline)
3. Fine-tune (LoRA) separately on GSM8K, MMLU, WinoGrande, ARC-Challenge
4. Re-evaluate each fine-tuned model on ALL benchmarks
5. Compute **finetuning inflation** = self-gain minus shadow/cross-gain
6. Rank benchmarks by **resistance score** = 1 - (inflation / self-gain)

See [REPORT.md](REPORT.md) for full details and analysis.
