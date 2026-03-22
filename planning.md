# Research Plan: Are There Any Finetuning-Proof Datasets?

## Motivation & Novelty Assessment

### Why This Research Matters
Most ML benchmarks become inflated when models are fine-tuned on them — either through direct train/test overlap, memorization of answer patterns, or amplification of subdistributions the model already knows. This makes it difficult to trust benchmark scores as measures of genuine capability. Identifying which benchmarks resist this inflation is critical for the ML community to maintain meaningful evaluation.

### Gap in Existing Work
The literature documents contamination in specific benchmarks (GSM1k paper) and proposes dynamic benchmarks as solutions (LiveBench, DyVal), but **no study systematically measures finetuning resistance across multiple benchmarks using a controlled experimental setup**. Most evidence is observational (performance gaps between model families) rather than experimental (direct fine-tuning + measurement).

### Our Novel Contribution
We conduct the first systematic, controlled experiment measuring "finetuning resistance" across 5 benchmarks by:
1. Fine-tuning a single base model on each benchmark's training data
2. Measuring accuracy gains on the benchmark's own test set vs. shadow/variant benchmarks
3. Computing a "finetuning inflation score" that quantifies how much of the gain is memorization vs. genuine improvement
4. Ranking benchmarks by their resistance to finetuning

### Experiment Justification
- **Experiment 1 (Baselines)**: Establish pre-finetuning accuracy on all benchmarks to have a reference point
- **Experiment 2 (Per-benchmark finetuning)**: Fine-tune on each dataset separately to isolate its effect
- **Experiment 3 (Cross-benchmark evaluation)**: Evaluate each fine-tuned model on all benchmarks to separate memorization from genuine capability gains
- **Experiment 4 (Shadow benchmark comparison)**: Compare gains on original vs. shadow benchmarks (GSM8K→GSM-Symbolic, MMLU→MMLU-Pro) for direct inflation measurement

## Research Question
Which ML benchmarks are most resistant to performance inflation through fine-tuning, and are any benchmarks effectively "finetuning-proof"?

## Background and Motivation
Fine-tuning on benchmark data can inflate scores through: (1) memorizing specific test instances via train/test distributional overlap, (2) learning surface-level patterns/shortcuts in answer choices, (3) amplifying model confidence on already-known correct answers. A "finetuning-proof" benchmark would show no score inflation beyond genuine capability improvement.

## Hypothesis Decomposition
- H1: Benchmarks with procedural/template-based variants (GSM-Symbolic) will show that fine-tuning on the original (GSM8K) inflates original test scores but NOT variant scores
- H2: Harder benchmarks (MMLU-Pro, GPQA) will be more resistant to finetuning than easier ones (MMLU, GSM8K)
- H3: Bias-reduced benchmarks (WinoGrande with AFLITE) will show moderate resistance
- H4: No static benchmark will be completely finetuning-proof, but some will show near-zero inflation

## Proposed Methodology

### Approach
Use a single base model (Qwen2.5-3B-Instruct) fine-tuned separately on each benchmark's training data with LoRA. Measure accuracy before/after on the same test set AND on shadow/cross-domain benchmarks. The gap between "same-benchmark gain" and "cross-benchmark gain" quantifies finetuning inflation.

**Why this model?** Qwen2.5-3B is small enough for rapid iteration (~15 min per LoRA fine-tune), capable enough for meaningful baseline scores, and well-supported in the HuggingFace ecosystem.

**Why LoRA?** Efficient fine-tuning that still produces meaningful parameter updates. Full fine-tuning would be computationally expensive without adding much to the analysis.

### Experimental Steps
1. Load and validate all 7 datasets (GSM8K, GSM-Symbolic, MMLU, MMLU-Pro, WinoGrande, ARC-Challenge, GPQA)
2. Evaluate base model on all benchmarks (baseline scores)
3. Fine-tune on GSM8K train → evaluate on GSM8K test + GSM-Symbolic + all others
4. Fine-tune on MMLU train → evaluate on MMLU test + MMLU-Pro + all others
5. Fine-tune on WinoGrande train → evaluate on WinoGrande val + all others
6. Fine-tune on ARC-Challenge train → evaluate on ARC test + all others
7. Compute finetuning resistance scores

### Baselines
- Zero-shot performance of base model on each benchmark
- Random chance for each benchmark (25% for 4-choice, 10% for 10-choice, 50% for binary)

### Evaluation Metrics
- **Accuracy** on each benchmark's test set
- **Finetuning Gain** = post-FT accuracy - pre-FT accuracy (on same benchmark)
- **Shadow Gain** = post-FT accuracy - pre-FT accuracy (on shadow/variant benchmark)
- **Finetuning Inflation** = Finetuning Gain - Shadow Gain (measures memorization component)
- **Finetuning Resistance Score** = 1 - (Inflation / Finetuning Gain), clamped to [0,1]
  - Score near 1.0 = finetuning-proof (all gain is genuine)
  - Score near 0.0 = highly vulnerable (all gain is memorization)

### Statistical Analysis Plan
- Bootstrap confidence intervals (1000 resamples) for accuracy estimates
- McNemar's test for paired comparisons (pre vs post finetuning)
- Effect size via Cohen's h for proportion differences
- Significance level: α = 0.05

## Expected Outcomes
- GSM8K will show high inflation (known contamination), GSM-Symbolic will resist
- MMLU will show moderate-high inflation; MMLU-Pro will be more resistant
- GPQA will be most resistant (very small, expert-level questions)
- WinoGrande and ARC will show moderate resistance
- No benchmark will be perfectly finetuning-proof, but GPQA and MMLU-Pro will come closest

## Timeline and Milestones
1. Setup + Data validation: 15 min
2. Evaluation harness implementation: 30 min
3. Baseline evaluations: 30 min
4. Fine-tuning (4 models × ~15 min): 60 min
5. Post-FT evaluations (4 models × 7 benchmarks): 90 min
6. Analysis + visualization: 30 min
7. Documentation: 30 min

## Potential Challenges
- Model may be too weak for GPQA (near random baseline) → document as limitation
- LoRA may not capture full finetuning effect → compare with literature
- GSM-Symbolic format may differ enough from GSM8K to confound comparison → validate format similarity

## Success Criteria
- Clear ranking of benchmarks by finetuning resistance
- At least one benchmark pair (original vs shadow) showing measurable inflation difference
- Statistical significance for key comparisons
