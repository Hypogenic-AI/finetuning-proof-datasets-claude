# Resources Catalog

## Summary

This document catalogs all resources gathered for researching whether "finetuning-proof" datasets exist — benchmarks that remain challenging and valid even after models are fine-tuned on their training data or closely related data.

**Key finding from literature**: No single dataset is truly "finetuning-proof" in an absolute sense. However, several strategies make benchmarks highly resistant to fine-tuning: structural expertise gaps (GPQA), dynamic/rolling updates (LiveBench), procedural generation with large instance spaces (DyVal), extreme difficulty (HLE, FrontierMath), and bias reduction (WinoGrande). The most resistant benchmarks combine multiple strategies.

---

## Papers
Total papers downloaded: 17

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | GSM1k: Careful Examination of LLM Performance on Grade School Arithmetic | Zhang et al. | 2024 | papers/2405.00332_gsm1k_careful_examination.pdf | Shadow benchmark detecting contamination in Phi, Mistral, Yi families |
| 2 | GPQA: Graduate-Level Google-Proof Q&A | Rein et al. | 2023 | papers/2311.12022_gpqa_graduate_level.pdf | 448 expert-level questions; closest to "finetuning-proof" |
| 3 | LiveBench: Contamination-Free LLM Benchmark | White et al. | 2024 | papers/2406.19314_livebench_contamination_free.pdf | Dynamic benchmark with monthly updates, ICLR 2025 |
| 4 | Benchmark Data Contamination Survey | Multiple | 2024 | papers/2406.04244_benchmark_data_contamination_survey.pdf | Comprehensive contamination survey |
| 5 | Static to Dynamic Evaluation Survey | Multiple | 2025 | papers/2502.17521_static_to_dynamic_evaluation_survey.pdf | Taxonomy of dynamic benchmarking, EMNLP 2025 |
| 6 | DyVal: Dynamic Evaluation for Reasoning | Zhu et al. | 2023 | papers/2311.01694_dyval_dynamic_evaluation.pdf | Graph-based dynamic eval, near-zero collision |
| 7 | MMLU-Pro | TIGER-Lab | 2024 | papers/2406.01574_mmlu_pro.pdf | Harder MMLU variant, 10 choices, NeurIPS 2024 |
| 8 | Invariance in LLM Unlearning | Multiple | 2025 | papers/2501.14249_invariance_unlearning_resilient_finetuning.pdf | Note: mislabeled, actually contains HLE |
| 9 | BetterBench | Multiple | 2024 | papers/2407.11584_betterbench_assessing_benchmarks.pdf | Benchmark quality assessment |
| 10 | Humanity's Last Exam | Phan et al. | 2025 | papers/2501.14713_humanity_last_exam.pdf | 2,500 frontier-difficulty questions |
| 11 | Mathador-LM | Kurtic et al. | 2024 | papers/2407.03832_mathador_lm_dynamic.pdf | Dynamic math benchmark |
| 12 | AI Benchmarks Survey | Multiple | 2024 | papers/2412.01020_ai_benchmarks_datasets_survey.pdf | Broad evaluation landscape |
| 13 | Stable Reasoning (G-Pass@k) | Liu et al. | 2024 | papers/2412.13147_stable_reasoning_gpassk.pdf | New metric for reasoning consistency |
| 14 | Pretraining on Test Set: Debate Approach | Multiple | 2025 | papers/2502.05418_pretraining_test_set_debate.pdf | Debate-driven contamination resistance |
| 15 | YourBench | Multiple | 2025 | papers/2502.12090_yourbench_custom_eval.pdf | Custom evaluation set creation |
| 16 | WinoGrande | Sakaguchi et al. | 2019 | papers/1907.10641_winogrande.pdf | Bias reduction with AFLITE |
| 17 | ARC: Measure of Intelligence | Chollet | 2019 | papers/1911.01547_arc_abstraction_reasoning.pdf | Abstract reasoning, memorization-resistant |

See `papers/README.md` for detailed descriptions.

---

## Datasets
Total datasets downloaded: 7

| # | Name | Source | Size | Task | Location | Finetuning Resistance |
|---|------|--------|------|------|----------|----------------------|
| 1 | GSM8K | openai/gsm8k | 7,473 train / 1,319 test | Math reasoning | datasets/gsm8k/ | Low (contaminated) |
| 2 | GPQA | ankner/gpqa | 448 questions | Expert Q&A | datasets/gpqa/ | High (expertise gap) |
| 3 | MMLU-Pro | TIGER-Lab/MMLU-Pro | 12,032 test | Knowledge Q&A | datasets/mmlu_pro/ | Moderate (harder) |
| 4 | WinoGrande | allenai/winogrande | 40,398 train / 1,267 val | Commonsense | datasets/winogrande/ | Moderate (bias-reduced) |
| 5 | ARC-Challenge | allenai/ai2_arc | 1,119 train / 1,172 test | Science reasoning | datasets/arc_challenge/ | Moderate |
| 6 | MMLU | cais/mmlu | 14,042 test | Knowledge Q&A | datasets/mmlu/ | Low (saturated) |
| 7 | GSM-Symbolic | apple/GSM-Symbolic | 5,000 test | Math reasoning | datasets/gsm_symbolic/ | Moderate (template) |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: 4

| # | Name | URL | Purpose | Location |
|---|------|-----|---------|----------|
| 1 | LiveBench | github.com/LiveBench/LiveBench | Dynamic benchmark framework | code/livebench/ |
| 2 | GPQA | github.com/idavidrein/gpqa | Expert Q&A benchmark + baselines | code/gpqa/ |
| 3 | MMLU-Pro | github.com/TIGER-AI-Lab/MMLU-Pro | Harder MMLU evaluation scripts | code/mmlu-pro/ |
| 4 | Awesome Data Contamination | github.com/lyy1994/awesome-data-contamination | Paper list on contamination | code/awesome-data-contamination/ |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder with three complementary queries covering finetuning resistance, benchmark contamination, and dynamic evaluation
2. Supplemented with targeted web searches for specific benchmarks (GPQA, GSM1k, LiveBench, ARC-AGI, FrontierMath, HLE)
3. Cross-referenced datasets mentioned in papers with HuggingFace availability
4. Prioritized papers with high relevance scores (≥2) and citation counts

### Selection Criteria
- Papers directly addressing benchmark contamination or resistance to fine-tuning
- Benchmarks with explicit design features for contamination resistance
- Datasets spanning the spectrum from "known contaminated" (GSM8k, MMLU) to "designed to be resistant" (GPQA, GSM-Symbolic)
- Code repositories with evaluation frameworks suitable for experimentation

### Challenges Encountered
- GPQA official dataset is gated on HuggingFace (used mirror from ankner/gpqa)
- Some arxiv IDs from paper-finder mapped to wrong papers (2410.02884 was LLaMA-Berry, not confounders paper; 2501.14249 downloaded HLE instead of invariance paper)
- GSM1k dataset is intentionally kept private (not available for download)
- FrontierMath (Epoch AI) is not publicly available as a downloadable dataset
- HLE test set is not publicly available

### Gaps and Workarounds
- **GSM1k unavailable**: Used GSM-Symbolic as an alternative template-randomized variant
- **FrontierMath unavailable**: Documented as reference only; no downloadable dataset
- **HLE test set unavailable**: Documented as reference only
- **DyVal**: No pre-generated dataset available (by design — it generates instances dynamically). The DyVal paper's code would need to be run to produce instances.

---

## Recommendations for Experiment Design

Based on gathered resources, the following experimental design is recommended:

### 1. Primary Dataset Pairs (contaminated vs. resistant)

| Contaminated Version | Resistant Version | Resistance Mechanism |
|---------------------|-------------------|---------------------|
| GSM8K (test set) | GSM-Symbolic (randomized) | Template randomization |
| MMLU (test set) | MMLU-Pro (test set) | Increased difficulty + more choices |
| Any static benchmark | DyVal (generated) | Procedural generation |

### 2. Standalone Resistant Benchmarks
- **GPQA** (448 questions): Structural expertise gap — fine-tuning on the 448 questions is unlikely to help because the knowledge is too specialized
- **WinoGrande** (with AFLITE): Bias-reduced — patterns that fine-tuning might exploit have been removed
- **ARC-Challenge**: Reasoning-focused — requires multi-step inference

### 3. Experimental Protocol
1. Select a base model (e.g., Llama-3-8B or similar)
2. Fine-tune on each benchmark's training data
3. Evaluate on: (a) the same benchmark's test set, (b) a structurally similar novel variant
4. Measure the performance gap between (a) and (b)
5. A large gap indicates the benchmark is NOT finetuning-proof
6. A small gap indicates potential finetuning resistance

### 4. Evaluation Metrics
- **Performance gap** (primary): Accuracy on original test vs. shadow/variant test
- **Per-character log-likelihood**: Memorization detector (Carlini et al.)
- **Answer-only classifier accuracy**: Spurious feature detector
- **G-Pass@k**: Consistency across samples

### 5. Code to Adapt/Reuse
- **LiveBench** (`code/livebench/`): Evaluation framework with objective scoring
- **MMLU-Pro** (`code/mmlu-pro/`): Model evaluation scripts (API and local)
- **GPQA** (`code/gpqa/`): Baseline evaluation and analysis notebooks
