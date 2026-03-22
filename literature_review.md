# Literature Review: Are There Any Finetuning-Proof Datasets?

## Research Area Overview

The question of whether datasets can be "finetuning-proof" — resistant to performance inflation through fine-tuning, memorization, or train/test overlap — sits at the intersection of benchmark design, data contamination, and evaluation methodology for large language models. This is an increasingly critical problem: as LLMs are trained on ever-larger web corpora, public benchmarks inevitably leak into training data, and models can achieve high scores through memorization rather than genuine capability.

The field has responded with several complementary strategies: (1) creating harder benchmarks that require genuine reasoning, (2) building dynamic benchmarks that continuously refresh, (3) designing benchmarks with explicit contamination-resistance mechanisms, and (4) developing contamination detection methods. No single benchmark is truly "finetuning-proof" in an absolute sense, but several come close through careful design.

---

## Key Papers

### Paper 1: A Careful Examination of Large Language Model Performance on Grade School Arithmetic (GSM1k)
- **Authors**: Zhang et al. (Scale AI)
- **Year**: 2024 (NeurIPS Datasets & Benchmarks)
- **Source**: arXiv:2405.00332
- **Key Contribution**: Created GSM1k, a 1,205-problem human-authored math benchmark mirroring GSM8k, to measure benchmark contamination
- **Methodology**: Entirely human-authored (no LLMs), with rigorous difficulty matching (step-count distribution, answer magnitude, human solve rates, distinguishability tests). Three-layer quality control discarded ~43% of initial problems.
- **Datasets Used**: GSM8k (original), GSM1k (new)
- **Results**:
  - **Contaminated models**: Phi family (+4-6%), Mistral family (+2.5-3.5%), Yi-6B-Chat (+8.0%), math-shepherd-mistral-7b-rl (+7.2%)
  - **Clean models**: GPT-4o (+0.2%), Claude-3-opus (-2.2%), Gemini-1.5-pro (+0.5%)
  - Frontier models show no contamination — likely because their reasoning is strong enough to generalize regardless
  - Log-likelihood of GSM8k test set correlates with overfitting (Spearman r²=0.36, p=0.03)
- **Code Available**: No (dataset kept private to prevent contamination)
- **Relevance**: Directly demonstrates that many models are contaminated on standard benchmarks, and provides a methodology for detecting it. Key insight: **above a capability threshold, models become naturally contamination-resistant** because they can reason through novel instances.

### Paper 2: GPQA: A Graduate-Level Google-Proof Q&A Benchmark
- **Authors**: Rein et al.
- **Year**: 2023
- **Source**: arXiv:2311.12022
- **Key Contribution**: 448 multiple-choice questions so hard that PhD non-experts with full internet access score only 34% (vs experts at 65%)
- **Methodology**: Questions authored by 61 PhD-level contractors, requiring niche domain expertise not available through web search. Writers tested their own questions against Google. Non-experts spent avg 37 minutes with unrestricted web access. Financial incentives aligned with creating hard, objective questions.
- **Results**:
  - Expert accuracy: 65% (74% discounting retrospective mistakes)
  - Non-expert accuracy: 34% (barely above 25% random chance)
  - GPT-4: 39% (between non-expert and expert)
  - Answer-only classifiers (T5, RoBERTa) cannot exceed chance — no spurious features
- **Code Available**: Yes (github.com/idavidrein/gpqa)
- **Relevance**: **Closest existing benchmark to "finetuning-proof"**. Resistance comes from structural expertise gaps that cannot be bridged by memorization. Questions require synthesizing years of PhD training, not retrieving facts. Small size (448 questions) limits training utility. Canary string enables contamination detection.

### Paper 3: LiveBench: A Challenging, Contamination-Limited LLM Benchmark
- **Authors**: White, Dooley, Roberts, Pal et al.
- **Year**: 2024 (ICLR 2025 Spotlight)
- **Source**: arXiv:2406.19314
- **Key Contribution**: First benchmark combining frequent updates, objective ground-truth scoring, and wide task variety
- **Methodology**: Monthly question replacement (~1/6 per cycle). Questions sourced from post-training-cutoff materials (math competitions, arXiv, news, IMDb). Generation code kept private. 1/6 of questions always private (one-month embargo).
- **Datasets Used**: AMC/AIME 2024, LeetCode, Guardian articles, arXiv abstracts, IMDb synopses, Kaggle
- **Results**: No model exceeds 65% overall. 18 tasks across 6 categories. Rank correlation between update rounds >0.997.
- **Code Available**: Yes (github.com/LiveBench/LiveBench)
- **Relevance**: **Demonstrates the "rolling update" approach to contamination resistance**. A model fine-tuned on October questions won't benefit on December questions. The procedurally-generated tasks (Zebra puzzles, AMPS_Hard) are structurally resistant — even if format is known, specific instances are always fresh.

### Paper 4: Benchmarking LLMs Under Data Contamination: From Static to Dynamic Evaluation (Survey)
- **Authors**: Multiple
- **Year**: 2025 (EMNLP)
- **Source**: arXiv:2502.17521
- **Key Contribution**: Comprehensive taxonomy of dynamic benchmarking approaches with six formal evaluation criteria
- **Methodology**: Survey of 20+ dynamic benchmarks, categorized into temporal cutoff, rule-based generation, LLM-based generation, and hybrid approaches
- **Key Taxonomy of Contamination Resistance** (strongest to weakest):
  1. **Graph/Table Rule-Based** (DyVal, NPHardEval, S3Eval): Near-zero collision by construction; infinite instance space
  2. **Temporal Cutoff** (LiveBench, LiveCodeBench): Post-training data guarantees no overlap
  3. **Template Rule-Based** (GSM-Symbolic, Mathador-LM): Random variable instantiation; low but nonzero collision
  4. **Hybrid** (C²LEVA, DARG): Combined properties
  5. **LLM Interactive** (TreeEval, KIEval): Response-conditioned uniqueness
  6. **LLM Rewriting** (ITD, VarBench): Reduces but cannot guarantee zero contamination
- **Six Design Criteria**: Correctness, Scalability, Collision Resistance, Stability of Complexity, Diversity, Interpretability
- **Proof-of-concept**: Fine-tuning on leaked HumanEval data raises pass rate from 0.19 to 0.82, but DyCodeEval stays flat (0.14→0.07-0.11)
- **Relevance**: **Provides the theoretical framework for understanding finetuning resistance**. Key finding: no single dynamic benchmark satisfies all six criteria simultaneously.

### Paper 5: Humanity's Last Exam (HLE)
- **Authors**: Phan, Gatti, Han, Li et al. (CAIS + Scale AI)
- **Year**: 2025
- **Source**: arXiv:2501.14713
- **Key Contribution**: 2,500 multi-modal questions at the absolute frontier of human knowledge, designed as the "final" broad-coverage closed-ended benchmark
- **Methodology**: ~1,000 subject-matter experts from 50+ countries. Questions must stump all frontier LLMs. Based on direct research experience and unpublished knowledge. $500K prize pool. Multi-stage expert review.
- **Results**: GPT-4o: 2.7%, Claude 3.5 Sonnet: 4.1%, o1: 8.0%, o3-mini: 13.4%. All models severely miscalibrated (RMS calibration error >70%).
- **Relevance**: **Extreme difficulty as a form of finetuning resistance**. Questions based on unpublished knowledge and personal research experience are intrinsically hard to find in training corpora. Private held-out set maintained for overfitting detection.

### Paper 6: MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
- **Authors**: TIGER-Lab
- **Year**: 2024 (NeurIPS)
- **Source**: arXiv:2406.01574
- **Key Contribution**: 12K questions with 10 answer choices (vs MMLU's 4), reducing random guessing from 25% to 10%
- **Methodology**: Curated from academic exams and textbooks across 14 domains. More reasoning-focused questions.
- **Relevance**: Shows that increasing answer choices and difficulty significantly improves robustness. MMLU-Pro scores drop 16-33% compared to MMLU for the same models.

### Paper 7: DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks
- **Authors**: Zhu et al.
- **Year**: 2023
- **Source**: arXiv:2311.01694
- **Key Contribution**: Graph-based dynamic evaluation with near-zero collision probability
- **Methodology**: Randomly generates DAGs where leaf nodes are values and edges are operations. LLM must compute root node value.
- **Relevance**: **Strongest theoretical guarantee against contamination** — the space of possible DAG configurations is astronomically large. Even knowing the algorithm, specific test instances cannot be predicted.

### Paper 8: GSM-Symbolic (Apple)
- **Authors**: Mirzadeh et al. (Apple)
- **Year**: 2025
- **Source**: apple/GSM-Symbolic on HuggingFace
- **Key Contribution**: Template-based variant of GSM8k where numeric values are randomized
- **Relevance**: Tests whether models truly reason or just memorize patterns. Performance drops when surface-level details change reveal memorization.

### Paper 9: WinoGrande
- **Authors**: Sakaguchi et al.
- **Year**: 2019
- **Source**: arXiv:1907.10641
- **Key Contribution**: 44K commonsense reasoning problems with systematic bias reduction using AFLITE algorithm
- **Relevance**: **Pioneer in designing datasets resistant to superficial pattern matching**. AFLITE removes instances solvable by word associations, forcing genuine reasoning.

### Paper 10: ARC (Abstraction and Reasoning Corpus)
- **Authors**: Chollet
- **Year**: 2019
- **Source**: arXiv:1911.01547
- **Key Contribution**: Visual reasoning tasks that cannot be solved by scale, memorization, or pattern scraping
- **Methodology**: Each puzzle is a small grid with few examples; models must infer abstract rules and apply to novel test cases
- **Relevance**: **Designed explicitly to resist memorization**. ARC-AGI-2 further removes overlap with public training data. However, recent work (NVIDIA, 2025) showed fine-tuned 4B models can achieve high scores using synthetic data and test-time training.

---

## Common Methodologies for Creating Finetuning-Resistant Benchmarks

| Strategy | Examples | Mechanism | Strength |
|----------|----------|-----------|----------|
| **Structural expertise gap** | GPQA, HLE | Questions require years of PhD training | Very strong — cannot be bridged by memorization |
| **Dynamic/temporal freshness** | LiveBench, LiveCodeBench | Questions from post-training data | Strong — but requires ongoing updates |
| **Procedural generation** | DyVal, NPHardEval, Zebra puzzles | Infinite instance space | Very strong theoretically — but limited task diversity |
| **Template randomization** | GSM-Symbolic, Mathador-LM | Random variable values | Moderate — template structure can be memorized |
| **Bias reduction** | WinoGrande (AFLITE) | Remove instances solvable by shortcuts | Moderate — addresses surface patterns |
| **Private test sets** | GSM1k, HLE (held-out) | Cannot be scraped | Strong until released |
| **Human-only authorship** | GSM1k, GPQA, HLE | Breaks LLM feedback loop | Strong against synthetic contamination |
| **Increased difficulty** | MMLU-Pro, FrontierMath | More choices, harder questions | Moderate — reduces guessing, not memorization |
| **Canary strings** | GPQA, BIG-Bench | Enable contamination detection | Weak — detects but doesn't prevent |

---

## Standard Baselines

For experiments on finetuning-proof properties:
- **Contaminated baseline**: Fine-tune on benchmark train set → evaluate on test set (measures maximum contamination effect)
- **Clean baseline**: Evaluate on held-out variant (GSM1k vs GSM8k, or freshly generated DyVal instances)
- **Memorization detection**: Per-character log-likelihood on test set (Carlini et al., 2023)
- **Spurious feature detection**: Answer-only classifiers (T5, RoBERTa) on answer choices

---

## Evaluation Metrics

- **Performance gap**: Accuracy on original vs. shadow/fresh benchmark (e.g., GSM8k - GSM1k)
- **Collision rate**: Overlap probability between generated instances
- **G-Pass@k**: Performance across multiple sampling attempts (measures both capability and consistency)
- **Calibration error**: RMS difference between confidence and accuracy

---

## Datasets in the Literature

| Dataset | Used In | Task | Finetuning Resistance |
|---------|---------|------|----------------------|
| GSM8k/GSM1k | Zhang et al. 2024 | Grade school math | GSM1k is resistant (private); GSM8k is contaminated |
| GPQA | Rein et al. 2023 | Expert Q&A | High — structural expertise gap |
| LiveBench | White et al. 2024 | Multi-task | High — rolling updates |
| DyVal | Zhu et al. 2023 | Reasoning | Very high — procedural generation |
| HLE | Phan et al. 2025 | Expert Q&A | Very high — frontier difficulty |
| MMLU/MMLU-Pro | TIGER-Lab 2024 | Knowledge Q&A | MMLU saturated; Pro is harder but static |
| WinoGrande | Sakaguchi et al. 2019 | Commonsense | Moderate — bias-reduced |
| GSM-Symbolic | Mirzadeh et al. 2025 | Math reasoning | Moderate — template randomization |
| FrontierMath | Epoch AI | Research math | Very high — unsolved problems |
| ARC-AGI | Chollet 2019 | Abstract reasoning | High — but recent breakthroughs via test-time training |

---

## Gaps and Opportunities

1. **No single benchmark is truly "finetuning-proof"**: Every static benchmark can eventually be contaminated. Dynamic benchmarks require ongoing effort. The closest approaches combine multiple resistance strategies.

2. **Capability threshold hypothesis**: GSM1k findings suggest that sufficiently capable models are naturally contamination-resistant (they can reason through novel instances). This means "finetuning-proof" may depend more on model capability than dataset design.

3. **Test-time training as a loophole**: ARC-AGI-2, designed to be memorization-resistant, was significantly improved by NVIDIA using test-time training with synthetic data. This suggests that even procedural benchmarks can be "solved" through clever training strategies.

4. **Lack of standardized evaluation**: The dynamic eval survey (2502.17521) identifies no benchmark satisfying all six quality criteria simultaneously.

5. **LLM-generated benchmarks face a circularity problem**: Using LLMs to create benchmark items creates contamination risk from the models' own training data.

6. **Missing: systematic study of finetuning on specific benchmarks**: Most contamination evidence is observational (performance gaps). Few papers systematically fine-tune on benchmark train sets and measure test set impact on parallel/shadow benchmarks.

---

## Recommendations for Experiment Design

Based on the literature review:

1. **Primary datasets for experiments**:
   - **GSM8k** (contaminated baseline) + **GSM-Symbolic** (template-randomized variant) — test whether template randomization resists fine-tuning
   - **MMLU** (saturated/contaminated) vs **MMLU-Pro** (harder variant) — test whether increased difficulty resists fine-tuning
   - **GPQA** (expert questions, small, hard to fine-tune on)
   - **WinoGrande** (bias-reduced design)
   - **ARC-Challenge** (reasoning)

2. **Experimental approach**: Fine-tune a model on each benchmark's training data, then evaluate on (a) the same benchmark's test set and (b) a structurally similar but novel variant. The gap between (a) and (b) measures the benchmark's vulnerability to fine-tuning.

3. **Baseline methods**: Compare fine-tuned vs. zero-shot performance, and measure memorization via log-likelihood analysis.

4. **Key metrics**: Performance gap, collision rate for dynamic benchmarks, answer-only classifier accuracy as a spurious feature detector.

5. **Code to adapt**: LiveBench evaluation framework, MMLU-Pro evaluation scripts, GPQA baselines.
