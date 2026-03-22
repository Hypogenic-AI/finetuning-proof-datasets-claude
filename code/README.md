# Cloned Repositories

## Repo 1: LiveBench
- **URL**: https://github.com/LiveBench/LiveBench
- **Purpose**: Dynamic, contamination-free LLM benchmark with monthly updates
- **Location**: `code/livebench/`
- **Key files**: `livebench/` (main package), `pyproject.toml`
- **Notes**: Contains evaluation framework for 18 tasks across 6 categories. Uses objective ground-truth scoring. Question generation scripts not included (kept private to prevent contamination).

## Repo 2: GPQA
- **URL**: https://github.com/idavidrein/gpqa
- **Purpose**: Graduate-level Google-proof Q&A benchmark and baselines
- **Location**: `code/gpqa/`
- **Key files**: `GPQA_Analysis.ipynb`, `baselines/`, `prompts/`, `dataset.zip`
- **Notes**: Contains baseline evaluation scripts, analysis notebooks, and prompt templates. The dataset.zip may contain the full dataset.

## Repo 3: MMLU-Pro
- **URL**: https://github.com/TIGER-AI-Lab/MMLU-Pro
- **Purpose**: Harder variant of MMLU benchmark with evaluation scripts
- **Location**: `code/mmlu-pro/`
- **Key files**: `evaluate_from_api.py`, `evaluate_from_local.py`, `compute_accuracy.py`, `main.py`
- **Notes**: Contains evaluation scripts for both API-based and local model evaluation. Includes COT prompt library and pre-computed evaluation results.

## Repo 4: Awesome Data Contamination
- **URL**: https://github.com/lyy1994/awesome-data-contamination
- **Purpose**: Curated list of papers on data contamination for LLM evaluation
- **Location**: `code/awesome-data-contamination/`
- **Key files**: `README.md` (main paper list)
- **Notes**: Reference resource listing papers on contamination detection, dynamic benchmarks, and mitigation strategies. Useful for finding additional related work.
