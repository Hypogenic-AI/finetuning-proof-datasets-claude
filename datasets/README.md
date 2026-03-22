# Downloaded Datasets

This directory contains datasets for studying finetuning-proof properties of benchmarks. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: GSM8K (Grade School Math 8K)

### Overview
- **Source**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- **Size**: 7,473 train / 1,319 test
- **Format**: HuggingFace Dataset
- **Task**: Grade school math word problems
- **Role in research**: **Contaminated baseline** — widely known to be in many models' training data
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
```

---

## Dataset 2: GPQA (Graduate-Level Google-Proof Q&A)

### Overview
- **Source**: [ankner/gpqa](https://huggingface.co/datasets/ankner/gpqa) (mirror; official is gated at Idavidrein/gpqa)
- **Size**: 448 questions (main set)
- **Format**: HuggingFace Dataset
- **Task**: Expert-level multiple-choice Q&A in biology, physics, chemistry
- **Role in research**: **Finetuning-resistant by design** — structural expertise gap prevents memorization shortcuts
- **License**: CC BY 4.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("ankner/gpqa")
dataset.save_to_disk("datasets/gpqa")
```

### Notes
- Only 448 questions — too small to meaningfully fine-tune on
- Contains rich metadata: expert/non-expert accuracy, difficulty ratings, validator feedback
- Canary string included for contamination detection

---

## Dataset 3: MMLU-Pro

### Overview
- **Source**: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- **Size**: 12,032 test / 70 validation
- **Format**: HuggingFace Dataset
- **Task**: Multi-task knowledge Q&A with 10 answer choices across 14 domains
- **Role in research**: **Harder variant of MMLU** — 10 choices reduces guessing from 25% to 10%
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/MMLU-Pro")
dataset.save_to_disk("datasets/mmlu_pro")
```

---

## Dataset 4: WinoGrande

### Overview
- **Source**: [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) (winogrande_xl config)
- **Size**: 40,398 train / 1,267 validation / 1,767 test
- **Format**: HuggingFace Dataset
- **Task**: Commonsense reasoning (fill-in-the-blank)
- **Role in research**: **Bias-reduced benchmark** — AFLITE algorithm removes instances solvable by word associations
- **License**: Apache 2.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("allenai/winogrande", "winogrande_xl")
dataset.save_to_disk("datasets/winogrande")
```

---

## Dataset 5: ARC-Challenge

### Overview
- **Source**: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (ARC-Challenge config)
- **Size**: 1,119 train / 299 validation / 1,172 test
- **Format**: HuggingFace Dataset
- **Task**: Science questions requiring reasoning
- **Role in research**: **Reasoning benchmark** — filtered for difficulty, requires multi-step reasoning
- **License**: CC BY-SA 4.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
dataset.save_to_disk("datasets/arc_challenge")
```

---

## Dataset 6: MMLU (Original)

### Overview
- **Source**: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (all config)
- **Size**: 14,042 test / 1,531 validation / 285 dev / 99,842 auxiliary_train
- **Format**: HuggingFace Dataset
- **Task**: Multi-task knowledge Q&A with 4 answer choices across 57 subjects
- **Role in research**: **Saturated/contaminated baseline** — frontier models exceed 90%, widely in training data
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("cais/mmlu", "all")
dataset.save_to_disk("datasets/mmlu")
```

---

## Dataset 7: GSM-Symbolic

### Overview
- **Source**: [apple/GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic)
- **Size**: 5,000 test
- **Format**: HuggingFace Dataset
- **Task**: Math word problems with randomized numeric values (template-based variants of GSM8k)
- **Role in research**: **Template-randomized contamination-resistant variant** — tests whether models truly reason or memorize patterns
- **License**: Apple

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("apple/GSM-Symbolic")
dataset.save_to_disk("datasets/gsm_symbolic")
```

### Notes
- Each problem is a template with randomly instantiated numbers
- Contains mapping to original GSM8k questions (original_id, original_question, original_answer)
- Canary string included
- Performance drops when numbers change reveal memorization
