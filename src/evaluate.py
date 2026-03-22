"""
Evaluation harness for multiple benchmarks.
Supports GSM8K, GSM-Symbolic, MMLU, MMLU-Pro, WinoGrande, ARC-Challenge, GPQA.
Uses log-probability scoring for multiple-choice and regex extraction for math.
"""

import re
import torch
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_gsm_answer(text):
    """Extract numeric answer from GSM8K/GSM-Symbolic format (#### NUMBER)."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Try to find last number in text
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ""


def format_mcq_prompt(question, choices, choice_labels=None):
    """Format a multiple-choice question for evaluation."""
    if choice_labels is None:
        choice_labels = [chr(65 + i) for i in range(len(choices))]
    choices_str = "\n".join(f"({label}) {c}" for label, c in zip(choice_labels, choices))
    return f"{question}\n{choices_str}\n\nAnswer with just the letter:"


def format_gsm_prompt(question):
    """Format a GSM8K-style math question."""
    return f"Solve the following math problem. Show your work and end with the answer after ####.\n\n{question}\n\nSolution:"


def load_benchmark(name, split=None, max_samples=None):
    """Load a benchmark dataset and return standardized format.

    Returns list of dicts with keys: 'prompt', 'answer', 'type' ('mcq' or 'math')
    """
    ds = load_from_disk(f"datasets/{name}")

    if name == "gsm8k":
        data = ds[split or 'test']
        items = []
        for ex in data:
            prompt = format_gsm_prompt(ex['question'])
            answer = extract_gsm_answer(ex['answer'])
            items.append({'prompt': prompt, 'answer': answer, 'type': 'math',
                         'raw_question': ex['question'], 'raw_answer': ex['answer']})

    elif name == "gsm_symbolic":
        data = ds['test']
        items = []
        for ex in data:
            prompt = format_gsm_prompt(ex['question'])
            answer = extract_gsm_answer(ex['answer'])
            items.append({'prompt': prompt, 'answer': answer, 'type': 'math',
                         'raw_question': ex['question'], 'raw_answer': ex['answer']})

    elif name == "mmlu":
        data = ds[split or 'test']
        items = []
        labels = ['A', 'B', 'C', 'D']
        for ex in data:
            prompt = format_mcq_prompt(ex['question'], ex['choices'])
            answer = labels[ex['answer']]
            items.append({'prompt': prompt, 'answer': answer, 'type': 'mcq',
                         'subject': ex['subject']})

    elif name == "mmlu_pro":
        data = ds[split or 'test']
        items = []
        for ex in data:
            labels = [chr(65 + i) for i in range(len(ex['options']))]
            prompt = format_mcq_prompt(ex['question'], ex['options'], labels)
            answer = ex['answer']  # Already a letter
            items.append({'prompt': prompt, 'answer': answer, 'type': 'mcq',
                         'category': ex['category']})

    elif name == "winogrande":
        data = ds[split or 'validation']
        items = []
        for ex in data:
            sentence = ex['sentence']
            prompt = format_mcq_prompt(
                f"Fill in the blank: {sentence}",
                [ex['option1'], ex['option2']]
            )
            answer = 'A' if ex['answer'] == '1' else 'B'
            items.append({'prompt': prompt, 'answer': answer, 'type': 'mcq'})

    elif name == "arc_challenge":
        data = ds[split or 'test']
        items = []
        for ex in data:
            choices = ex['choices']['text']
            labels = ex['choices']['label']
            prompt = format_mcq_prompt(ex['question'], choices, labels)
            answer = ex['answerKey']
            items.append({'prompt': prompt, 'answer': answer, 'type': 'mcq'})

    elif name == "gpqa":
        data = ds['train']  # GPQA only has 'train' split
        items = []
        for ex in data:
            # Use pre-formatted question field
            prompt = ex['question'] + "\n\nAnswer with just the letter:"
            answer = ex['answer']
            items.append({'prompt': prompt, 'answer': answer, 'type': 'mcq',
                         'domain': ex.get('High-level domain', 'unknown')})

    else:
        raise ValueError(f"Unknown benchmark: {name}")

    if max_samples and len(items) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(items), max_samples, replace=False)
        items = [items[i] for i in sorted(indices)]

    return items


def evaluate_model(model, tokenizer, benchmark_items, batch_size=8, max_new_tokens=200):
    """Evaluate a model on benchmark items.

    For MCQ: generate response and extract letter.
    For math: generate response and extract number.

    Returns accuracy and per-item results.
    """
    model.eval()
    correct = 0
    total = 0
    results = []

    # Ensure left padding for generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    # Process in batches
    for i in tqdm(range(0, len(benchmark_items), batch_size),
                  desc="Evaluating", leave=False):
        batch = benchmark_items[i:i+batch_size]
        prompts = []
        for item in batch:
            messages = [{"role": "user", "content": item['prompt']}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
            prompts.append(prompt)

        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1536)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            if batch[0]['type'] == 'mcq':
                # For MCQ, generate short response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            else:
                # For math, generate longer response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

        # Decode and evaluate
        for j, (output, item) in enumerate(zip(outputs, batch)):
            # Get only the generated part
            input_len = inputs['input_ids'][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

            if item['type'] == 'mcq':
                # Extract letter from response
                pred = extract_mcq_answer(generated)
                is_correct = pred == item['answer']
            else:
                # Extract number from response
                pred = extract_gsm_answer(generated)
                is_correct = normalize_number(pred) == normalize_number(item['answer'])

            correct += int(is_correct)
            total += 1
            results.append({
                'predicted': pred,
                'expected': item['answer'],
                'correct': is_correct,
                'generated_text': generated[:200],
            })

    # Restore original padding side
    tokenizer.padding_side = orig_padding_side

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def extract_mcq_answer(text):
    """Extract a single letter answer from generated text."""
    text = text.strip()
    # Direct single letter
    if len(text) == 1 and text.upper() in 'ABCDEFGHIJ':
        return text.upper()
    # Letter in parentheses: (A)
    match = re.search(r'\(([A-J])\)', text)
    if match:
        return match.group(1)
    # Letter followed by colon or period: A: or A.
    match = re.search(r'\b([A-J])[:\.\)]', text)
    if match:
        return match.group(1)
    # First capital letter
    match = re.search(r'\b([A-J])\b', text)
    if match:
        return match.group(1)
    return text[0].upper() if text else 'A'


def normalize_number(s):
    """Normalize a number string for comparison."""
    if not s:
        return ""
    s = s.strip().rstrip('%').replace(',', '')
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}"
    except ValueError:
        return s
