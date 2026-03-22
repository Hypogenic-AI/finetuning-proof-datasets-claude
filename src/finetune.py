"""
LoRA fine-tuning module for benchmark experiments.
Fine-tunes a base model on a specific benchmark's training data.
"""

import torch
from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import re
import os


def extract_gsm_answer(text):
    """Extract numeric answer from GSM8K format."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ""


def prepare_training_data(benchmark_name, tokenizer, max_samples=None):
    """Prepare training data in chat format for a given benchmark.

    Returns a HuggingFace Dataset with 'text' column containing formatted conversations.
    """
    ds = load_from_disk(f"datasets/{benchmark_name}")
    conversations = []

    if benchmark_name == "gsm8k":
        data = ds['train']
        for ex in data:
            user_msg = f"Solve the following math problem. Show your work and end with the answer after ####.\n\n{ex['question']}"
            assistant_msg = ex['answer']
            conv = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_msg}],
                tokenize=False
            )
            conversations.append({"text": conv})

    elif benchmark_name == "mmlu":
        # Use auxiliary_train + dev for training
        labels = ['A', 'B', 'C', 'D']
        for split_name in ['auxiliary_train', 'dev']:
            if split_name in ds:
                data = ds[split_name]
                for ex in data:
                    choices_str = "\n".join(f"({labels[i]}) {c}" for i, c in enumerate(ex['choices']))
                    user_msg = f"{ex['question']}\n{choices_str}\n\nAnswer with just the letter:"
                    assistant_msg = labels[ex['answer']]
                    conv = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_msg},
                         {"role": "assistant", "content": assistant_msg}],
                        tokenize=False
                    )
                    conversations.append({"text": conv})

    elif benchmark_name == "winogrande":
        data = ds['train']
        for ex in data:
            sentence = ex['sentence']
            user_msg = f"Fill in the blank: {sentence}\n(A) {ex['option1']}\n(B) {ex['option2']}\n\nAnswer with just the letter:"
            answer = 'A' if ex['answer'] == '1' else 'B'
            conv = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": answer}],
                tokenize=False
            )
            conversations.append({"text": conv})

    elif benchmark_name == "arc_challenge":
        data = ds['train']
        for ex in data:
            choices = ex['choices']['text']
            labels = ex['choices']['label']
            choices_str = "\n".join(f"({l}) {c}" for l, c in zip(labels, choices))
            user_msg = f"{ex['question']}\n{choices_str}\n\nAnswer with just the letter:"
            assistant_msg = ex['answerKey']
            conv = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_msg}],
                tokenize=False
            )
            conversations.append({"text": conv})

    elif benchmark_name == "gpqa":
        data = ds['train']
        for ex in data:
            user_msg = ex['question'] + "\n\nAnswer with just the letter:"
            assistant_msg = ex['answer']
            conv = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_msg}],
                tokenize=False
            )
            conversations.append({"text": conv})

    else:
        raise ValueError(f"No training data handler for: {benchmark_name}")

    if max_samples and len(conversations) > max_samples:
        import random
        random.seed(42)
        conversations = random.sample(conversations, max_samples)

    return Dataset.from_list(conversations)


def finetune_model(model, tokenizer, benchmark_name, output_dir,
                   max_train_samples=5000, num_epochs=2, batch_size=4,
                   learning_rate=2e-4, max_seq_length=1024):
    """Fine-tune model with LoRA on a benchmark's training data.

    Returns the fine-tuned model.
    """
    # Prepare data
    train_dataset = prepare_training_data(benchmark_name, tokenizer, max_train_samples)
    print(f"Training on {len(train_dataset)} examples from {benchmark_name}")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=2,
        max_length=max_seq_length,
    )

    # Trainer
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    return peft_model
