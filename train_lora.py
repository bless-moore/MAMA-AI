"""Pathway 2: fine-tune Gemma 3 1B (instruction-tuned) with LoRA on maternal-health Q&A.

Runs on a free Colab T4 (fp32 weights, ~1B params) or any GPU with bf16 support.

Usage:
    huggingface-cli login            # needed once, after accepting the Gemma licence
    python train_lora.py --data data/dataset.jsonl
"""
import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from common import MODEL_ID, SYSTEM_PROMPT, load_jsonl, split


def to_prompt_completion(row):
    # prompt/completion format => loss is computed on the assistant answer only
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "completion": [{"role": "assistant", "content": row["answer"]}],
    }


def main(args):
    train_rows, val_rows = split(load_jsonl(args.data))
    train_ds = Dataset.from_list([to_prompt_completion(r) for r in train_rows])
    val_ds = Dataset.from_list([to_prompt_completion(r) for r in val_rows])

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32  # fp16 is unstable with Gemma 3

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation="eager"
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        max_length=768,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(f"{args.out}/adapter")
    tokenizer.save_pretrained(f"{args.out}/adapter")
    print(f"Adapter saved to {args.out}/adapter")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seed_examples.jsonl")
    ap.add_argument("--out", default="artifacts/mama-lora")
    ap.add_argument("--epochs", type=int, default=3)
    main(ap.parse_args())
