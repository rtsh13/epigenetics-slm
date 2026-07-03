"""
QLoRA fine-tuning for Llama 3.2 1B on the SLM dataset.

The pure helpers (load_jsonl_dataset, format_example, build_training_args)
are tested on Mac. The train() entry point imports Unsloth and TRL lazily
so importing this module without those dependencies is safe.
"""

import json
from pathlib import Path


LLAMA_CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
)


def load_jsonl_dataset(path: str, split: str) -> list[dict]:
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split", "train") == split:
                records.append(rec)
    return records


def format_example(record: dict) -> dict:
    text = LLAMA_CHAT_TEMPLATE.format(
        prompt=record["prompt"],
        response=record["response"],
    )
    return {"text": text}


def build_training_args(
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 2e-4,
) -> dict:
    return {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "bf16": True,
        "optim": "paged_adamw_8bit",
    }
