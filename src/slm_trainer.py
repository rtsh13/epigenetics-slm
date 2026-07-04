"""
QLoRA fine-tuning for Llama 3.2 1B on the SLM dataset.

The pure helpers (load_jsonl_dataset, format_example, build_training_args)
are tested on Mac. The train() entry point imports Unsloth and TRL lazily
so importing this module without those dependencies is safe.
"""

import json
from pathlib import Path

from slm_prompt import build_training_text


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
    text = build_training_text(record["prompt"], record["response"])
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
        "optim": "adamw_torch",
        "max_grad_norm": 0.3,
    }


def train(
    dataset_path: str,
    output_dir: str,
    base_model: str = "unsloth/Llama-3.2-1B-Instruct",
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_seq_length: int = 2048,
) -> None:
    import inspect
    from unsloth import FastLanguageModel
    import trl
    from datasets import Dataset
    from transformers import TrainingArguments
    import torch

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=compute_dtype,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=True,
    )
    print({n: p.dtype for n, p in model.named_parameters() if p.requires_grad}.popitem())

    records = load_jsonl_dataset(dataset_path, "train")
    train_ds = Dataset.from_list([format_example(r) for r in records])

    args_dict = build_training_args(
        output_dir=output_dir, epochs=epochs, batch_size=batch_size,
        grad_accum=grad_accum, lr=lr,
    )
    args_dict["bf16"] = bf16_ok
    args_dict["fp16"] = not bf16_ok and torch.cuda.is_available()

    trainer_kwargs = dict(model=model, train_dataset=train_ds)

    sft_config_cls = getattr(trl, "SFTConfig", None)
    if sft_config_cls is not None:
        cfg_params = inspect.signature(sft_config_cls).parameters
        if "dataset_text_field" in cfg_params:
            args_dict["dataset_text_field"] = "text"
        if "max_seq_length" in cfg_params:
            args_dict["max_seq_length"] = max_seq_length
        if "padding_free" in cfg_params:
            args_dict["padding_free"] = False
        trainer_kwargs["args"] = sft_config_cls(**args_dict)
    else:
        trainer_kwargs["args"] = TrainingArguments(**args_dict)
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = max_seq_length

    try:
        trainer = trl.SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)
    except TypeError:
        trainer = trl.SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
