"""
Merge a LoRA adapter into its base model and export the result as a
GGUF file for CPU inference via llama-cpp-python.

Heavy dependencies (peft, transformers, llama.cpp subprocess) are
imported lazily so the module and its unit tests run without them.
"""

import os
import subprocess
from pathlib import Path


def merge_lora(base_model_id: str, adapter_dir: str, out_dir: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
