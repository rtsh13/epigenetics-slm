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


def convert_to_gguf(
    hf_dir: str,
    out_path: str,
    quantization: str = "q4_k_m",
    llama_cpp_dir: str | None = None,
) -> None:
    llama_dir = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if not llama_dir:
        raise RuntimeError(
            "llama.cpp checkout not found. Pass llama_cpp_dir=... or set the "
            "LLAMA_CPP_DIR environment variable to your llama.cpp clone."
        )
    llama_path = Path(llama_dir)
    convert_script = llama_path / "convert_hf_to_gguf.py"
    quantize_bin = llama_path / "llama-quantize"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fp16_path = str(Path(out_path).with_suffix(".fp16.gguf"))

    subprocess.run(
        ["python", str(convert_script), hf_dir, "--outfile", fp16_path, "--outtype", "f16"],
        check=True,
    )
    subprocess.run(
        [str(quantize_bin), fp16_path, str(out_path), quantization],
        check=True,
    )
