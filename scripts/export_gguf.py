"""
End-to-end LoRA to GGUF export.

Usage:
    python scripts/export_gguf.py \
        --base-model unsloth/Llama-3.2-1B-Instruct \
        --adapter models/slm_lora \
        --merged-out models/slm_merged \
        --gguf-out models/slm.q4_k_m.gguf \
        --quantization q4_k_m \
        --llama-cpp-dir /path/to/llama.cpp
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from gguf_exporter import convert_to_gguf, merge_lora


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export GGUF")
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--adapter", default="models/slm_lora")
    parser.add_argument("--merged-out", default="models/slm_merged")
    parser.add_argument("--gguf-out", default="models/slm.q4_k_m.gguf")
    parser.add_argument("--quantization", default="q4_k_m")
    parser.add_argument("--llama-cpp-dir", default=None)
    args = parser.parse_args()

    merge_lora(
        base_model_id=args.base_model,
        adapter_dir=args.adapter,
        out_dir=args.merged_out,
    )
    convert_to_gguf(
        hf_dir=args.merged_out,
        out_path=args.gguf_out,
        quantization=args.quantization,
        llama_cpp_dir=args.llama_cpp_dir,
    )


if __name__ == "__main__":
    main()
