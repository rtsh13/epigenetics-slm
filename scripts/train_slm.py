"""
Colab entrypoint for QLoRA fine-tuning.

Copy this script into a Colab notebook cell (or run via `!python
scripts/train_slm.py ...`) after installing unsloth, trl, and datasets.
Reads the JSONL produced by src/slm_dataset_builder.py and writes a
LoRA adapter to the output directory.
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from slm_trainer import train


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Llama 3.2 1B on SLM dataset")
    parser.add_argument("--dataset", default="data/slm_train.jsonl")
    parser.add_argument("--output", default="models/slm_lora")
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
