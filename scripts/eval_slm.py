"""
Evaluate a fine-tuned SLM against the JSONL eval split.

Usage:
    python scripts/eval_slm.py \\
        --model models/slm.q4_k_m.gguf \\
        --dataset data/slm_train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from slm_evaluator import evaluate_dataset
from slm_generator import SLMGenerator


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLM on the eval split")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/slm_train.jsonl")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--backend", default="auto", choices=["auto", "gguf", "hf"])
    args = parser.parse_args()

    generator = SLMGenerator(model_path=args.model, backend=args.backend)
    result = evaluate_dataset(args.dataset, generator, split=args.split)

    summary = {k: v for k, v in result.items() if k != "per_record"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
