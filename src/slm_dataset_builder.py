"""
Generate the SLM training dataset from NHANES features + Bio-RAG.

Walks data/nhanes_features.parquet, classifies each participant's
biomarkers, retrieves top-2 Bio-RAG chunks per participant, fills the
shared prompt/response templates, and writes one JSONL line per valid
participant to data/slm_train.jsonl.

CLI: python -m slm_dataset_builder --parquet data/nhanes_features.parquet --out data/slm_train.jsonl
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import pandas as pd

from biomarker_classifier import (
    classify_aging,
    classify_circadian,
    classify_hba1c,
    classify_nlr,
    classify_sleep,
)
from rag_retriever import BioRAG
from slm_prompt import build_prompt, build_response

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("hba1c", "nlr")


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _build_query(row: dict, classifications: dict) -> str:
    advance = row["cosinorage_advance"]
    parts = [f"biological age acceleration {advance:+.1f} years"]
    hba1c_label = classifications["hba1c"][0]
    nlr_label = classifications["nlr"][0]
    if hba1c_label != "Normal":
        parts.append(f"HbA1c {row['hba1c']:.1f}% {hba1c_label.lower()}")
    if nlr_label != "Normal":
        parts.append(f"NLR {row['nlr']:.1f} {nlr_label.lower()} inflammation")
    if row["sleep_TST_mean"] / 60.0 < 6.0:
        parts.append("short sleep duration circadian disruption")
    parts.append("epigenetic aging interventions")
    return ", ".join(parts)


def row_to_pair(row: dict, rag: BioRAG) -> dict | None:
    for field in REQUIRED_FIELDS:
        if _is_missing(row.get(field)):
            return None

    classifications = {
        "hba1c": classify_hba1c(float(row["hba1c"])),
        "nlr": classify_nlr(float(row["nlr"])),
        "aging": classify_aging(float(row["cosinorage_advance"])),
        "circadian": classify_circadian(
            float(row["nonparam_IS_mean"]),
            float(row["nonparam_IV_mean"]),
            float(row["nonparam_RA_mean"]),
        ),
        "sleep": classify_sleep(
            float(row["sleep_TST_mean"]),
            float(row["sleep_SRI_mean"]),
        ),
    }

    query = _build_query(row, classifications)
    rag_chunks = rag.search(query, n_results=2)

    biomarkers = {
        "age": float(row["age"]),
        "sex": row["sex"],
        "hba1c": float(row["hba1c"]),
        "nlr": float(row["nlr"]),
        "wbc": float(row["wbc"]),
        "cosinorage_advance": float(row["cosinorage_advance"]),
        "is_value": float(row["nonparam_IS_mean"]),
        "iv_value": float(row["nonparam_IV_mean"]),
        "ra_value": float(row["nonparam_RA_mean"]),
        "tst_minutes": float(row["sleep_TST_mean"]),
        "sri": float(row["sleep_SRI_mean"]),
    }

    return {
        "seqn": int(row["seqn"]),
        "prompt": build_prompt(biomarkers, rag_chunks),
        "response": build_response(biomarkers, classifications, rag_chunks),
        "biomarkers": biomarkers,
        "rag_chunks": rag_chunks,
    }


def build_dataset(
    parquet_path: str,
    output_path: str,
    rag: BioRAG,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    df = pd.read_parquet(parquet_path)

    pairs: list[dict] = []
    n_skipped = 0
    for _, row in df.iterrows():
        pair = row_to_pair(row.to_dict(), rag)
        if pair is None:
            n_skipped += 1
            continue
        pairs.append(pair)

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_eval = int(round(len(pairs) * eval_fraction))
    for i, pair in enumerate(pairs):
        pair["split"] = "eval" if i < n_eval else "train"

    pairs.sort(key=lambda p: p["seqn"])  # stable on-disk order

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    n_train = sum(1 for p in pairs if p["split"] == "train")
    log.info("wrote %d pairs (train=%d eval=%d, skipped=%d) to %s",
             len(pairs), n_train, n_eval, n_skipped, output_path)
    return {"n_train": n_train, "n_eval": n_eval, "n_skipped": n_skipped}


def main():
    parser = argparse.ArgumentParser(description="Generate SLM training dataset")
    parser.add_argument("--parquet", default="data/nhanes_features.parquet")
    parser.add_argument("--out", default="data/slm_train.jsonl")
    parser.add_argument("--chroma-dir", default="data/chroma_db")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rag = BioRAG(chroma_dir=args.chroma_dir)
    build_dataset(
        parquet_path=args.parquet,
        output_path=args.out,
        rag=rag,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
