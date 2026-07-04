"""
Offline eval harness for the fine-tuned SLM.

Metrics:
- category_coverage: presence of the five section headers.
- classification_exact_match: does the response mention the expected
  Pre-diabetic / Elevated / Favorable / Short-style label per section.
- rouge_l: longest-common-subsequence F1 against the reference response.
"""

import json as _json
import re


CATEGORIES = ("AGING", "STRESS", "METABOLISM", "INFLAMMATION", "SLEEP")

_SECTION_KEYS = {
    "hba1c": "METABOLISM",
    "nlr": "INFLAMMATION",
    "aging": "AGING",
    "sleep": "SLEEP",
}


def category_coverage(generated: str) -> dict:
    return {cat: bool(re.search(rf"\b{cat}\b\s*:", generated)) for cat in CATEGORIES}


def _section_text(generated: str, header: str) -> str:
    pattern = rf"{header}\s*:(.*?)(?=(?:\n|^)(?:{'|'.join(CATEGORIES)})\s*:|\Z)"
    m = re.search(pattern, generated, flags=re.DOTALL | re.MULTILINE)
    return m.group(1) if m else ""


def classification_exact_match(generated: str, expected_classifications: dict) -> dict:
    matches = {}
    for key, header in _SECTION_KEYS.items():
        expected_label = expected_classifications[key][0]
        section = _section_text(generated, header)
        matches[key] = expected_label.lower() in section.lower()
    return matches


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_generation(
    generated: str,
    reference: str,
    expected_classifications: dict,
) -> dict:
    coverage = category_coverage(generated)
    matches = classification_exact_match(generated, expected_classifications)
    match_rate = sum(matches.values()) / len(matches) if matches else 0.0
    return {
        "category_coverage": coverage,
        "category_coverage_all": all(coverage.values()),
        "classification_matches": matches,
        "classification_match_rate": match_rate,
        "rouge_l": rouge_l(reference, generated),
    }


def _expected_classifications_from_biomarkers(biomarkers: dict) -> dict:
    from biomarker_classifier import (
        classify_aging, classify_hba1c, classify_nlr, classify_sleep,
    )
    return {
        "hba1c": classify_hba1c(float(biomarkers["hba1c"])),
        "nlr": classify_nlr(float(biomarkers["nlr"])),
        "aging": classify_aging(float(biomarkers["cosinorage_advance"])),
        "sleep": classify_sleep(
            float(biomarkers["tst_minutes"]),
            float(biomarkers["sri"]),
        ),
    }


def evaluate_dataset(jsonl_path: str, generator, split: str = "eval") -> dict:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            if rec.get("split", "train") == split:
                records.append(rec)

    per_record = []
    for rec in records:
        if "biomarkers" not in rec:
            raise RuntimeError(
                f"JSONL record (split={rec.get('split', 'train')}, "
                f"seqn={rec.get('seqn', '?')}) is missing the 'biomarkers' field. "
                "Extend the dataset builder (Task 12) to include biomarkers in every record."
            )
        biomarkers = rec["biomarkers"]
        rag_chunks = rec.get("rag_chunks", [])
        expected = _expected_classifications_from_biomarkers(biomarkers)
        generated = generator.generate(biomarkers, rag_chunks)
        per_record.append(evaluate_generation(
            generated=generated,
            reference=rec["response"],
            expected_classifications=expected,
        ))

    n = len(per_record)
    if n == 0:
        return {"n_records": 0}

    coverage_all = sum(1 for r in per_record if r["category_coverage_all"]) / n
    match_rate = sum(r["classification_match_rate"] for r in per_record) / n
    rouge_mean = sum(r["rouge_l"] for r in per_record) / n

    return {
        "n_records": n,
        "category_coverage_all_rate": coverage_all,
        "classification_match_rate_mean": match_rate,
        "rouge_l_mean": rouge_mean,
        "per_record": per_record,
    }
