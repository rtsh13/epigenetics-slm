"""
demo.py - Side-by-side epigenetic aging contrast demo.

Loads two real NHANES participants:
  - SEQN 62347: 71F, CosinorAge advance +2.4 yr  (healthy aging)
  - SEQN 71166: 55M, CosinorAge advance +13.7 yr (accelerated aging, diabetic, high NLR)

Runs the labs-only XGBoost model, queries Bio-RAG, and prints a formatted
five-category health report for each participant.

Usage (from project root, venv activated):
    source wearable-age/bin/activate
    python scripts/demo.py
    python scripts/demo.py --slm-model path/to/model.gguf
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from baseline_trainer import XGBoostAgePredictor
from rag_retriever import BioRAG

HEALTHY_SEQN = 62347
ACCELERATED_SEQN = 71166

RACE_LABELS = {1.0: "Mexican American", 2.0: "Other Hispanic",
               3.0: "Non-Hispanic White", 4.0: "Non-Hispanic Black",
               5.0: "Other Race / Multi-Racial"}

W = 78


def hr(char="="):
    return char * W


def section(title):
    print(f"\n  {title}")
    print("  " + "-" * (W - 2))


def row(label, value, note=""):
    label_str = f"    {label:<28}"
    note_str = f"  {note}" if note else ""
    print(f"{label_str}{value}{note_str}")


def wrap_print(text, indent=4):
    prefix = " " * indent
    for line in textwrap.wrap(text, width=W - indent):
        print(prefix + line)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).astype(float)
    if "race_ethnicity" in df.columns:
        df["race_ethnicity"] = df["race_ethnicity"].astype("category").cat.codes.astype(float)
    return df


def hrv_note():
    section("STRESS  (Heart Rate Variability)")
    row("Dataset:", "Autonomic Aging - PhysioNet (1,121 participants)")
    row("Features:", "RMSSD, SDNN, pNN50, LF/HF, sample entropy, DFA")
    row("Model MAE:", "~5.6 years  (matches Schumann et al. 2023 benchmark)")
    row("Note:", "HRV not in NHANES (accelerometer only, no heart rate)")


def aging_interp(advance: float) -> str:
    if advance < 3:
        return "Biological age closely tracks chronological age. Favorable aging trajectory."
    elif advance < 8:
        return "Modest biological age acceleration. Circadian or lifestyle factors may contribute."
    else:
        return "Significant biological age acceleration. Circadian disruption is a key driver."


def circadian_interp(is_val: float, iv_val: float, ra_val: float) -> str:
    flags = []
    if is_val >= 0.6:
        flags.append("strong day-to-day rhythm consistency (IS=%.2f)" % is_val)
    elif is_val < 0.35:
        flags.append("fragmented circadian rhythm (IS=%.2f, low)" % is_val)
    if iv_val < 0.6:
        flags.append("stable within-day activity pattern (IV=%.2f, low)" % iv_val)
    elif iv_val > 0.9:
        flags.append("high within-day fragmentation (IV=%.2f, elevated)" % iv_val)
    if ra_val >= 0.75:
        flags.append("sharp day/night activity contrast (RA=%.2f, high)" % ra_val)
    return ("; ".join(flags) + ".") if flags else "Circadian metrics within range."


def hba1c_interp(val: float) -> str:
    if val < 5.7:
        return "Normal. Stable glucose metabolism, low epigenetic oxidative stress risk."
    elif val < 6.5:
        return "Pre-diabetic range. Elevated glucose may accelerate DNA methylation drift."
    else:
        return "Diabetic range. Chronic hyperglycemia drives oxidative stress and methylation damage."


def nlr_interp(val: float) -> str:
    if val < 3.0:
        return "Normal. No systemic inflammatory signal detected."
    elif val < 6.0:
        return "Elevated. Suggests chronic low-grade inflammation (inflammaging risk)."
    else:
        return "High. Strong systemic inflammation - primary driver of epigenetic erosion."


def sleep_interp(tst_min: float, sri: float) -> str:
    tst_h = tst_min / 60
    parts = []
    if tst_h < 5.5:
        parts.append(f"short sleep ({tst_h:.1f}h/night - below recommended 7h)")
    elif tst_h > 9.0:
        parts.append(f"extended sleep ({tst_h:.1f}h/night - may reflect fragmented quality)")
    else:
        parts.append(f"adequate sleep duration ({tst_h:.1f}h/night)")
    if sri < 40:
        parts.append("very irregular sleep timing (low SRI - circadian misalignment risk)")
    elif sri < 60:
        parts.append("moderate sleep regularity")
    else:
        parts.append("consistent sleep schedule")
    return "; ".join(parts).capitalize() + "."


def build_rag_query(p: pd.Series) -> str:
    advance = p["cosinorage_advance"]
    hba1c_class = p["hba1c_class"]
    nlr_class = p["nlr_class"]
    tst_h = p["sleep_TST_mean"] / 60
    parts = [f"biological age acceleration {advance:+.1f} years"]
    if hba1c_class != "Normal":
        parts.append(f"HbA1c {p['hba1c']:.1f}% {hba1c_class.lower()}")
    if nlr_class != "Normal":
        parts.append(f"NLR {p['nlr']:.1f} {nlr_class.lower()} inflammation")
    if tst_h < 6.0:
        parts.append("short sleep duration circadian disruption")
    parts.append("epigenetic aging interventions")
    return ", ".join(parts)


def print_report(label: str, idx: int, p_full: pd.Series, pred_age: float, rag_results: list):
    print("\n" + hr())
    title = f"[{idx}/2]  {label} - SEQN {int(p_full['seqn'])}"
    sex = p_full["sex"]
    age = int(p_full["age"])
    bmi = p_full["bmi"]
    race = RACE_LABELS.get(p_full["race_ethnicity"], "Unknown")
    print(f"  {title}")
    print(f"  {age}-year-old {sex}  |  BMI {bmi:.1f}  |  {race}")
    print(hr())

    section("AGING  (CosinorAge - Wearable-Derived Biological Age Clock)")
    cosinorage = p_full["cosinorage"]
    advance = p_full["cosinorage_advance"]
    advance_flag = "aging close to expected" if advance < 5 else "ACCELERATED AGING"
    row("Chronological age:", f"{age} years")
    row("Biological age:", f"{cosinorage:.1f} years  (CosinorAge)")
    row("Age acceleration:", f"{advance:+.1f} years    <<  {advance_flag}")
    is_v = p_full["nonparam_IS_mean"]
    iv_v = p_full["nonparam_IV_mean"]
    ra_v = p_full["nonparam_RA_mean"]
    row("Circadian (IS / IV / RA):", f"{is_v:.2f}  /  {iv_v:.2f}  /  {ra_v:.2f}")
    wrap_print(aging_interp(advance), indent=4)
    wrap_print(circadian_interp(is_v, iv_v, ra_v), indent=4)

    hrv_note()

    section("METABOLISM  (HbA1c - 120-day Glucose Average)")
    hba1c = p_full["hba1c"]
    hba1c_class = p_full["hba1c_class"]
    row("HbA1c:", f"{hba1c:.1f}%    |  {hba1c_class}  (<5.7 Normal / 5.7-6.4 Pre-diabetic / >=6.5 Diabetic)")
    wrap_print(hba1c_interp(hba1c), indent=4)

    section("INFLAMMATION  (NLR - Neutrophil-to-Lymphocyte Ratio)")
    nlr = p_full["nlr"]
    nlr_class = p_full["nlr_class"]
    wbc = p_full["wbc"]
    row("NLR:", f"{nlr:.2f}    |  {nlr_class}  (<3.0 Normal / 3.0-6.0 Elevated / >6.0 High)")
    row("WBC:", f"{wbc:.1f} x10³/μL")
    wrap_print(nlr_interp(nlr), indent=4)

    section("SLEEP  (Accelerometry-Derived Sleep Architecture)")
    tst_min = p_full["sleep_TST_mean"]
    tst_h = tst_min / 60
    sol = p_full["sleep_SOL_mean"]
    sri = p_full["sleep_SRI_mean"]
    row("Total Sleep Time:", f"{tst_h:.1f} hours/night  ({tst_min:.0f} min)")
    row("Sleep Onset Latency:", f"{sol:.0f} min")
    row("Sleep Regularity Index:", f"{sri:.1f}  (0=random, 100=perfectly regular)")
    wrap_print(sleep_interp(tst_min, sri), indent=4)

    section("ML PREDICTION  (Labs-Only XGBoost  |  no CosinorAge leakage)")
    error = pred_age - age
    row("Predicted age:", f"{pred_age:.1f} years")
    row("Actual age:", f"{age} years")
    row("Error:", f"{error:+.1f} years    (model MAE on test set: 10.5 yr, r=0.67)")

    section("BIO-RAG EVIDENCE  (ChromaDB  |  all-MiniLM-L6-v2  |  top 2 chunks)")
    if not rag_results:
        print("    No results returned.")
    for i, r in enumerate(rag_results[:2], 1):
        cat = r.get("category", "?")
        title_r = r.get("title", "")
        source = r.get("source", "")
        context = r.get("context", "")
        print(f"\n    [{i}]  {cat}  |  {title_r}")
        print(f"         Source: {source}")
        wrap_print(context[:300] + ("..." if len(context) > 300 else ""), indent=8)


def _build_biomarker_dict(p_full, rag_chunks):
    return {
        "age": float(p_full["age"]),
        "sex": p_full["sex"],
        "hba1c": float(p_full["hba1c"]),
        "nlr": float(p_full["nlr"]),
        "wbc": float(p_full["wbc"]),
        "cosinorage_advance": float(p_full["cosinorage_advance"]),
        "is_value": float(p_full["nonparam_IS_mean"]),
        "iv_value": float(p_full["nonparam_IV_mean"]),
        "ra_value": float(p_full["nonparam_RA_mean"]),
        "tst_minutes": float(p_full["sleep_TST_mean"]),
        "sri": float(p_full["sleep_SRI_mean"]),
    }


def print_slm_section(generator, p_full, rag_chunks):
    section("SLM NARRATIVE  (Llama 3.2 1B, fine-tuned)")
    biomarkers = _build_biomarker_dict(p_full, rag_chunks)
    text = generator.generate(biomarkers, rag_chunks)
    wrap_print(text, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Epigenetic aging demo")
    parser.add_argument("--slm-model", default=None,
                        help="Optional path to a fine-tuned SLM (GGUF or HF dir)")
    args = parser.parse_args()

    print("\n" + hr())
    print("  EPIGENETIC AGING DEMO  |  SLM-Powered Wearable Health Prototype")
    print("  NHANES 2011-2012 Cohort  |  Five-Category Biological Age Assessment")
    print(hr())
    print("  Loading data and models...")

    df_full = pd.read_parquet(PROJECT_DIR / "data" / "nhanes_features.parquet")
    df_labs = pd.read_parquet(PROJECT_DIR / "data" / "nhanes_participants.parquet")
    df_labs_enc = encode(df_labs)

    model = XGBoostAgePredictor.load(str(PROJECT_DIR / "models" / "nhanes_labs_only.json"))
    print("  XGBoost labs-only model loaded  (MAE 10.5 yr, Pearson r=0.67)")

    print("  Initialising Bio-RAG (ChromaDB + all-MiniLM-L6-v2)...")
    rag = BioRAG(chroma_dir=str(PROJECT_DIR / "data" / "chroma_db"))
    n_chunks = rag._collection.count()
    print(f"  Bio-RAG ready  ({n_chunks} knowledge chunks across 6 categories)")

    slm_generator = None
    if args.slm_model:
        from slm_generator import SLMGenerator
        print(f"  Loading fine-tuned SLM from {args.slm_model}...")
        slm_generator = SLMGenerator(model_path=args.slm_model)
        print("  SLM ready.")

    seqns = [HEALTHY_SEQN, ACCELERATED_SEQN]
    labels = ["Healthy Aging Profile", "Accelerated Aging Profile"]

    for idx, (seqn, label) in enumerate(zip(seqns, labels), 1):
        p_full = df_full[df_full["seqn"] == seqn].iloc[0]
        p_labs = df_labs_enc[df_labs_enc["seqn"] == seqn]
        pred_age = float(model.predict(p_labs)[0])

        query = build_rag_query(p_full)
        rag_results = rag.search(query, n_results=2)

        print_report(label, idx, p_full, pred_age, rag_results)

        if slm_generator is not None:
            print_slm_section(slm_generator, p_full, rag_results)

    print("\n" + hr())
    print("  END OF DEMO")
    if slm_generator is None:
        print("  Pass --slm-model to include the fine-tuned SLM narrative.")
    print(hr() + "\n")


if __name__ == "__main__":
    main()
