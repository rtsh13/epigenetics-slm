"""
autonomic_aging_processor.py - Batch HRV feature extraction from the
Autonomic Aging dataset (PhysioNet).

Public functions:
    load_subject_metadata(csv_path: str) -> pd.DataFrame
    process_single_participant(record_path: str, subject_id: str) -> dict | None
    batch_process(...) -> pd.DataFrame
    run_eda(df: pd.DataFrame) -> None
    main() -> None
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Support both script execution (src/ on sys.path) and test imports (project
# root on sys.path, src imported as namespace package).
try:
    from hrv_pipeline import load_wfdb_record, process_ecg_segment, process_bp_segment
except ImportError:
    from src.hrv_pipeline import load_wfdb_record, process_ecg_segment, process_bp_segment

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "autonomic_aging"
SUBJECT_CSV = DATA_DIR / "subject-info.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "autonomic_features.parquet"
CHECKPOINT_PATH = PROJECT_DIR / "data" / "autonomic_checkpoint.json"
EDA_DIR = PROJECT_DIR / "reports" / "week3_eda"


# Function 1: load_subject_metadata

def load_subject_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load subject metadata CSV and normalise column names.

    Raises FileNotFoundError if the file does not exist.
    Column normalisation: lowercase, spaces → '_', hyphens → '_'.
    The 'subject_id' column is cast to str.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Subject metadata CSV not found: {csv_path}")

    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [
        col.lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]

    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)

    return df


# Function 2: process_single_participant

def process_single_participant(record_path: str, subject_id: str) -> dict | None:
    """
    Load a WFDB record and extract HRV (and optionally BP) features.

    Returns None on any loading/processing failure.
    """
    # Load the WFDB record
    try:
        ecg_signal, bp_signal, fs = load_wfdb_record(record_path)
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("Failed to load record %s: %s", record_path, exc)
        return None

    # Extract ECG/HRV features
    hrv_features = process_ecg_segment(ecg_signal, sampling_rate=fs)
    if hrv_features is None:
        log.warning("ECG processing returned None for subject %s", subject_id)
        return None

    features: dict = {"subject_id": subject_id}
    features.update(hrv_features)

    # Extract BP features if available
    if bp_signal is not None:
        bp_features = process_bp_segment(bp_signal, sampling_rate=fs)
        if bp_features is not None:
            features.update(bp_features)

    return features


# Function 3: _save_checkpoint

def _save_checkpoint(results: list, path: Path) -> None:
    """Atomically save checkpoint via tmp file + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    tmp.rename(path)


# Function 4: batch_process

def batch_process(
    data_dir: Path = DATA_DIR,
    subject_csv: Path = SUBJECT_CSV,
    checkpoint_path: Path = CHECKPOINT_PATH,
    max_participants: int | None = None,
) -> pd.DataFrame:
    """
    Batch-process all WFDB records in data_dir, with checkpointing.

    Returns a DataFrame of HRV (and BP) features, one row per participant.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        log.error("Data directory does not exist: %s", data_dir)
        return pd.DataFrame()

    # Glob all .hea header files and derive record paths (strip .hea suffix)
    record_paths = [str(p.with_suffix("")) for p in sorted(data_dir.glob("**/*.hea"))]

    if max_participants is not None:
        record_paths = record_paths[:max_participants]

    # Load existing checkpoint
    results: list = []
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                results = json.load(f)
            log.info("Loaded checkpoint with %d records.", len(results))
        except Exception as exc:
            log.warning("Could not load checkpoint: %s", exc)
            results = []

    processed_ids: set[str] = {r.get("subject_id", "") for r in results}

    # Filter out already-processed records
    remaining = [p for p in record_paths if Path(p).stem not in processed_ids]

    log.info(
        "Total records: %d | Already processed: %d | Remaining: %d",
        len(record_paths),
        len(results),
        len(remaining),
    )

    for i, record_path in enumerate(tqdm(remaining, desc="Processing participants")):
        subject_id = Path(record_path).stem
        features = process_single_participant(record_path, subject_id=subject_id)
        if features is not None:
            results.append(features)

        # Save checkpoint every 100 participants
        if (i + 1) % 100 == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(results, checkpoint_path)
            log.info("Checkpoint saved at %d participants processed.", i + 1)

    # Save final checkpoint (covers last partial batch not caught by the every-100 save)
    if remaining:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(results, checkpoint_path)

    return pd.DataFrame(results)


# Function 5: run_eda

def run_eda(df: pd.DataFrame) -> None:
    """
    Generate exploratory data analysis plots and save to EDA_DIR.

    Produces:
        rmssd_vs_age.png
        sdnn_vs_age.png
        resting_hr_vs_age.png
        hrv_feature_distributions.png
    """
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    def _scatter(x_col: str, y_col: str, filename: str, color: str = "steelblue") -> None:
        subset = df[[x_col, y_col]].dropna()
        if len(subset) <= 1:
            log.warning("Not enough data to plot %s vs %s", x_col, y_col)
            return
        r = subset[x_col].corr(subset[y_col])
        fig, ax = plt.subplots()
        ax.scatter(subset[x_col], subset[y_col], alpha=0.5, color=color, s=15)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}  (r={r:.3f})")
        fig.savefig(EDA_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 1. RMSSD vs age
    _scatter("age", "hrv_rmssd", "rmssd_vs_age.png")

    # 2. SDNN vs age
    _scatter("age", "hrv_sdnn", "sdnn_vs_age.png", color="darkorange")

    # 3. Resting HR vs age
    _scatter("age", "hrv_mean_hr", "resting_hr_vs_age.png", color="crimson")

    # 4. HRV feature distributions - 2×3 grid
    hrv_cols = [
        "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
        "hrv_lf_hf", "hrv_mean_hr", "hrv_dfa_alpha1",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, hrv_cols):
        data = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        ax.hist(data, bins=30, edgecolor="white", linewidth=0.4)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "hrv_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("EDA plots saved to %s", EDA_DIR)


# Function 6: main

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 3: Autonomic Aging HRV Pipeline")
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    features_df = batch_process(max_participants=args.max_participants)

    if features_df.empty:
        log.error("No features extracted. Download data to %s first.", DATA_DIR)
        return

    metadata = load_subject_metadata(str(SUBJECT_CSV))
    merged = metadata.merge(features_df, on="subject_id", how="inner")
    log.info("Final dataset: %d participants × %d columns", *merged.shape)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    merged.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)
    log.info("Saved → %s", OUTPUT_PATH)

    print(f"\n{'='*60}")
    print(f"Feature matrix: {merged.shape[0]} participants × {merged.shape[1]} columns")
    nulls = merged.isnull().sum().sort_values(ascending=False).head(10)
    for col, n in nulls.items():
        print(f"  {col}: {n} ({100*n/len(merged):.1f}%)")
    print(f"{'='*60}\n")

    if not args.skip_eda:
        run_eda(merged)


if __name__ == "__main__":
    main()
