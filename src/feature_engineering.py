"""
Week 2: Feature Engineering Pipeline

Batch-converts NHANES accelerometry data, extracts circadian/sleep/activity
features via CosinorAge, computes biological age, and produces a merged
feature matrix saved as data/nhanes_features.parquet.
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

from cosinorage.bioages import CosinorAge
from cosinorage.datahandlers import GenericDataHandler
from cosinorage.features import WearableFeatures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
CSV_DIR = PROJECT_DIR / "data" / "accel_csv"
PARTICIPANTS_PATH = PROJECT_DIR / "data" / "nhanes_participants.parquet"
OUTPUT_PATH = PROJECT_DIR / "data" / "nhanes_features.parquet"
CHECKPOINT_PATH = PROJECT_DIR / "data" / "feature_checkpoint.json"
EDA_DIR = PROJECT_DIR / "reports" / "week2_eda"

# Per-day keys that should be aggregated to mean + std
PER_DAY_KEYS = {
    "nonparam": ["M10", "M10_start", "L5", "L5_start", "RA", "IS", "IV"],
    "physical_activity": ["sedentary", "light", "moderate", "vigorous"],
    "sleep": ["TST", "WASO", "PTA", "NWB", "SOL", "SRI"],
}

# Scalar cosinor keys kept as-is (library uses lowercase 'mesor')
COSINOR_KEYS = ["mesor", "amplitude", "acrophase", "acrophase_time"]


def flatten_features(feature_dict: dict) -> dict:
    """Convert nested WearableFeatures output to a flat dict.

    Per-day lists → mean + std columns.
    Cosinor params stay scalar.
    """
    flat = {}

    # Cosinor: scalar values
    cosinor = feature_dict.get("cosinor", {})
    for key in COSINOR_KEYS:
        val = cosinor.get(key)
        flat[f"cosinor_{key}"] = val

    # Non-parametric, physical_activity, sleep: may be per-day lists
    for category, keys in PER_DAY_KEYS.items():
        cat_data = feature_dict.get(category, {})
        for key in keys:
            val = cat_data.get(key)
            if val is None:
                flat[f"{category}_{key}_mean"] = np.nan
                flat[f"{category}_{key}_std"] = np.nan
            elif isinstance(val, (list, np.ndarray)):
                # Convert Timestamps to fractional hours
                if len(val) > 0 and hasattr(val[0], "hour"):
                    arr = np.array([v.hour + v.minute / 60.0 for v in val])
                else:
                    arr = np.array(val, dtype=float)
                arr = arr[~np.isnan(arr)]
                if len(arr) > 0:
                    flat[f"{category}_{key}_mean"] = float(np.mean(arr))
                    flat[f"{category}_{key}_std"] = float(np.std(arr))
                else:
                    flat[f"{category}_{key}_mean"] = np.nan
                    flat[f"{category}_{key}_std"] = np.nan
            else:
                # Scalar value (e.g. IS, IV are single values)
                flat[f"{category}_{key}_mean"] = float(val)
                flat[f"{category}_{key}_std"] = 0.0

    return flat


def extract_participant_features(csv_path: Path) -> dict | None:
    """Load a participant CSV and extract wearable features.

    Returns flat dict with 'seqn' key, or None on failure.
    """
    seqn = int(csv_path.stem.split("_")[1])
    try:
        handler = GenericDataHandler(
            file_path=str(csv_path),
            data_type="enmo-mg",
            time_format="datetime",
            time_column="timestamp",
            data_columns=["enmo"],
            verbose=False,
        )
        features = WearableFeatures(handler)
        feature_dict = features.get_features()
        flat = flatten_features(feature_dict)
        flat["seqn"] = seqn
        return flat
    except Exception as e:
        log.warning("Participant %d failed: %s", seqn, e)
        return None


def _save_checkpoint(results: list, path: Path):
    """Atomically save checkpoint via tmp file + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    tmp.rename(path)


def batch_extract_features(
    csv_dir: Path = CSV_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    max_participants: int | None = None,
) -> pd.DataFrame:
    """Extract features from all participant CSVs with checkpointing.

    Saves a JSON checkpoint every 100 participants and resumes on re-run.
    """
    csv_files = sorted(csv_dir.glob("participant_*.csv"))
    if max_participants:
        csv_files = csv_files[:max_participants]

    # Load checkpoint
    done_seqns = set()
    results = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_seqns = {r["seqn"] for r in results}
        log.info("Resumed from checkpoint: %d participants already done", len(done_seqns))

    remaining = [f for f in csv_files if int(f.stem.split("_")[1]) not in done_seqns]
    log.info("Extracting features: %d remaining, %d total", len(remaining), len(csv_files))

    for i, csv_path in enumerate(tqdm(remaining, desc="Extracting features"), 1):
        feat = extract_participant_features(csv_path)
        if feat is not None:
            results.append(feat)
        if i % 100 == 0:
            _save_checkpoint(results, checkpoint_path)

    # Final save
    if remaining:
        _save_checkpoint(results, checkpoint_path)

    log.info("Feature extraction complete: %d / %d succeeded", len(results), len(csv_files))
    return pd.DataFrame(results)


def compute_cosinorage(
    features_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    csv_dir: Path = CSV_DIR,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Compute CosinorAge biological age for each participant.

    Processes in batches to limit memory. Falls back to per-participant
    processing if a batch fails.
    """
    # Merge to get age and sex
    merged = features_df.merge(
        participants_df[["seqn", "age", "sex"]],
        on="seqn",
        how="inner",
    )

    gender_map = {"Male": "male", "Female": "female"}
    cosinorage_vals = {}
    advance_vals = {}

    seqns = merged["seqn"].tolist()
    ages = merged.set_index("seqn")["age"].to_dict()
    sexes = merged.set_index("seqn")["sex"].to_dict()

    for batch_start in range(0, len(seqns), batch_size):
        batch_seqns = seqns[batch_start : batch_start + batch_size]
        log.info(
            "CosinorAge batch %d-%d / %d",
            batch_start,
            batch_start + len(batch_seqns),
            len(seqns),
        )

        records = []
        record_seqns = []
        for seqn in batch_seqns:
            csv_path = csv_dir / f"participant_{seqn}.csv"
            if not csv_path.exists():
                continue
            try:
                handler = GenericDataHandler(
                    file_path=str(csv_path),
                    data_type="enmo-mg",
                    time_format="datetime",
                    time_column="timestamp",
                    data_columns=["enmo"],
                    verbose=False,
                )
                records.append({
                    "handler": handler,
                    "age": float(ages[seqn]),
                    "gender": gender_map.get(sexes[seqn], "unknown"),
                })
                record_seqns.append(seqn)
            except Exception as e:
                log.warning("Handler failed for %d: %s", seqn, e)

        if not records:
            continue

        # Try batch computation
        try:
            ca = CosinorAge(records)
            preds = ca.get_predictions()
            for seqn, pred in zip(record_seqns, preds):
                cosinorage_vals[seqn] = pred.get("cosinorage")
                advance_vals[seqn] = pred.get("cosinorage_advance")
        except Exception as e:
            log.warning("Batch CosinorAge failed, falling back to per-participant: %s", e)
            for seqn, rec in zip(record_seqns, records):
                try:
                    ca = CosinorAge([rec])
                    pred = ca.get_predictions()[0]
                    cosinorage_vals[seqn] = pred.get("cosinorage")
                    advance_vals[seqn] = pred.get("cosinorage_advance")
                except Exception as e2:
                    log.warning("CosinorAge failed for %d: %s", seqn, e2)

    features_df = features_df.copy()
    features_df["cosinorage"] = features_df["seqn"].map(cosinorage_vals)
    features_df["cosinorage_advance"] = features_df["seqn"].map(advance_vals)
    return features_df


def run_eda(df: pd.DataFrame):
    """Generate Week 2 EDA plots and save to reports/week2_eda/."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CosinorAge vs chronological age scatter
    valid = df.dropna(subset=["cosinorage", "age"])
    if len(valid) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(valid["age"], valid["cosinorage"], alpha=0.3, s=10)
        lims = [
            min(valid["age"].min(), valid["cosinorage"].min()) - 5,
            max(valid["age"].max(), valid["cosinorage"].max()) + 5,
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="y=x")
        corr = valid["age"].corr(valid["cosinorage"])
        ax.set_xlabel("Chronological Age")
        ax.set_ylabel("CosinorAge (Biological)")
        ax.set_title(f"CosinorAge vs Chronological Age (r={corr:.3f}, n={len(valid)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(EDA_DIR / "cosinorage_vs_age.png", dpi=150)
        plt.close(fig)
        log.info("Saved cosinorage_vs_age.png (r=%.3f)", corr)

    # 2. CosinorAge advance distribution
    valid_adv = df.dropna(subset=["cosinorage_advance"])
    if len(valid_adv) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(valid_adv["cosinorage_advance"], bins=50, edgecolor="black", alpha=0.7)
        mean_adv = valid_adv["cosinorage_advance"].mean()
        ax.axvline(mean_adv, color="r", linestyle="--", label=f"Mean={mean_adv:.1f}")
        ax.axvline(0, color="k", linestyle="-", alpha=0.3)
        ax.set_xlabel("CosinorAge Advance (years)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Biological Age Advance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(EDA_DIR / "cosinorage_advance_dist.png", dpi=150)
        plt.close(fig)

    # 3. Key feature distributions (2x3 grid)
    feat_cols = [
        ("cosinor_mesor", "MESOR"),
        ("cosinor_amplitude", "Amplitude"),
        ("nonparam_RA_mean", "Relative Amplitude"),
        ("sleep_TST_mean", "Total Sleep Time"),
        ("sleep_WASO_mean", "WASO"),
        ("physical_activity_moderate_mean", "Moderate Activity"),
    ]
    available = [(c, l) for c, l in feat_cols if c in df.columns]
    if available:
        n = len(available)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, (col, label) in enumerate(available):
            vals = df[col].dropna()
            if len(vals) > 0:
                axes[i].hist(vals, bins=40, edgecolor="black", alpha=0.7)
            axes[i].set_title(label)
            axes[i].set_ylabel("Count")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        fig.savefig(EDA_DIR / "feature_distributions.png", dpi=150)
        plt.close(fig)

    # 4. Feature correlation matrix
    numeric = df.select_dtypes(include=[np.number])
    # Drop columns with all NaN
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] > 2:
        corr_matrix = numeric.corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr_matrix.columns, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "correlation_matrix.png", dpi=150)
        plt.close(fig)

    log.info("EDA plots saved to %s", EDA_DIR)


def main():
    parser = argparse.ArgumentParser(description="Week 2: Feature Engineering Pipeline")
    parser.add_argument("--skip-convert", action="store_true", help="Skip CSV conversion step")
    parser.add_argument("--max-participants", type=int, default=None, help="Limit number of participants")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plot generation")
    args = parser.parse_args()

    # Step 1: Convert accelerometry data to CSVs
    if not args.skip_convert:
        log.info("Step 1: Converting accelerometry data to CSVs...")
        from nhanes_accel_adapter import batch_convert
        batch_convert(max_participants=args.max_participants)
    else:
        log.info("Step 1: Skipping CSV conversion (--skip-convert)")

    # Step 2: Extract wearable features
    log.info("Step 2: Extracting wearable features...")
    features_df = batch_extract_features(
        csv_dir=CSV_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        max_participants=args.max_participants,
    )

    if features_df.empty:
        log.error("No features extracted. Exiting.")
        return

    log.info("Extracted features for %d participants, %d columns", *features_df.shape)

    # Step 3: Load demographics
    log.info("Step 3: Loading demographics...")
    participants_df = pd.read_parquet(PARTICIPANTS_PATH)

    # Step 4: Compute CosinorAge
    log.info("Step 4: Computing CosinorAge...")
    features_df = compute_cosinorage(features_df, participants_df, csv_dir=CSV_DIR)

    valid_ca = features_df["cosinorage"].notna().sum()
    log.info("CosinorAge computed for %d / %d participants", valid_ca, len(features_df))

    # Step 5: Merge with demographics
    log.info("Step 5: Merging with demographics...")
    merged = participants_df.merge(features_df, on="seqn", how="inner")
    log.info("Final dataset: %d participants, %d columns", *merged.shape)

    # Step 6: Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    merged.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)
    log.info("Saved → %s (.parquet + .csv)", OUTPUT_PATH)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Feature matrix: {merged.shape[0]} participants × {merged.shape[1]} columns")
    print(f"Columns: {merged.columns.tolist()}")
    print(f"\nNull counts (top 10):")
    nulls = merged.isnull().sum().sort_values(ascending=False).head(10)
    for col, n in nulls.items():
        print(f"  {col}: {n} ({100*n/len(merged):.1f}%)")
    print(f"{'='*60}\n")

    # Step 7: EDA
    if not args.skip_eda:
        log.info("Step 7: Generating EDA plots...")
        run_eda(merged)
    else:
        log.info("Step 7: Skipping EDA (--skip-eda)")


if __name__ == "__main__":
    main()
