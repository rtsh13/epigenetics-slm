"""
NHANES Accelerometry Adapter for CosinorAge

Converts PhysioNet preprocessed NHANES accelerometry data (wide format)
into per-participant CSV files compatible with CosinorAge's GenericDataHandler.

PhysioNet source: https://physionet.org/content/minute-level-step-count-nhanes/1.0.1/
"""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data" / "nhanes"
ACCEL_CSV_DIR = Path(__file__).parent.parent / "data" / "accel_csv"


def load_physionet_wide(file_path: Path, sample_seqns: list = None) -> pd.DataFrame:
    """Load PhysioNet wide-format .xpt file (SAS XPORT V8).

    Each row = one participant-day, columns min_0001 to min_1440.
    """
    df, _ = pyreadstat.read_xport(str(file_path))
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].astype(int)
    if sample_seqns is not None:
        df = df[df["SEQN"].isin(sample_seqns)]
    return df


def wide_to_long(mtsm_df: pd.DataFrame, predm_df: pd.DataFrame, seqn: int) -> pd.DataFrame:
    """Convert wide-format PhysioNet data to long-format minute-level time series
    for a single participant.

    Args:
        mtsm_df: Wide-format PAXMTSM (triaxial MIMS) for all participants
        predm_df: Wide-format PAXPREDM (wear prediction) for all participants
        seqn: Participant ID

    Returns:
        DataFrame with columns [timestamp, enmo, wear, sleep] indexed by timestamp
    """
    # Filter for this participant
    mtsm_p = mtsm_df[mtsm_df["SEQN"] == seqn].copy()
    predm_p = predm_df[predm_df["SEQN"] == seqn].copy()

    if mtsm_p.empty:
        return pd.DataFrame()

    # Identify minute columns (min_0001 to min_1440)
    min_cols = [c for c in mtsm_p.columns if c.startswith("min_")]
    if not min_cols:
        # Try alternative column naming
        min_cols = [c for c in mtsm_p.columns if c not in ["SEQN", "day"]]

    records = []
    for day_idx in range(len(mtsm_p)):
        mtsm_row = mtsm_p.iloc[day_idx]
        predm_row = predm_p.iloc[day_idx] if day_idx < len(predm_p) else None

        # Create a fake base date (CosinorAge just needs relative timing)
        base_date = datetime(2012, 1, 1) + timedelta(days=day_idx)

        for i, col in enumerate(min_cols):
            mims_val = mtsm_row[col]
            if pd.isna(mims_val) or mims_val < 0:
                continue

            timestamp = base_date + timedelta(minutes=i)

            # Get wear prediction: 1=wake, 2=sleep, 3=non-wear, 4=unknown
            pred_val = predm_row[col] if predm_row is not None and col in predm_row.index else 1
            if pd.isna(pred_val):
                pred_val = 3  # treat NaN as non-wear

            pred_val = int(pred_val)
            wear = 1 if pred_val in [1, 2] else 0
            sleep = 1 if pred_val == 2 else 0

            # Use MIMS directly as activity metric.
            # CosinorAge was trained on ENMO (mg) from UK Biobank.
            # NHANES MIMS and ENMO are both minute-level activity summaries
            # with similar circadian patterns. The cosinor analysis extracts
            # relative rhythm parameters (MESOR, amplitude, acrophase) which
            # are then fed into the CosinorAge regression model.
            enmo = mims_val

            records.append({
                "timestamp": timestamp,
                "enmo": enmo,
                "wear": wear,
                "sleep": sleep,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index("timestamp")
    return df


def convert_participant_to_csv(
    mtsm_df: pd.DataFrame,
    predm_df: pd.DataFrame,
    seqn: int,
    output_dir: Path = ACCEL_CSV_DIR,
) -> Path:
    """Convert a single participant's data to CSV for GenericDataHandler.

    Returns path to the CSV file, or None if participant has insufficient data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"participant_{seqn}.csv"

    if output_path.exists():
        return output_path

    df = wide_to_long(mtsm_df, predm_df, seqn)

    if df.empty:
        return None

    # Check minimum data requirement: at least 4 days
    n_days = len(np.unique(df.index.date))
    if n_days < 4:
        return None

    # Save with timestamp column for GenericDataHandler
    df.reset_index().to_csv(output_path, index=False)
    return output_path


def get_valid_seqns(subject_info_path: Path = DATA_DIR / "subject-info.csv") -> list:
    """Get list of participant IDs from subject-info.csv."""
    df = pd.read_csv(subject_info_path)
    seqn_col = "SEQN" if "SEQN" in df.columns else df.columns[0]
    return df[seqn_col].astype(int).tolist()


def batch_convert(seqns: list = None, max_participants: int = None):
    """Convert multiple participants from PhysioNet wide format to CSV.

    Args:
        seqns: List of participant IDs. If None, uses all from subject-info.csv.
        max_participants: Limit number of participants (for testing).
    """
    from tqdm import tqdm

    if seqns is None:
        seqns = get_valid_seqns()
    if max_participants:
        seqns = seqns[:max_participants]

    print(f"Loading PhysioNet accelerometry files...")
    mtsm_df = load_physionet_wide(DATA_DIR / "nhanes_1440_PAXMTSM.xpt", seqns)
    predm_df = load_physionet_wide(DATA_DIR / "nhanes_1440_PAXPREDM.xpt", seqns)
    print(f"  PAXMTSM: {len(mtsm_df)} rows, PAXPREDM: {len(predm_df)} rows")

    converted = 0
    skipped = 0
    for seqn in tqdm(seqns, desc="Converting participants"):
        path = convert_participant_to_csv(mtsm_df, predm_df, seqn)
        if path:
            converted += 1
        else:
            skipped += 1

    print(f"\nDone: {converted} converted, {skipped} skipped (insufficient data)")
    return converted


if __name__ == "__main__":
    # Test with 5 participants first
    print("=== Test: Converting 5 participants ===")
    batch_convert(max_participants=5)

    # Show a sample
    csv_files = list(ACCEL_CSV_DIR.glob("*.csv"))
    if csv_files:
        sample = pd.read_csv(csv_files[0], nrows=10)
        print(f"\nSample from {csv_files[0].name}:")
        print(sample)
