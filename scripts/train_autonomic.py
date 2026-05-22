"""
Run HRV extraction (if needed) then train XGBoost model on Autonomic Aging dataset.

Expects data at data/autonomic_aging/ (WFDB format from PhysioNet).
If data/autonomic_features.parquet already exists, skips extraction.

Usage:
    python scripts/train_autonomic.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from baseline_trainer import XGBoostAgePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_DIR / "data" / "autonomic_features.parquet"
DATA_DIR = PROJECT_DIR / "data" / "autonomic_aging"
SUBJECT_CSV = DATA_DIR / "subject-info.csv"
MODEL_PATH = PROJECT_DIR / "models" / "autonomic_model.json"
SHAP_PATH = PROJECT_DIR / "figures" / "autonomic_shap_beeswarm.png"

HRV_FEATURES = [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_lf_hf",
    "hrv_mean_hr", "hrv_dfa_alpha1",
]
DEMO_FEATURES = ["bmi"]


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sex_col = next(
        (c for c in df.columns if c.lower() in ("sex", "gender")), None
    )
    if sex_col:
        df["sex_encoded"] = (
            df[sex_col].map({"M": 1, "F": 0, "Male": 1, "Female": 0}).astype(float)
        )
    return df


def main() -> None:
    if not FEATURES_PATH.exists():
        log.info("autonomic_features.parquet not found, running HRV extraction pipeline...")
        from autonomic_aging_processor import batch_process, load_subject_metadata

        if not DATA_DIR.exists():
            log.error(
                "Autonomic Aging data directory not found: %s\n"
                "Download from https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/",
                DATA_DIR,
            )
            return

        features_df = batch_process(data_dir=DATA_DIR)
        if features_df.empty:
            log.error("No features extracted. Check WFDB files in %s.", DATA_DIR)
            return

        metadata = load_subject_metadata(str(SUBJECT_CSV))
        merged = metadata.merge(features_df, on="subject_id", how="inner")
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(FEATURES_PATH, index=False)
        log.info("Saved autonomic_features.parquet: %d rows", len(merged))
        df = merged
    else:
        df = pd.read_parquet(FEATURES_PATH)
        log.info("Loaded autonomic_features.parquet: %d rows, %d cols", *df.shape)

    df = encode_sex(df)

    feature_cols = [
        f for f in HRV_FEATURES + DEMO_FEATURES + ["sex_encoded"] if f in df.columns
    ]
    log.info(
        "Training autonomic_hrv with %d features on %d participants",
        len(feature_cols), len(df),
    )

    predictor = XGBoostAgePredictor(feature_cols, label_col="age", model_name="autonomic_hrv")
    metrics = predictor.fit(df)

    print(f"\nModel: autonomic_hrv")
    print(f"  MAE:            {metrics['mae']:.2f} years")
    print(f"  RMSE:           {metrics['rmse']:.2f} years")
    print(f"  Pearson r:      {metrics['pearson_r']:.3f}")
    print(f"  Published MAE:  5.62 years (Schumann et al. 2023)")
    print(f"  Params:         {metrics['best_params']}")

    predictor.save(str(MODEL_PATH))
    predictor.shap_summary(df, save_path=str(SHAP_PATH))
    log.info("Done. Model → %s", MODEL_PATH)


if __name__ == "__main__":
    main()
