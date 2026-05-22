"""
Train XGBoost biological age model on NHANES features.

Loads nhanes_features.parquet (full circadian+sleep+labs) if available,
falls back to nhanes_participants.parquet (labs only) if not.

Usage:
    python scripts/train_nhanes.py
    python scripts/train_nhanes.py --labs-only
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from baseline_trainer import XGBoostAgePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
FULL_FEATURES_PATH = PROJECT_DIR / "data" / "nhanes_features.parquet"
LABS_ONLY_PATH = PROJECT_DIR / "data" / "nhanes_participants.parquet"
FIGURES_DIR = PROJECT_DIR / "figures"

LAB_FEATURES = [
    "hba1c", "wbc", "neutrophils", "lymphocytes", "nlr",
    "bmi", "waist_circumference", "height", "weight",
]

CIRCADIAN_FEATURES = [
    "cosinor_mesor", "cosinor_amplitude", "cosinor_acrophase", "cosinor_acrophase_time",
    "nonparam_M10_mean", "nonparam_L5_mean", "nonparam_RA_mean",
    "nonparam_IS_mean", "nonparam_IV_mean",
    "sleep_TST_mean", "sleep_WASO_mean", "sleep_SOL_mean", "sleep_SRI_mean",
    "physical_activity_sedentary_mean", "physical_activity_light_mean",
    "physical_activity_moderate_mean", "physical_activity_vigorous_mean",
    "cosinorage",
]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).astype(float)
    if "race_ethnicity" in df.columns:
        df["race_ethnicity"] = (
            df["race_ethnicity"].astype("category").cat.codes.astype(float)
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NHANES XGBoost age model")
    parser.add_argument(
        "--labs-only",
        action="store_true",
        help="Force labs-only feature set even if full features are available",
    )
    args = parser.parse_args()

    use_full = FULL_FEATURES_PATH.exists() and not args.labs_only
    if use_full:
        df = pd.read_parquet(FULL_FEATURES_PATH)
        log.info("Loaded nhanes_features.parquet: %d rows, %d cols", *df.shape)
    else:
        if args.labs_only:
            log.info("--labs-only flag set, using nhanes_participants.parquet")
        else:
            log.warning("nhanes_features.parquet not found, falling back to labs only")
        df = pd.read_parquet(LABS_ONLY_PATH)
        log.info("Loaded nhanes_participants.parquet: %d rows, %d cols", *df.shape)

    df = encode_categoricals(df)

    demo_features = ["sex", "race_ethnicity"]
    if use_full:
        candidate_features = LAB_FEATURES + CIRCADIAN_FEATURES + demo_features
        model_name = "nhanes_full"
    else:
        candidate_features = LAB_FEATURES + demo_features
        model_name = "nhanes_labs_only"

    feature_cols = [f for f in candidate_features if f in df.columns]
    model_path = PROJECT_DIR / "models" / f"{model_name}.json"
    shap_path = FIGURES_DIR / f"{model_name}_shap_beeswarm.png"

    log.info(
        "Training '%s' with %d features on %d participants",
        model_name, len(feature_cols), len(df),
    )

    predictor = XGBoostAgePredictor(feature_cols, label_col="age", model_name=model_name)
    metrics = predictor.fit(df)

    print(f"\nModel: {model_name}")
    print(f"  MAE:       {metrics['mae']:.2f} years")
    print(f"  RMSE:      {metrics['rmse']:.2f} years")
    print(f"  Pearson r: {metrics['pearson_r']:.3f}")
    print(f"  Params:    {metrics['best_params']}")

    predictor.save(str(model_path))
    predictor.shap_summary(df, save_path=str(shap_path))
    log.info("Done. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
