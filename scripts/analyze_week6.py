"""
Week 6 analysis: ablations, feature importance, biological interpretation.

Tasks covered:
  6.1  SHAP beeswarm - NHANES full model
  6.2  SHAP beeswarm - Autonomic Aging model (if model file exists)
  6.3  Category ablation table (MAE per category vs all combined)
  6.4  Cross-category feature importance ranking (top 20)
  6.5  Predicted vs actual scatterplots (both models)
  6.6  Biological interpretation (biomarkers vs CosinorAge acceleration)
  6.7  Combined score exploration (simulated HRV component if autonomic absent)

Usage:
    python scripts/analyze_week6.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from baseline_trainer import XGBoostAgePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
FIGURES_DIR.mkdir(exist_ok=True)

CIRCADIAN_FEATURES = [
    "cosinor_mesor", "cosinor_amplitude", "cosinor_acrophase", "cosinor_acrophase_time",
    "nonparam_M10_mean", "nonparam_L5_mean", "nonparam_RA_mean",
    "nonparam_IS_mean", "nonparam_IV_mean",
]
SLEEP_FEATURES = ["sleep_TST_mean", "sleep_WASO_mean", "sleep_SOL_mean", "sleep_SRI_mean"]
ACTIVITY_FEATURES = [
    "physical_activity_sedentary_mean", "physical_activity_light_mean",
    "physical_activity_moderate_mean", "physical_activity_vigorous_mean",
]
LAB_FEATURES = [
    "hba1c", "wbc", "neutrophils", "lymphocytes", "nlr",
    "bmi", "waist_circumference", "height", "weight",
]
DEMO_FEATURES = ["sex", "race_ethnicity"]

CATEGORY_MAP = {
    **{f: "Circadian" for f in CIRCADIAN_FEATURES},
    **{f: "Sleep" for f in SLEEP_FEATURES},
    **{f: "Activity" for f in ACTIVITY_FEATURES},
    "hba1c": "Metabolism",
    "wbc": "Inflammation", "neutrophils": "Inflammation",
    "lymphocytes": "Inflammation", "nlr": "Inflammation",
    "bmi": "Body", "waist_circumference": "Body",
    "height": "Body", "weight": "Body",
    "cosinorage": "CosinorAge (leakage risk)",
    "sex": "Demographics", "race_ethnicity": "Demographics",
}

CATEGORY_COLORS = {
    "Circadian": "#2196F3",
    "Sleep": "#9C27B0",
    "Activity": "#4CAF50",
    "Metabolism": "#FF9800",
    "Inflammation": "#F44336",
    "Body": "#795548",
    "CosinorAge (leakage risk)": "#607D8B",
    "Demographics": "#9E9E9E",
}


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sex" in df.columns and df["sex"].dtype == object:
        df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).astype(float)
    if "race_ethnicity" in df.columns and df["race_ethnicity"].dtype == object:
        df["race_ethnicity"] = df["race_ethnicity"].astype("category").cat.codes.astype(float)
    return df


def quick_train(df: pd.DataFrame, feature_cols: list[str], name: str) -> dict:
    """Train XGBoost with fixed params for ablation speed."""
    cols_present = [c for c in feature_cols if c in df.columns]
    if len(cols_present) < 2:
        log.warning("Skipping '%s': only %d features available", name, len(cols_present))
        return {}
    clean = df[cols_present + ["age"]].dropna()
    if len(clean) < 50:
        log.warning("Skipping '%s': only %d samples after dropna", name, len(clean))
        return {}
    X = clean[cols_present].values
    y = clean["age"].values
    age_decade = (y // 10).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=age_decade
    )
    model = XGBRegressor(
        max_depth=5, n_estimators=200, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    r, _ = pearsonr(preds, y_test)
    log.info("  %-42s  n=%4d  MAE=%5.2f  r=%.3f", name, len(clean), mae, r)
    return {
        "name": name,
        "n_features": len(cols_present),
        "n_samples": len(clean),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": r,
    }


def task_61_shap(predictor: XGBoostAgePredictor, df: pd.DataFrame, out_name: str) -> None:
    log.info("Task 6.1/6.2: SHAP beeswarm - %s", out_name)
    predictor.shap_summary(df, save_path=str(FIGURES_DIR / f"{out_name}_shap_beeswarm.png"))


def task_63_ablations(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Task 6.3: Category ablation table")
    groups = [
        ("Circadian only", CIRCADIAN_FEATURES + DEMO_FEATURES),
        ("Sleep only", SLEEP_FEATURES + DEMO_FEATURES),
        ("Activity only", ACTIVITY_FEATURES + DEMO_FEATURES),
        ("Metabolism only (HbA1c)", ["hba1c", "bmi", "waist_circumference", "height", "weight"] + DEMO_FEATURES),
        ("Inflammation only (NLR+WBC)", ["wbc", "neutrophils", "lymphocytes", "nlr"] + DEMO_FEATURES),
        ("All wearable (no CosinorAge)", CIRCADIAN_FEATURES + SLEEP_FEATURES + ACTIVITY_FEATURES + DEMO_FEATURES),
        ("All labs", LAB_FEATURES + DEMO_FEATURES),
        ("Full (no CosinorAge)", LAB_FEATURES + CIRCADIAN_FEATURES + SLEEP_FEATURES + ACTIVITY_FEATURES + DEMO_FEATURES),
        ("Full + CosinorAge [leakage]", LAB_FEATURES + CIRCADIAN_FEATURES + SLEEP_FEATURES + ACTIVITY_FEATURES + ["cosinorage"] + DEMO_FEATURES),
    ]
    rows = [quick_train(df, cols, name) for name, cols in groups]
    rows = [r for r in rows if r]
    return pd.DataFrame(rows)


def task_63_plot(ablation_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#607D8B" if "leakage" in n else "#1565C0" for n in ablation_df["name"]]
    bars = ax.bar(range(len(ablation_df)), ablation_df["mae"], color=colors, width=0.6)
    ax.set_xticks(range(len(ablation_df)))
    ax.set_xticklabels(ablation_df["name"], rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("MAE (years)")
    ax.set_title(
        "Category Ablation — NHANES XGBoost Biological Age Prediction\n"
        "[leakage] = CosinorAge is a derived biological age, not a raw biomarker"
    )
    for bar, mae in zip(bars, ablation_df["mae"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{mae:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "nhanes_ablation_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Ablation bar chart saved")


def task_64_feature_importance(predictor: XGBoostAgePredictor) -> None:
    log.info("Task 6.4: Cross-category feature importance ranking")
    feature_cols = predictor.feature_cols
    importances = predictor.model.feature_importances_

    feat_imp = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
            "category": [CATEGORY_MAP.get(f, "Other") for f in feature_cols],
        })
        .sort_values("importance", ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        feat_imp["feature"],
        feat_imp["importance"],
        color=[CATEGORY_COLORS.get(c, "#9E9E9E") for c in feat_imp["category"]],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (XGBoost gain)")
    ax.set_title("Top 20 Features by Category — NHANES Full Model")

    seen: set[str] = set()
    patches = []
    for cat, color in CATEGORY_COLORS.items():
        if cat in feat_imp["category"].values and cat not in seen:
            patches.append(mpatches.Patch(color=color, label=cat))
            seen.add(cat)
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "nhanes_feature_importance_ranked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature importance figure saved")


def task_65_pred_vs_actual(predictor: XGBoostAgePredictor, df: pd.DataFrame, title: str, out_name: str) -> None:
    log.info("Task 6.5: Predicted vs actual - %s", title)
    cols_needed = [c for c in predictor.feature_cols if c in df.columns]
    clean = df[cols_needed + ["age"]].dropna()

    temp_predictor = XGBoostAgePredictor(cols_needed, label_col="age", model_name=predictor.model_name)
    temp_predictor.model = predictor.model
    preds = temp_predictor.predict(clean)

    actual = clean["age"].values
    mae = float(np.mean(np.abs(preds - actual)))
    r, _ = pearsonr(preds, actual)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, preds, alpha=0.3, s=10, color="#1565C0")
    lims = [min(actual.min(), preds.min()) - 2, max(actual.max(), preds.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Chronological Age (years)")
    ax.set_ylabel("Predicted Age (years)")
    ax.set_title(f"Predicted vs Actual Age\n{title}   MAE={mae:.2f} yr  r={r:.3f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{out_name}_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Scatterplot saved: %s_pred_vs_actual.png", out_name)


def task_66_biological_interpretation(df: pd.DataFrame) -> None:
    log.info("Task 6.6: Biological interpretation")
    if "cosinorage" not in df.columns:
        log.warning("cosinorage not in df, skipping 6.6")
        return

    df = df.copy()
    df["age_acceleration"] = df["cosinorage"] - df["age"]

    biomarkers = [
        ("HbA1c (%)", "hba1c", "positive expected (higher HbA1c -> faster aging)"),
        ("NLR", "nlr", "positive expected (inflammation -> faster aging)"),
        ("WBC (x10^3/uL)", "wbc", "positive expected"),
        ("Sleep TST (hrs)", "sleep_TST_mean", "negative expected (more sleep -> slower aging)"),
        ("Circadian IV", "nonparam_IV_mean", "positive expected (irregular rhythm -> faster aging)"),
        ("Circadian IS", "nonparam_IS_mean", "negative expected (stable rhythm -> slower aging)"),
    ]
    available = [(label, col, note) for label, col, note in biomarkers if col in df.columns]
    if not available:
        log.warning("No biomarker columns found for 6.6")
        return

    ncols = 3
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.array(axes).flatten()

    print("\nBiomarker correlations with CosinorAge acceleration:")
    print(f"  {'Biomarker':<28} {'Pearson r':>10} {'Spearman rho':>13} {'n':>6}  Direction check")

    for i, (label, col, note) in enumerate(available):
        sub = df[["age_acceleration", col]].dropna()
        x = sub[col].values
        y = sub["age_acceleration"].values
        r, p = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        print(f"  {label:<28} {r:>10.3f} {rho:>13.3f} {len(sub):>6}  {note}")

        ax = axes[i]
        ax.scatter(x, y, alpha=0.2, s=8, color="#1565C0")
        m, b = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, m * xfit + b, "r-", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("CosinorAge Acceleration (yr)", fontsize=9)
        ax.set_title(f"r={r:.2f}, rho={rho:.2f}, p={p:.3f}", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Biomarker Correlations with CosinorAge Acceleration", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "nhanes_biomarker_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Biomarker correlations saved")


def task_67_combined_score(predictor: XGBoostAgePredictor, df: pd.DataFrame) -> None:
    log.info("Task 6.7: Combined score exploration")
    cols_needed = [c for c in predictor.feature_cols if c in df.columns]
    clean = df[cols_needed + ["age"]].dropna()
    temp = XGBoostAgePredictor(cols_needed, label_col="age", model_name=predictor.model_name)
    temp.model = predictor.model
    nhanes_preds = temp.predict(clean)
    actual = clean["age"].values

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 5.5, size=len(actual))
    sim_hrv_preds = 0.85 * actual + 0.15 * actual.mean() + noise

    scores = {
        "NHANES only": nhanes_preds,
        "Combined 60/40 (simulated HRV)": 0.6 * nhanes_preds + 0.4 * sim_hrv_preds,
        "Combined 50/50 (simulated HRV)": 0.5 * nhanes_preds + 0.5 * sim_hrv_preds,
    }

    print("\nCombined score exploration (HRV component is simulated, not real):")
    for label, preds in scores.items():
        mae = float(np.mean(np.abs(preds - actual)))
        r, _ = pearsonr(preds, actual)
        print(f"  {label:<38}  MAE={mae:.2f} yr  r={r:.3f}")

    fig, axes = plt.subplots(1, len(scores), figsize=(6 * len(scores), 6))
    for ax, (label, preds) in zip(axes, scores.items()):
        mae = float(np.mean(np.abs(preds - actual)))
        r, _ = pearsonr(preds, actual)
        ax.scatter(actual, preds, alpha=0.3, s=10, color="#1565C0")
        lims = [actual.min() - 2, actual.max() + 2]
        ax.plot(lims, lims, "r--", linewidth=1.5)
        ax.set_xlabel("Chronological Age (years)")
        ax.set_ylabel("Predicted Age (years)")
        ax.set_title(f"{label}\nMAE={mae:.2f} yr  r={r:.3f}", fontsize=9)
    fig.suptitle("Combined Score Exploration (HRV simulated until autonomic model is trained)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "nhanes_combined_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Combined score figure saved")


def main() -> None:
    nhanes_path = DATA_DIR / "nhanes_features.parquet"
    if not nhanes_path.exists():
        log.error("nhanes_features.parquet not found. Run train_nhanes.py first.")
        sys.exit(1)

    df = pd.read_parquet(nhanes_path)
    df = encode_categoricals(df)
    log.info("Loaded nhanes_features.parquet: %d rows, %d cols", *df.shape)

    nhanes_model_path = MODELS_DIR / "nhanes_full.json"
    if not nhanes_model_path.exists():
        log.error("nhanes_full.json not found. Run train_nhanes.py first.")
        sys.exit(1)

    nhanes_predictor = XGBoostAgePredictor.load(str(nhanes_model_path))
    log.info(
        "Loaded nhanes_full model (training MAE=%.2f, r=%.3f)",
        nhanes_predictor._metrics["mae"],
        nhanes_predictor._metrics["pearson_r"],
    )

    task_61_shap(nhanes_predictor, df, "nhanes_full")

    ablation_df = task_63_ablations(df)
    ablation_df.to_csv(FIGURES_DIR / "nhanes_ablation_table.csv", index=False)
    print("\nCategory Ablation Table:")
    print(ablation_df[["name", "n_features", "n_samples", "mae", "rmse", "pearson_r"]].to_string(index=False))
    task_63_plot(ablation_df)

    task_64_feature_importance(nhanes_predictor)

    task_65_pred_vs_actual(nhanes_predictor, df, "NHANES Full Model", "nhanes_full")

    task_66_biological_interpretation(df)

    task_67_combined_score(nhanes_predictor, df)

    auto_model_path = MODELS_DIR / "autonomic_model.json"
    auto_feat_path = DATA_DIR / "autonomic_features.parquet"
    if auto_model_path.exists() and auto_feat_path.exists():
        log.info("Task 6.2: Autonomic model found, running analysis")
        auto_predictor = XGBoostAgePredictor.load(str(auto_model_path))
        auto_df = pd.read_parquet(auto_feat_path)
        sex_col = next((c for c in auto_df.columns if c.lower() in ("sex", "gender")), None)
        if sex_col:
            auto_df["sex_encoded"] = (
                auto_df[sex_col].map({"M": 1, "F": 0, "Male": 1, "Female": 0}).astype(float)
            )
        task_61_shap(auto_predictor, auto_df, "autonomic")
        task_65_pred_vs_actual(auto_predictor, auto_df, "Autonomic HRV Model", "autonomic")
    else:
        log.info(
            "Task 6.2: autonomic_model.json or autonomic_features.parquet not found. "
            "Run train_autonomic.py to enable autonomic analysis."
        )

    print("\nWeek 6 complete. Figures saved to figures/:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")
    print("Ablation CSV: figures/nhanes_ablation_table.csv")


if __name__ == "__main__":
    main()
