"""
baseline_trainer.py

Shared XGBoost age-prediction trainer used by both NHANES and
Autonomic Aging training scripts.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

log = logging.getLogger(__name__)


class XGBoostAgePredictor:
    """Train, evaluate, save, and load an XGBoost chronological-age predictor.

    Usage:
        predictor = XGBoostAgePredictor(["hba1c", "nlr", "bmi"])
        metrics = predictor.fit(df)          # stratified split + CV tuning
        preds   = predictor.predict(df)
        predictor.save("models/nhanes.json")
        loaded  = XGBoostAgePredictor.load("models/nhanes.json")
    """

    def __init__(
        self,
        feature_cols: list[str],
        label_col: str = "age",
        model_name: str = "model",
    ) -> None:
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.model_name = model_name
        self.model: XGBRegressor | None = None
        self._metrics: dict | None = None

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        required = self.feature_cols + [self.label_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df[required].dropna()

    def fit(self, df: pd.DataFrame) -> dict:
        """Stratified 70/15/15 split, 5-fold CV grid search, retrain on train+val.

        Returns dict with mae, rmse, pearson_r, best_params (evaluated on test set).
        """
        clean = self._clean(df)
        X = clean[self.feature_cols].values
        y = clean[self.label_col].values

        age_decade = (y // 10).astype(int)

        X_temp, X_test, y_temp, y_test, dec_temp, _ = train_test_split(
            X, y, age_decade, test_size=0.15, random_state=42, stratify=age_decade
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=dec_temp
        )

        param_grid = {
            "max_depth": [3, 5, 7],
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        cv = GridSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0,
        )
        cv.fit(X_train, y_train)
        best_params = cv.best_params_

        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        self.model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        self.model.fit(X_trainval, y_trainval)

        metrics = self._compute_metrics(X_test, y_test)
        metrics["best_params"] = best_params
        self._metrics = metrics
        log.info(
            "%s test set MAE=%.2f yr  RMSE=%.2f yr  r=%.3f  params=%s",
            self.model_name, metrics["mae"], metrics["rmse"],
            metrics["pearson_r"], best_params,
        )
        return metrics

    def _compute_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        r, _ = pearsonr(preds, y)
        return {"mae": mae, "rmse": rmse, "pearson_r": float(r)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        X = df[self.feature_cols].dropna()[self.feature_cols].values
        return self.model.predict(X)

    def evaluate(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate().")
        clean = self._clean(df)
        return self._compute_metrics(
            clean[self.feature_cols].values,
            clean[self.label_col].values,
        )

    def shap_summary(self, df: pd.DataFrame, save_path: str | None = None) -> None:
        if self.model is None:
            raise RuntimeError("Call fit() before shap_summary().")
        clean = self._clean(df)
        X = clean[self.feature_cols].values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_cols, show=False)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("SHAP beeswarm saved to %s", save_path)
        plt.close()

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Call fit() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        meta = {
            "feature_cols": self.feature_cols,
            "label_col": self.label_col,
            "model_name": self.model_name,
            "metrics": self._metrics,
        }
        with open(Path(path).with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Saved %s + .meta.json", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostAgePredictor":
        meta_path = Path(path).with_suffix(".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        predictor = cls(
            feature_cols=meta["feature_cols"],
            label_col=meta["label_col"],
            model_name=meta["model_name"],
        )
        predictor.model = XGBRegressor()
        predictor.model.load_model(path)
        predictor._metrics = meta.get("metrics")
        return predictor
