import tempfile

import numpy as np
import pandas as pd
import pytest
from baseline_trainer import XGBoostAgePredictor


def make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    return pd.DataFrame({
        "age": age,
        "hba1c": rng.uniform(4.5, 9.0, n),
        "nlr": rng.uniform(1.0, 8.0, n),
        "bmi": rng.uniform(18.0, 40.0, n),
    })


FEATURES = ["hba1c", "nlr", "bmi"]


def test_fit_returns_metrics():
    df = make_df()
    predictor = XGBoostAgePredictor(FEATURES)
    metrics = predictor.fit(df)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "pearson_r" in metrics
    assert "best_params" in metrics
    assert metrics["mae"] > 0


def test_predict_shape():
    df = make_df()
    predictor = XGBoostAgePredictor(FEATURES)
    predictor.fit(df)
    preds = predictor.predict(df)
    assert len(preds) == len(df)


def test_evaluate_metrics_range():
    df = make_df()
    predictor = XGBoostAgePredictor(FEATURES)
    predictor.fit(df)
    metrics = predictor.evaluate(df)
    assert metrics["mae"] > 0
    assert -1.0 <= metrics["pearson_r"] <= 1.0


def test_save_and_load_roundtrip():
    df = make_df()
    predictor = XGBoostAgePredictor(FEATURES)
    predictor.fit(df)
    preds_before = predictor.predict(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.json"
        predictor.save(path)
        loaded = XGBoostAgePredictor.load(path)
        preds_after = loaded.predict(df)
    np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_drops_nulls_before_training():
    df = make_df()
    df.loc[:10, "hba1c"] = np.nan
    predictor = XGBoostAgePredictor(FEATURES)
    metrics = predictor.fit(df)
    assert metrics["mae"] > 0


def test_raises_on_missing_label_col():
    df = make_df().drop(columns=["age"])
    predictor = XGBoostAgePredictor(FEATURES)
    with pytest.raises(ValueError, match="Missing columns"):
        predictor.fit(df)
