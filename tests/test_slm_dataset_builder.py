import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_dataset_builder import build_dataset, row_to_pair


def _fake_rag():
    rag = MagicMock()
    rag.search.return_value = [
        {
            "context": "Inflammaging context from NLR research.",
            "source": "Liu et al. 2020",
            "category": "Inflammation",
            "title": "NLR and inflammaging",
        },
        {
            "context": "Circadian fragmentation drives epigenetic aging.",
            "source": "Smith et al. 2021",
            "category": "Aging",
            "title": "Circadian and aging",
        },
    ]
    return rag


def _good_row() -> dict:
    return {
        "seqn": 71166,
        "age": 55,
        "sex": "Male",
        "hba1c": 7.2,
        "nlr": 5.8,
        "wbc": 8.1,
        "cosinorage_advance": 13.7,
        "nonparam_IS_mean": 0.41,
        "nonparam_IV_mean": 0.82,
        "nonparam_RA_mean": 0.71,
        "sleep_TST_mean": 330.0,
        "sleep_SRI_mean": 42.0,
    }


class TestRowToPair:
    def test_valid_row_produces_pair(self):
        pair = row_to_pair(_good_row(), _fake_rag())
        assert pair is not None
        assert pair["seqn"] == 71166
        assert "prompt" in pair and "response" in pair
        assert "7.2" in pair["prompt"]  # HbA1c surfaces in prompt
        assert "Diabetic" in pair["response"]
        assert "Liu et al. 2020" in pair["response"]

    def test_missing_hba1c_returns_none(self):
        row = _good_row()
        row["hba1c"] = None
        assert row_to_pair(row, _fake_rag()) is None

    def test_missing_nlr_returns_none(self):
        row = _good_row()
        row["nlr"] = None
        assert row_to_pair(row, _fake_rag()) is None

    def test_nan_hba1c_returns_none(self):
        row = _good_row()
        row["hba1c"] = float("nan")
        assert row_to_pair(row, _fake_rag()) is None


class TestBuildDataset:
    def test_writes_jsonl_with_train_and_eval_split(self, tmp_path):
        rows = []
        for i in range(20):
            r = _good_row()
            r["seqn"] = 10000 + i
            r["age"] = 40 + i
            rows.append(r)
        df = pd.DataFrame(rows)

        parquet_path = tmp_path / "mini.parquet"
        df.to_parquet(parquet_path)

        out_path = tmp_path / "out.jsonl"

        stats = build_dataset(
            parquet_path=str(parquet_path),
            output_path=str(out_path),
            rag=_fake_rag(),
            eval_fraction=0.2,
            seed=42,
        )

        assert stats["n_skipped"] == 0
        assert stats["n_train"] + stats["n_eval"] == 20
        assert stats["n_eval"] == 4  # 20 * 0.2

        lines = out_path.read_text().splitlines()
        assert len(lines) == 20
        splits = [json.loads(line)["split"] for line in lines]
        assert splits.count("train") == 16
        assert splits.count("eval") == 4

    def test_skips_rows_missing_required_biomarkers(self, tmp_path):
        rows = [_good_row(), _good_row(), _good_row()]
        rows[1]["hba1c"] = None
        rows[2]["nlr"] = float("nan")
        df = pd.DataFrame(rows)
        df["seqn"] = [1, 2, 3]

        parquet_path = tmp_path / "mini.parquet"
        df.to_parquet(parquet_path)
        out_path = tmp_path / "out.jsonl"

        stats = build_dataset(
            parquet_path=str(parquet_path),
            output_path=str(out_path),
            rag=_fake_rag(),
            eval_fraction=0.0,
            seed=42,
        )

        assert stats["n_skipped"] == 2
        assert stats["n_train"] + stats["n_eval"] == 1

    def test_deterministic_split_with_seed(self, tmp_path):
        rows = [_good_row() for _ in range(10)]
        for i, r in enumerate(rows):
            r["seqn"] = i
        df = pd.DataFrame(rows)
        df.to_parquet(tmp_path / "mini.parquet")

        out1 = tmp_path / "out1.jsonl"
        out2 = tmp_path / "out2.jsonl"
        build_dataset(str(tmp_path / "mini.parquet"), str(out1), _fake_rag(), 0.3, seed=42)
        build_dataset(str(tmp_path / "mini.parquet"), str(out2), _fake_rag(), 0.3, seed=42)
        assert out1.read_text() == out2.read_text()
