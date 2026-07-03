import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_generator import SLMGenerator
from slm_prompt import build_prompt


BIOMARKERS = {
    "age": 71, "sex": "Female", "hba1c": 5.8, "nlr": 4.2, "wbc": 7.1,
    "cosinorage_advance": 2.4, "is_value": 0.42, "iv_value": 0.71,
    "ra_value": 0.74, "tst_minutes": 372.0, "sri": 58.0,
}

RAG_CHUNKS = [
    {"context": "chunk1 body", "source": "Src1", "category": "Aging", "title": "T1"},
    {"context": "chunk2 body", "source": "Src2", "category": "Sleep", "title": "T2"},
]


def _stub_generator(monkeypatch, canned="AGING: ok."):
    gen = SLMGenerator.__new__(SLMGenerator)
    gen._backend_name = "stub"
    gen._model = MagicMock()
    gen._call_backend = MagicMock(return_value=canned)
    return gen


def test_generate_feeds_backend_the_shared_prompt(monkeypatch):
    gen = _stub_generator(monkeypatch)
    gen.generate(BIOMARKERS, RAG_CHUNKS, max_new_tokens=128, temperature=0.1)
    expected_prompt = build_prompt(BIOMARKERS, RAG_CHUNKS)
    (prompt_arg,), kwargs = gen._call_backend.call_args
    assert prompt_arg == expected_prompt, "prompt fed to backend must be byte-identical to slm_prompt.build_prompt(...)"


def test_generate_returns_backend_output(monkeypatch):
    gen = _stub_generator(monkeypatch, canned="AGING: favorable.")
    out = gen.generate(BIOMARKERS, RAG_CHUNKS)
    assert out == "AGING: favorable."


def test_generate_passes_generation_kwargs(monkeypatch):
    gen = _stub_generator(monkeypatch)
    gen.generate(BIOMARKERS, RAG_CHUNKS, max_new_tokens=42, temperature=0.7)
    _, kwargs = gen._call_backend.call_args
    assert kwargs.get("max_new_tokens") == 42
    assert kwargs.get("temperature") == 0.7
