import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_merge_lora_calls_peft_and_saves(tmp_path, monkeypatch):
    from gguf_exporter import merge_lora

    calls = {}

    fake_peft = types.SimpleNamespace()

    class _PeftModel:
        @staticmethod
        def from_pretrained(base, adapter_dir):
            calls["from_pretrained"] = (base, adapter_dir)

            def _merge_and_unload():
                calls["merged"] = True
                return base
            base.merge_and_unload = _merge_and_unload
            return base

    fake_peft.PeftModel = _PeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    fake_transformers = types.SimpleNamespace()

    class _AMForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["base_loaded"] = model_id
            m = types.SimpleNamespace()
            def _save(path):
                calls["model_saved_to"] = path
            m.save_pretrained = _save
            return m

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            calls["tokenizer_loaded"] = model_id
            t = types.SimpleNamespace()
            def _save(path):
                calls["tokenizer_saved_to"] = path
            t.save_pretrained = _save
            return t

    fake_transformers.AutoModelForCausalLM = _AMForCausalLM
    fake_transformers.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    out_dir = tmp_path / "merged"
    merge_lora(
        base_model_id="unsloth/Llama-3.2-1B-Instruct",
        adapter_dir=str(tmp_path / "adapter"),
        out_dir=str(out_dir),
    )

    assert calls["base_loaded"] == "unsloth/Llama-3.2-1B-Instruct"
    assert calls["tokenizer_loaded"] == "unsloth/Llama-3.2-1B-Instruct"
    assert calls["from_pretrained"][1] == str(tmp_path / "adapter")
    assert calls["merged"] is True
    assert calls["model_saved_to"] == str(out_dir)
    assert calls["tokenizer_saved_to"] == str(out_dir)
