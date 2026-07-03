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


def test_convert_to_gguf_invokes_llama_cpp_subprocess(tmp_path, monkeypatch):
    from gguf_exporter import convert_to_gguf

    invocations = []

    def _fake_run(cmd, check, **kwargs):
        invocations.append(cmd)
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr("subprocess.run", _fake_run)

    hf_dir = tmp_path / "merged"
    hf_dir.mkdir()
    out_path = tmp_path / "model.gguf"
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()

    convert_to_gguf(
        hf_dir=str(hf_dir),
        out_path=str(out_path),
        quantization="q4_k_m",
        llama_cpp_dir=str(llama_dir),
    )

    assert len(invocations) == 2, "expected convert + quantize invocations"
    convert_cmd = invocations[0]
    assert "convert_hf_to_gguf.py" in " ".join(convert_cmd)
    assert str(hf_dir) in convert_cmd
    quantize_cmd = invocations[1]
    assert any("quantize" in part for part in quantize_cmd)
    assert "q4_k_m" in " ".join(quantize_cmd).lower()


def test_convert_to_gguf_raises_when_llama_cpp_dir_missing(tmp_path, monkeypatch):
    from gguf_exporter import convert_to_gguf

    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    hf_dir = tmp_path / "merged"
    hf_dir.mkdir()

    with pytest.raises(RuntimeError, match="LLAMA_CPP_DIR"):
        convert_to_gguf(
            hf_dir=str(hf_dir),
            out_path=str(tmp_path / "m.gguf"),
            quantization="q4_k_m",
        )


def test_convert_to_gguf_reads_env_when_arg_absent(tmp_path, monkeypatch):
    from gguf_exporter import convert_to_gguf

    invocations = []
    monkeypatch.setattr("subprocess.run",
                        lambda cmd, check, **kw: (invocations.append(cmd) or type("R", (), {"returncode": 0})()))

    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    monkeypatch.setenv("LLAMA_CPP_DIR", str(llama_dir))

    hf_dir = tmp_path / "merged"
    hf_dir.mkdir()

    convert_to_gguf(hf_dir=str(hf_dir), out_path=str(tmp_path / "m.gguf"))
    assert len(invocations) == 2
