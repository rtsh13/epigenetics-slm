import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_trainer import build_training_args, format_example, load_jsonl_dataset


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_jsonl_dataset_returns_only_requested_split(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [
        {"seqn": 1, "prompt": "a", "response": "b", "split": "train"},
        {"seqn": 2, "prompt": "c", "response": "d", "split": "eval"},
        {"seqn": 3, "prompt": "e", "response": "f", "split": "train"},
    ])
    train = load_jsonl_dataset(str(p), "train")
    assert [r["seqn"] for r in train] == [1, 3]
    ev = load_jsonl_dataset(str(p), "eval")
    assert [r["seqn"] for r in ev] == [2]


def test_load_jsonl_dataset_defaults_to_train_when_split_missing(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"seqn": 1, "prompt": "a", "response": "b"}])
    train = load_jsonl_dataset(str(p), "train")
    assert len(train) == 1


def test_format_example_uses_llama_chat_template():
    rec = {"prompt": "USER_TEXT", "response": "ASSISTANT_TEXT"}
    out = format_example(rec)
    assert "text" in out
    text = out["text"]
    assert text.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>user<|end_header_id|>" in text
    assert "USER_TEXT" in text
    assert "<|start_header_id|>assistant<|end_header_id|>" in text
    assert "ASSISTANT_TEXT" in text
    assert text.endswith("<|eot_id|>")


def test_build_training_args_defaults():
    args = build_training_args(output_dir="models/slm_lora")
    assert args["output_dir"] == "models/slm_lora"
    assert args["num_train_epochs"] == 3
    assert args["per_device_train_batch_size"] == 2
    assert args["gradient_accumulation_steps"] == 4
    assert args["learning_rate"] == 2e-4


def test_build_training_args_overrides():
    args = build_training_args(
        output_dir="out", epochs=5, batch_size=1, grad_accum=8, lr=1e-4,
    )
    assert args["num_train_epochs"] == 5
    assert args["per_device_train_batch_size"] == 1
    assert args["gradient_accumulation_steps"] == 8
    assert args["learning_rate"] == 1e-4


def test_train_wires_helpers_and_calls_trainer(tmp_path, monkeypatch):
    from slm_trainer import train

    ds_path = tmp_path / "d.jsonl"
    _write_jsonl(ds_path, [
        {"seqn": 1, "prompt": "p1", "response": "r1", "split": "train"},
        {"seqn": 2, "prompt": "p2", "response": "r2", "split": "eval"},
    ])

    calls = {}

    fake_unsloth = types.SimpleNamespace()
    fake_model = types.SimpleNamespace(save_pretrained=lambda path: None)
    fake_tokenizer = types.SimpleNamespace(save_pretrained=lambda path: None)

    class _FastLM:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["from_pretrained"] = kwargs
            return fake_model, fake_tokenizer

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["peft_kwargs"] = kwargs
            return model

    fake_unsloth.FastLanguageModel = _FastLM
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

    class _Trainer:
        def __init__(self, **kwargs):
            calls["trainer_kwargs"] = kwargs

        def train(self):
            calls["train_called"] = True

    fake_trl = types.SimpleNamespace(SFTTrainer=_Trainer)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    fake_datasets = types.SimpleNamespace(
        Dataset=types.SimpleNamespace(from_list=lambda rows: rows),
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

    fake_transformers = types.SimpleNamespace(TrainingArguments=_FakeTrainingArguments)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    train(
        dataset_path=str(ds_path),
        output_dir=str(tmp_path / "out"),
        epochs=1, batch_size=1, grad_accum=1,
    )

    assert calls["from_pretrained"]["model_name"] == "unsloth/Llama-3.2-1B-Instruct"
    assert calls["train_called"] is True
    trainer_kwargs = calls["trainer_kwargs"]
    assert trainer_kwargs["model"] is fake_model
    tokenizer_arg = trainer_kwargs.get("tokenizer") or trainer_kwargs.get("processing_class")
    assert tokenizer_arg is fake_tokenizer
    training_args = trainer_kwargs["args"]
    assert isinstance(training_args, _FakeTrainingArguments), (
        "SFTTrainer must receive a TrainingArguments instance, not a plain dict"
    )
    assert training_args._kwargs["num_train_epochs"] == 1
    train_dataset = trainer_kwargs["train_dataset"]
    assert len(train_dataset) == 1
    assert train_dataset[0]["text"].startswith("<|begin_of_text|>")
