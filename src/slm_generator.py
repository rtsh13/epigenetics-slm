"""
Inference-time wrapper around a fine-tuned SLM.

Loads either a GGUF file via llama-cpp-python or a HuggingFace directory
via transformers. Uses slm_prompt.build_prompt as the single source of
truth for the instruction prompt, guaranteeing byte-identical formatting
between training and inference.
"""

from pathlib import Path

import slm_prompt
from slm_prompt import build_prompt


class SLMGenerator:
    def __init__(self, model_path: str, backend: str = "auto", n_ctx: int = 4096):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._backend_name = self._resolve_backend(model_path, backend)
        self._model = self._load_model()

    @staticmethod
    def _resolve_backend(model_path: str, backend: str) -> str:
        if backend != "auto":
            if backend not in ("gguf", "hf"):
                raise ValueError(f"unknown backend: {backend}")
            return backend
        if model_path.endswith(".gguf"):
            return "gguf"
        return "hf"

    def _load_model(self):
        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"model path does not exist: {self._model_path}")
        if self._backend_name == "gguf":
            from llama_cpp import Llama
            return Llama(model_path=self._model_path, n_ctx=self._n_ctx, verbose=False)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        model = AutoModelForCausalLM.from_pretrained(self._model_path)
        return (tokenizer, model)

    def _call_backend(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        if self._backend_name == "gguf":
            out = self._model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                echo=False,
                stop=["<|eot_id|>"],
            )
            return out["choices"][0]["text"]
        tokenizer, model = self._model
        inputs = tokenizer(prompt, return_tensors="pt")
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text

    def generate(
        self,
        biomarkers: dict,
        rag_chunks: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        raw_prompt = build_prompt(biomarkers, rag_chunks)
        prompt = slm_prompt.build_inference_prompt(raw_prompt)
        return self._call_backend(prompt, max_new_tokens=max_new_tokens, temperature=temperature)