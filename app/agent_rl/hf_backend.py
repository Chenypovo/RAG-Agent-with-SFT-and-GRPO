from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional

from app.agent.llm import CompleteFn


class TransformersChatBackend:
    """Hugging Face chat backend with an optional inference-only PEFT adapter."""

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        enable_thinking: bool = False,
        seed: int = 42,
        revision: Optional[str] = None,
        adapter_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16 or float32")

        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.revision = revision.strip() if revision and revision.strip() else None
        self.adapter_path = (
            adapter_path.strip() if adapter_path and adapter_path.strip() else None
        )
        if tokenizer is None or model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.manual_seed(seed)
            load_dtype: Any = "auto" if dtype == "auto" else getattr(torch, dtype)
            load_options: Dict[str, Any] = {}
            if self.revision is not None:
                load_options["revision"] = self.revision
            tokenizer = AutoTokenizer.from_pretrained(model_name, **load_options)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=load_dtype,
                **load_options,
            )
        self.resolved_commit = _resolved_commit(model, tokenizer, self.revision)
        self.adapter_provenance = None
        if self.adapter_path is not None:
            self.adapter_provenance = _adapter_provenance(self.adapter_path)
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                self.adapter_path,
                is_trainable=False,
            )
        self.tokenizer = tokenizer
        self.model = model.eval()

    def make_complete_fn(
        self,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> CompleteFn:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        def complete(system_prompt: str, user_prompt: str) -> str:
            return self.complete(
                system_prompt,
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        return complete

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        template_options: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            template_options["enable_thinking"] = self.enable_thinking
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **template_options)
        except TypeError:
            template_options.pop("enable_thinking", None)
            prompt = self.tokenizer.apply_chat_template(messages, **template_options)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_device = self.model.get_input_embeddings().weight.device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        generation_options: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_options.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        if self.tokenizer.pad_token_id is not None:
            generation_options["pad_token_id"] = self.tokenizer.pad_token_id
        with torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_options)
        prompt_length = encoded["input_ids"].shape[1]
        output_tokens = generated[0, prompt_length:]
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()


def _resolved_commit(model: Any, tokenizer: Any, revision: Optional[str]) -> Optional[str]:
    model_config = getattr(model, "config", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    candidates = (
        getattr(model_config, "_commit_hash", None),
        getattr(model, "_commit_hash", None),
        getattr(tokenizer, "_commit_hash", None),
        tokenizer_kwargs.get("_commit_hash") if isinstance(tokenizer_kwargs, dict) else None,
        revision,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", candidate.strip()):
            return candidate.strip().lower()
    return None


def _adapter_provenance(adapter_path: str) -> Dict[str, Any]:
    adapter_dir = Path(adapter_path).expanduser()
    if not adapter_dir.is_dir():
        raise ValueError(f"adapter_path must be a local PEFT adapter directory: {adapter_path}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter_config.json not found in adapter directory: {adapter_dir}")

    weight_paths = sorted(
        path
        for pattern in ("adapter_model*.safetensors", "adapter_model*.bin")
        for path in adapter_dir.glob(pattern)
        if path.is_file()
    )
    if not weight_paths:
        raise ValueError(f"adapter weights not found in adapter directory: {adapter_dir}")

    return {
        "path": str(adapter_dir.resolve()),
        "adapter_config_sha256": _sha256_file(config_path),
        "weight_files": [
            {
                "path": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in weight_paths
        ],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
