"""Train the agent controller with single-GPU QLoRA.

The input is the JSONL emitted by ``scripts/build_agent_sft_data.py``. A dry
run validates the config and data without importing the GPU training stack,
downloading a model, or creating an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CONFIG_SCHEMA_VERSION = "agent-sft-train-config-v1"
DRY_RUN_SCHEMA_VERSION = "agent-sft-dry-run-v1"
TRAIN_REPORT_SCHEMA_VERSION = "agent-sft-train-report-v1"
REQUIRED_TRAINING_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result < 1:
        raise ValueError(f"{name} must be a finite number in [0, 1)")
    return result


def load_config(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {path}: {exc}") from exc
    config = dict(_mapping(value, "config"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")

    model = _mapping(config.get("model"), "model")
    if not isinstance(model.get("name_or_path"), str) or not model["name_or_path"].strip():
        raise ValueError("model.name_or_path must be a non-empty string")
    revision = model.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("model.revision must be null or a non-empty string")
    if model.get("torch_dtype") not in ("bfloat16", "float16"):
        raise ValueError("model.torch_dtype must be bfloat16 or float16")
    if not isinstance(model.get("trust_remote_code"), bool):
        raise ValueError("model.trust_remote_code must be a boolean")

    data = _mapping(config.get("data"), "data")
    if not isinstance(data.get("train_file"), str) or not data["train_file"].strip():
        raise ValueError("data.train_file must be a non-empty string")
    _positive_int(data.get("max_seq_length"), "data.max_seq_length")
    preserve = _non_negative_int(
        data.get("preserve_prompt_prefix_tokens"),
        "data.preserve_prompt_prefix_tokens",
    )
    if preserve >= data["max_seq_length"]:
        raise ValueError("data.preserve_prompt_prefix_tokens must be smaller than max_seq_length")
    _mapping(data.get("chat_template_kwargs", {}), "data.chat_template_kwargs")

    quantization = _mapping(config.get("quantization"), "quantization")
    if quantization.get("bits") != 4:
        raise ValueError("quantization.bits must be 4 for QLoRA")
    if quantization.get("quant_type") not in ("nf4", "fp4"):
        raise ValueError("quantization.quant_type must be nf4 or fp4")
    if not isinstance(quantization.get("double_quant"), bool):
        raise ValueError("quantization.double_quant must be a boolean")

    lora = _mapping(config.get("lora"), "lora")
    _positive_int(lora.get("r"), "lora.r")
    _positive_int(lora.get("alpha"), "lora.alpha")
    _rate(lora.get("dropout"), "lora.dropout")
    targets = lora.get("target_modules")
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes, bytearray))
        or not targets
        or not all(isinstance(item, str) and item.strip() for item in targets)
    ):
        raise ValueError("lora.target_modules must be a non-empty list of strings")

    training = _mapping(config.get("training"), "training")
    _positive_number(training.get("num_train_epochs"), "training.num_train_epochs")
    _positive_int(training.get("per_device_train_batch_size"), "training.per_device_train_batch_size")
    _positive_int(training.get("gradient_accumulation_steps"), "training.gradient_accumulation_steps")
    _positive_number(training.get("learning_rate"), "training.learning_rate")
    _rate(training.get("warmup_ratio"), "training.warmup_ratio")
    _positive_number(training.get("weight_decay"), "training.weight_decay")
    _positive_number(training.get("max_grad_norm"), "training.max_grad_norm")
    _positive_int(training.get("logging_steps"), "training.logging_steps")
    _positive_int(training.get("save_steps"), "training.save_steps")
    _positive_int(training.get("save_total_limit"), "training.save_total_limit")
    _non_negative_int(training.get("seed"), "training.seed")
    if not isinstance(training.get("gradient_checkpointing"), bool):
        raise ValueError("training.gradient_checkpointing must be a boolean")
    if training.get("optimizer") not in ("paged_adamw_8bit", "adamw_torch"):
        raise ValueError("training.optimizer must be paged_adamw_8bit or adamw_torch")

    output = _mapping(config.get("output"), "output")
    if not isinstance(output.get("dir"), str) or not output["dir"].strip():
        raise ValueError("output.dir must be a non-empty string")


def _normalize_messages(row: Mapping[str, Any], line_number: int) -> List[Dict[str, str]]:
    messages = row.get("messages")
    if messages is None and "prompt" in row and "response" in row:
        prompt, response = row.get("prompt"), row.get("response")
        if not isinstance(prompt, list) or not isinstance(response, str):
            raise ValueError(f"line {line_number}: prompt_response row has invalid prompt or response")
        messages = [*prompt, {"role": "assistant", "content": response}]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"line {line_number}: messages must contain at least two items")

    normalized: List[Dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"line {line_number}: message {index} must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"line {line_number}: message {index} has unsupported role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"line {line_number}: message {index} content must be non-empty")
        normalized.append({"role": role, "content": content})

    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"line {line_number}: final message must be assistant")
    if any(message["role"] == "assistant" for message in normalized[:-1]):
        raise ValueError(f"line {line_number}: only the final message may use the assistant role")
    if not any(message["role"] == "user" for message in normalized[:-1]):
        raise ValueError(f"line {line_number}: prompt must include a user message")
    return normalized


def load_training_rows(path: Path) -> Tuple[List[List[Dict[str, str]]], Dict[str, Any]]:
    rows: List[List[Dict[str, str]]] = []
    character_lengths: List[int] = []
    task_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"line {line_number}: training row must be an object")
            messages = _normalize_messages(raw, line_number)
            rows.append(messages)
            character_lengths.append(sum(len(item["content"]) for item in messages))
            if isinstance(raw.get("task_id"), str) and raw["task_id"]:
                task_ids.add(raw["task_id"])
    if not rows:
        raise ValueError(f"training file contains no examples: {path}")
    return rows, {
        "examples": len(rows),
        "unique_task_ids": len(task_ids),
        "characters": {
            "min": min(character_lengths),
            "max": max(character_lengths),
            "mean": sum(character_lengths) / len(character_lengths),
        },
    }


def dependency_status() -> Dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in REQUIRED_TRAINING_PACKAGES}


def _resolve_path(value: str, *, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[1]
    repository_candidate = project_root / path
    if repository_candidate.exists() or config_path.parent.name == "agent_rl":
        return repository_candidate
    return config_path.parent / path


def make_dry_run_report(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    train_path: Path,
    output_dir: Path,
    data_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    training = config["training"]
    examples = int(data_stats["examples"])
    examples_per_update = int(training["per_device_train_batch_size"]) * int(training["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(examples / examples_per_update)
    optimizer_updates = math.ceil(updates_per_epoch * float(training["num_train_epochs"]))
    dependencies = dependency_status()
    return {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "valid": True,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "train_path": str(train_path),
        "train_sha256": _sha256(train_path),
        "output_dir": str(output_dir),
        "model": dict(config["model"]),
        "data": dict(data_stats),
        "plan": {
            "max_seq_length": config["data"]["max_seq_length"],
            "examples_per_optimizer_update": examples_per_update,
            "updates_per_epoch": updates_per_epoch,
            "estimated_optimizer_updates": optimizer_updates,
            "num_train_epochs": training["num_train_epochs"],
        },
        "dependencies": dependencies,
        "ready_for_training": all(dependencies.values()),
        "effects": {"model_downloaded": False, "gpu_initialized": False, "output_created": False},
    }


def _tokenize_rows(
    tokenizer: Any,
    rows: Sequence[Sequence[Mapping[str, str]]],
    *,
    max_seq_length: int,
    preserve_prompt_prefix_tokens: int,
    chat_template_kwargs: Mapping[str, Any],
) -> Tuple[List[Dict[str, List[int]]], Dict[str, int]]:
    tokenized: List[Dict[str, List[int]]] = []
    truncated = 0
    for row_index, messages in enumerate(rows, start=1):
        prompt_text = tokenizer.apply_chat_template(
            list(messages[:-1]), tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        full_text = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False, **chat_template_kwargs
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(f"row {row_index}: chat template does not preserve the prompt prefix")
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"row {row_index}: tokenized chat prompt is not a prefix of the full row")
        response_ids = full_ids[len(prompt_ids) :]
        if not response_ids:
            raise ValueError(f"row {row_index}: assistant response has no trainable tokens")
        if len(response_ids) >= max_seq_length:
            raise ValueError(f"row {row_index}: assistant response alone exceeds max_seq_length")
        available_prompt = max_seq_length - len(response_ids)
        if len(prompt_ids) > available_prompt:
            truncated += 1
            prefix_size = min(preserve_prompt_prefix_tokens, available_prompt)
            suffix_size = available_prompt - prefix_size
            prompt_ids = prompt_ids[:prefix_size] + (prompt_ids[-suffix_size:] if suffix_size else [])
        input_ids = prompt_ids + response_ids
        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + response_ids,
        })
    return tokenized, {"examples": len(tokenized), "truncated_prompts": truncated}


def run_training(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    train_path: Path,
    output_dir: Path,
    rows: Sequence[Sequence[Mapping[str, str]]],
    resume_from_checkpoint: Optional[str],
    overwrite_output_dir: bool,
) -> Dict[str, Any]:
    missing = [name for name, present in dependency_status().items() if not present]
    if missing:
        raise RuntimeError(
            "missing QLoRA dependencies: " + ", ".join(missing)
            + "; install configs/agent_rl/requirements-sft.txt on top of requirements.txt"
        )

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite_output_dir and resume_from_checkpoint is None:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite-output-dir or --resume-from-checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config, data_config = config["model"], config["data"]
    quant_config, lora_config, training_config = config["quantization"], config["lora"], config["training"]
    set_seed(int(training_config["seed"]))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[model_config["torch_dtype"]]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured bfloat16, but this GPU does not support it")
    revision_kwargs = {"revision": model_config["revision"]} if model_config.get("revision") else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"], trust_remote_code=model_config["trust_remote_code"], **revision_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_rows, tokenization_stats = _tokenize_rows(
        tokenizer,
        rows,
        max_seq_length=int(data_config["max_seq_length"]),
        preserve_prompt_prefix_tokens=int(data_config["preserve_prompt_prefix_tokens"]),
        chat_template_kwargs=data_config.get("chat_template_kwargs", {}),
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_config["quant_type"],
        bnb_4bit_use_double_quant=quant_config["double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        trust_remote_code=model_config["trust_remote_code"],
        **revision_kwargs,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_config["gradient_checkpointing"])
    model = get_peft_model(model, LoraConfig(
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_config["target_modules"]),
    ))

    class TokenizedDataset:
        def __len__(self) -> int:
            return len(tokenized_rows)

        def __getitem__(self, index: int) -> Dict[str, List[int]]:
            return tokenized_rows[index]

    def collate(features: Sequence[Mapping[str, Sequence[int]]]) -> Dict[str, Any]:
        padded_length = math.ceil(max(len(feature["input_ids"]) for feature in features) / 8) * 8
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            padding = padded_length - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [tokenizer.pad_token_id] * padding)
            attention_masks.append(list(feature["attention_mask"]) + [0] * padding)
            labels.append(list(feature["labels"]) + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    training_argument_values = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": overwrite_output_dir,
        "num_train_epochs": float(training_config["num_train_epochs"]),
        "per_device_train_batch_size": int(training_config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training_config["gradient_accumulation_steps"]),
        "learning_rate": float(training_config["learning_rate"]),
        "warmup_ratio": float(training_config["warmup_ratio"]),
        "weight_decay": float(training_config["weight_decay"]),
        "max_grad_norm": float(training_config["max_grad_norm"]),
        "logging_steps": int(training_config["logging_steps"]),
        "save_steps": int(training_config["save_steps"]),
        "save_total_limit": int(training_config["save_total_limit"]),
        "gradient_checkpointing": bool(training_config["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": training_config["optimizer"],
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
        "tf32": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": int(training_config["seed"]),
        "data_seed": int(training_config["seed"]),
    }
    # Transformers 4.x accepts overwrite_output_dir; 5.x removed it. Output
    # safety is enforced above, so filtering the removed compatibility option
    # does not weaken overwrite protection.
    supported_arguments = inspect.signature(TrainingArguments).parameters
    arguments = TrainingArguments(
        **{
            key: value
            for key, value in training_argument_values.items()
            if key in supported_arguments
        }
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=TokenizedDataset(), data_collator=collate)
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    report = {
        "schema_version": TRAIN_REPORT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "train_path": str(train_path),
        "train_sha256": _sha256(train_path),
        "output_dir": str(output_dir),
        "model": dict(model_config),
        "tokenization": tokenization_stats,
        "train_metrics": dict(result.metrics),
        "adapter_path": str(output_dir / "adapter"),
        "runtime": {
            "total_seconds_this_process": time.perf_counter() - run_started,
            "training_seconds": result.metrics.get("train_runtime"),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Single-GPU QLoRA SFT for the strict-JSON agent controller")
    parser.add_argument("--config", required=True, help="training config JSON")
    parser.add_argument("--input", help="override data.train_file")
    parser.add_argument("--output-dir", help="override output.dir")
    parser.add_argument("--model", help="override model.name_or_path, for example a downloaded ModelScope directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", help="optional dry-run JSON report path")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"config does not exist: {config_path}")
    try:
        config = load_config(config_path)
        if args.model:
            config["model"]["name_or_path"] = args.model
            # A local mirror cannot be addressed by the upstream Hub revision.
            if Path(args.model).expanduser().is_dir():
                config["model"]["revision"] = None
        train_path = Path(args.input).expanduser().resolve() if args.input else _resolve_path(config["data"]["train_file"], config_path=config_path)
        if not train_path.is_file():
            raise ValueError(f"training input does not exist: {train_path}")
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _resolve_path(config["output"]["dir"], config_path=config_path)
        rows, data_stats = load_training_rows(train_path)
        if args.dry_run:
            report = make_dry_run_report(
                config=config, config_path=config_path, train_path=train_path, output_dir=output_dir, data_stats=data_stats
            )
            if args.report:
                report_path = Path(args.report).expanduser().resolve()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            if args.report:
                raise ValueError("--report is only valid with --dry-run")
            report = run_training(
                config=config,
                config_path=config_path,
                train_path=train_path,
                output_dir=output_dir,
                rows=rows,
                resume_from_checkpoint=args.resume_from_checkpoint,
                overwrite_output_dir=args.overwrite_output_dir,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
