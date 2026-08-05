"""Single-GPU multi-turn GRPO for the strict-JSON retrieval controller.

The policy starts from an SFT LoRA adapter.  Each task is rolled out several
times from the same deterministic environment state; verifier-backed episode
returns are normalized only inside that task group.  Loss is applied only to
generated action tokens.  The initial SFT adapter remains loaded as a frozen
reference adapter while the policy adapter is updated.

``--dry-run`` validates data, configuration and dependencies without importing
the GPU training stack or creating the output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_rl.actions import AgentAction  # noqa: E402
from app.agent_rl.adapters import build_minimal_registry  # noqa: E402
from app.agent_rl.evaluation import aggregate_policy_rollouts  # noqa: E402
from app.agent_rl.finalizers import FrozenAnswerFinalizer  # noqa: E402
from app.agent_rl.grpo import (  # noqa: E402
    grpo_clipped_loss,
    group_relative_advantages,
    mean_dict,
    sum_reward_components,
)
from app.agent_rl.hf_backend import (  # noqa: E402
    TransformersChatBackend,
    _adapter_provenance,
)
from app.agent_rl.policies import (  # noqa: E402
    PolicyDecision,
    PromptOnlyPolicy,
    _SYSTEM_PROMPT,
)
from app.agent_rl.rewards import RewardConfig  # noqa: E402
from app.agent_rl.rollouts import PolicyRollout, run_policy_episode  # noqa: E402
from app.agent_rl.tasks import AgentRLTask, load_tasks  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402


CONFIG_SCHEMA_VERSION = "agent-grpo-train-config-v1"
DRY_RUN_SCHEMA_VERSION = "agent-grpo-dry-run-v1"
TRAIN_REPORT_SCHEMA_VERSION = "agent-grpo-train-report-v1"
POLICY_VERSION = "multi-turn-grpo-controller-v1"
REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "rank_bm25",
)
VERSIONED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number


def _probability(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = _non_negative_number(value, name)
    lower_ok = number >= 0 if allow_zero else number > 0
    if not lower_ok or number > 1:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return number


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
    for key in ("name_or_path", "init_adapter"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError(f"model.{key} must be a non-empty string")
    revision = model.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("model.revision must be null or a non-empty string")
    if model.get("torch_dtype") not in ("bfloat16", "float16"):
        raise ValueError("model.torch_dtype must be bfloat16 or float16")
    if not isinstance(model.get("trust_remote_code"), bool):
        raise ValueError("model.trust_remote_code must be a boolean")
    if model.get("quant_type") not in ("nf4", "fp4"):
        raise ValueError("model.quant_type must be nf4 or fp4")
    if not isinstance(model.get("double_quant"), bool):
        raise ValueError("model.double_quant must be a boolean")

    finalizer = _mapping(config.get("finalizer"), "finalizer")
    if not isinstance(finalizer.get("name_or_path"), str) or not finalizer["name_or_path"].strip():
        raise ValueError("finalizer.name_or_path must be a non-empty string")
    if finalizer.get("torch_dtype") not in ("bfloat16", "float16", "float32"):
        raise ValueError("finalizer.torch_dtype must be bfloat16, float16 or float32")
    _positive_int(finalizer.get("max_new_tokens"), "finalizer.max_new_tokens")

    data = _mapping(config.get("data"), "data")
    for key in ("data_dir", "task_file"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"data.{key} must be a non-empty string")
    _non_negative_int(data.get("max_tasks"), "data.max_tasks")
    _positive_int(data.get("retrieval_top_k"), "data.retrieval_top_k")

    rollout = _mapping(config.get("rollout"), "rollout")
    if _positive_int(rollout.get("group_size"), "rollout.group_size") < 2:
        raise ValueError("rollout.group_size must be at least 2")
    _positive_int(rollout.get("tasks_per_update"), "rollout.tasks_per_update")
    _positive_int(rollout.get("max_steps"), "rollout.max_steps")
    _positive_int(rollout.get("max_new_tokens"), "rollout.max_new_tokens")
    _positive_number(rollout.get("temperature"), "rollout.temperature")
    _probability(rollout.get("top_p"), "rollout.top_p")
    _non_negative_int(rollout.get("top_k"), "rollout.top_k")
    if (
        float(rollout["temperature"]) != 1.0
        or float(rollout["top_p"]) != 1.0
        or int(rollout["top_k"]) != 0
    ):
        raise ValueError(
            "on-policy log probabilities require rollout.temperature=1, "
            "rollout.top_p=1 and rollout.top_k=0"
        )

    reward = _mapping(config.get("reward"), "reward")
    expected_reward_keys = set(RewardConfig.__dataclass_fields__)
    if set(reward) != expected_reward_keys:
        missing = sorted(expected_reward_keys - set(reward))
        extra = sorted(set(reward) - expected_reward_keys)
        raise ValueError(f"reward keys mismatch; missing={missing}, extra={extra}")
    for key, value in reward.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"reward.{key} must be a finite number")
    if float(reward["valid_action_format"]) + float(reward["tool_call_cost"]) > 0:
        raise ValueError(
            "a valid tool call must not earn positive net process reward"
        )

    training = _mapping(config.get("training"), "training")
    _positive_int(training.get("max_updates"), "training.max_updates")
    _positive_int(
        training.get("optimization_epochs"),
        "training.optimization_epochs",
    )
    _positive_number(training.get("learning_rate"), "training.learning_rate")
    _non_negative_number(training.get("beta"), "training.beta")
    clip_epsilon = _probability(training.get("clip_epsilon"), "training.clip_epsilon")
    if clip_epsilon >= 1:
        raise ValueError("training.clip_epsilon must be smaller than 1")
    _positive_int(training.get("max_seq_length"), "training.max_seq_length")
    preserve = _non_negative_int(
        training.get("preserve_prompt_prefix_tokens"),
        "training.preserve_prompt_prefix_tokens",
    )
    if preserve >= training["max_seq_length"]:
        raise ValueError("training.preserve_prompt_prefix_tokens must be smaller than max_seq_length")
    if training["max_seq_length"] <= rollout["max_new_tokens"]:
        raise ValueError("training.max_seq_length must exceed rollout.max_new_tokens")
    _positive_number(training.get("max_grad_norm"), "training.max_grad_norm")
    _positive_int(training.get("save_steps"), "training.save_steps")
    _non_negative_int(training.get("seed"), "training.seed")

    output = _mapping(config.get("output"), "output")
    if not isinstance(output.get("dir"), str) or not output["dir"].strip():
        raise ValueError("output.dir must be a non-empty string")


def dependency_status() -> Dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in REQUIRED_PACKAGES}


def dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _file_provenance(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _code_provenance(config_path: Path) -> Dict[str, Any]:
    return {
        "train_script": _file_provenance(Path(__file__).resolve()),
        "grpo_module": _file_provenance(PROJECT_ROOT / "app" / "agent_rl" / "grpo.py"),
        "config": _file_provenance(config_path),
    }


def _base_model_provenance(
    name_or_path: str,
    *,
    revision: Optional[str],
) -> Dict[str, Any]:
    model_path = Path(name_or_path).expanduser()
    if not model_path.is_dir():
        return {
            "source": "hub_revision",
            "name_or_path": name_or_path,
            "revision": revision,
            "local_artifact_hashes": [],
        }

    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"local base model is missing config.json: {model_path}")
    index_paths = sorted(model_path.glob("*.index.json"))
    weight_paths = sorted({
        path
        for pattern in ("model*.safetensors", "pytorch_model*.bin")
        for path in model_path.glob(pattern)
        if path.is_file()
    })
    if not weight_paths:
        raise RuntimeError(f"local base model has no recognized weight files: {model_path}")
    artifact_paths = [config_path, *index_paths, *weight_paths]
    return {
        "source": "local_directory",
        "name_or_path": str(model_path.resolve()),
        "revision": revision,
        "local_artifact_hashes": [_file_provenance(path) for path in artifact_paths],
    }


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolved_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    data_dir = _resolve_path(str(config["data"]["data_dir"]))
    return {
        "data_dir": data_dir,
        "task_file": data_dir / str(config["data"]["task_file"]),
        "bm25": data_dir / "bm25.json",
        "manifest": data_dir / "manifest.json",
        "init_adapter": _resolve_path(str(config["model"]["init_adapter"])),
        "output_dir": _resolve_path(str(config["output"]["dir"])),
    }


def _adapter_artifacts_ready(path: Path) -> bool:
    if not path.is_dir() or not (path / "adapter_config.json").is_file():
        return False
    return any(
        candidate.is_file()
        for pattern in ("adapter_model*.safetensors", "adapter_model*.bin")
        for candidate in path.glob(pattern)
    )


def make_dry_run_report(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> Dict[str, Any]:
    paths = _resolved_paths(config)
    required_files = {
        key: path
        for key, path in paths.items()
        if key in {"task_file", "bm25", "manifest"}
    }
    missing_files = [str(path) for path in required_files.values() if not path.is_file()]
    tasks: List[AgentRLTask] = []
    if not missing_files:
        tasks = load_tasks(paths["task_file"])
        max_tasks = int(config["data"]["max_tasks"])
        if max_tasks:
            tasks = tasks[:max_tasks]
    deps = dependency_status()
    adapter_ready = _adapter_artifacts_ready(paths["init_adapter"])
    rollout = config["rollout"]
    training = config["training"]
    episodes = (
        int(training["max_updates"])
        * int(rollout["tasks_per_update"])
        * int(rollout["group_size"])
    )
    return {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "valid": not missing_files and bool(tasks),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "resolved_config_sha256": _canonical_sha256(config),
        "paths": {key: str(path) for key, path in paths.items()},
        "data": {
            "available_tasks": len(tasks),
            "missing_files": missing_files,
            "input_sha256": {
                key: _sha256(path)
                for key, path in required_files.items()
                if path.is_file()
            },
        },
        "plan": {
            "updates": int(training["max_updates"]),
            "optimization_epochs": int(training["optimization_epochs"]),
            "tasks_per_update": int(rollout["tasks_per_update"]),
            "group_size": int(rollout["group_size"]),
            "planned_episodes": episodes,
            "maximum_decisions": episodes * int(rollout["max_steps"]),
            "optimization_epochs_per_rollout_batch": int(
                training["optimization_epochs"]
            ),
            "reference_policy": "frozen copy of the initial SFT adapter",
            "loss_scope": "generated action tokens only",
        },
        "dependencies": deps,
        "init_adapter_ready": adapter_ready,
        "static_preflight_passed": (
            not missing_files
            and bool(tasks)
            and adapter_ready
            and all(deps.values())
        ),
        "effects": {
            "model_loaded": False,
            "gpu_initialized": False,
            "output_created": False,
        },
        "unchecked_runtime_requirements": [
            "CUDA and bfloat16 support",
            "model/adapter compatibility",
            "available GPU memory",
            "one-update forward/backward/save/reload smoke",
        ],
    }


@dataclass(frozen=True)
class ActionTrace:
    prompt_ids: Tuple[int, ...]
    completion_ids: Tuple[int, ...]
    old_log_probs: Tuple[float, ...]
    reference_log_probs: Tuple[float, ...]
    prompt_truncated: bool


def _activate_adapter(model: Any, adapter: str) -> None:
    if adapter not in {"policy", "reference"}:
        raise ValueError(f"unknown adapter: {adapter}")
    model.set_adapter(adapter)
    for name, parameter in model.named_parameters():
        if ".reference." in name:
            parameter.requires_grad_(False)
        elif ".policy." in name:
            parameter.requires_grad_(adapter == "policy")


def _assert_adapter_training_state(model: Any, *, active_adapter: str) -> None:
    policy_parameters = []
    reference_parameters = []
    unexpected_trainable = []
    for name, parameter in model.named_parameters():
        if ".policy." in name:
            policy_parameters.append(parameter)
        elif ".reference." in name:
            reference_parameters.append(parameter)
        elif parameter.requires_grad:
            unexpected_trainable.append(name)
    if not policy_parameters or not reference_parameters:
        raise RuntimeError("policy and reference adapters must both expose parameters")
    if any(parameter.requires_grad for parameter in reference_parameters):
        raise RuntimeError("reference adapter must remain frozen")
    policy_should_train = active_adapter == "policy"
    if any(parameter.requires_grad != policy_should_train for parameter in policy_parameters):
        raise RuntimeError(
            f"policy adapter training state does not match active adapter {active_adapter!r}"
        )
    if unexpected_trainable:
        raise RuntimeError(
            "non-policy parameters are trainable: " + ", ".join(unexpected_trainable[:5])
        )


def _disable_all_dropout(model: Any) -> int:
    import torch

    changed = 0
    for module in model.modules():
        if (
            isinstance(module, torch.nn.modules.dropout._DropoutNd)
            and float(module.p) != 0.0
        ):
            module.p = 0.0
            changed += 1
    return changed


def _assert_dropout_disabled(model: Any) -> None:
    import torch

    active = [
        float(module.p)
        for module in model.modules()
        if isinstance(module, torch.nn.modules.dropout._DropoutNd)
        and float(module.p) != 0.0
    ]
    if active:
        raise RuntimeError(f"dropout must be disabled for GRPO, found p={active[:5]}")


def _assert_policy_matches_reference(model: Any) -> None:
    import torch

    def tensors_for(adapter: str) -> Dict[str, Any]:
        marker = f".{adapter}."
        return {
            name.replace(marker, ".__adapter__.", 1): tensor.detach()
            for name, tensor in model.state_dict().items()
            if marker in name
        }

    policy = tensors_for("policy")
    reference = tensors_for("reference")
    if not policy or policy.keys() != reference.keys():
        raise RuntimeError("policy/reference adapter tensor sets do not match")
    mismatched = [
        name
        for name in policy
        if policy[name].shape != reference[name].shape
        or not torch.equal(policy[name], reference[name])
    ]
    if mismatched:
        raise RuntimeError(
            "policy and reference adapters must start from identical weights: "
            + ", ".join(mismatched[:5])
        )


def _assert_optimizer_finite(optimizer: Any) -> None:
    import torch

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not bool(torch.isfinite(parameter.detach()).all()):
                raise FloatingPointError("optimizer produced a non-finite policy parameter")
            state = optimizer.state.get(parameter, {})
            for name, value in state.items():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"optimizer state {name!r} contains non-finite values"
                    )


def _adapter_state_sha256(model: Any, adapter: str) -> str:
    import torch

    digest = hashlib.sha256()
    marker = f".{adapter}."
    matched = 0
    for name, tensor in sorted(model.state_dict().items()):
        if marker not in name:
            continue
        matched += 1
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    if matched == 0:
        raise RuntimeError(f"adapter {adapter!r} has no state tensors")
    return digest.hexdigest()


class TrainableGRPOPolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_seq_length: int,
        preserve_prompt_prefix_tokens: int,
        compute_dtype: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        self.preserve_prompt_prefix_tokens = preserve_prompt_prefix_tokens
        self.compute_dtype = compute_dtype
        self.traces: List[ActionTrace] = []
        self._episode_seed = 0
        self._decision_index = 0
        self._renderer = PromptOnlyPolicy(lambda _system, _user: "")

    def begin_episode(self, seed: int) -> None:
        self.traces = []
        self._episode_seed = seed
        self._decision_index = 0

    def decide(self, observation: Mapping[str, Any]) -> PolicyDecision:
        user_prompt = self._renderer._render_observation(observation)
        trace, raw_output = self._sample_action(
            _SYSTEM_PROMPT,
            user_prompt,
            seed=self._episode_seed + self._decision_index,
        )
        self._decision_index += 1
        self.traces.append(trace)
        action: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        try:
            action = AgentAction.from_raw(json.loads(raw_output)).to_dict()
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            parse_error = str(exc)
        return PolicyDecision(
            raw_output=raw_output,
            action=action,
            parse_error=parse_error,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            policy_version=POLICY_VERSION,
        )

    def _sample_action(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        seed: int,
    ) -> Tuple[ActionTrace, str]:
        import torch

        prompt_ids, truncated = self._encode_prompt(system_prompt, user_prompt)
        device = self.model.get_input_embeddings().weight.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        _activate_adapter(self.model, "policy")
        _assert_adapter_training_state(self.model, active_adapter="policy")
        _assert_dropout_disabled(self.model)
        self.model.eval()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), self._autocast():
            completion = self._sample_raw_policy_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        if not completion:
            fallback = self.tokenizer.eos_token_id
            if fallback is None:
                raise RuntimeError("model generated no tokens and tokenizer has no eos token")
            completion = [int(fallback)]
        with torch.no_grad():
            old = self.completion_log_probs(prompt_ids, completion, adapter="policy")
            reference = self.completion_log_probs(
                prompt_ids,
                completion,
                adapter="reference",
            )
        _activate_adapter(self.model, "policy")
        raw_output = self.tokenizer.decode(
            completion,
            skip_special_tokens=True,
        ).strip()
        return ActionTrace(
            prompt_ids=tuple(int(value) for value in prompt_ids),
            completion_ids=tuple(int(value) for value in completion),
            old_log_probs=tuple(float(value) for value in old.detach().cpu()),
            reference_log_probs=tuple(float(value) for value in reference.detach().cpu()),
            prompt_truncated=truncated,
        ), raw_output

    def _sample_raw_policy_tokens(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
    ) -> List[int]:
        """Sample directly from the policy softmax, without generation warpers."""

        import torch

        completion: List[int] = []
        past_key_values = None
        next_input_ids = input_ids
        eos_token_id = self.tokenizer.eos_token_id
        eos_ids = (
            {int(value) for value in eos_token_id}
            if isinstance(eos_token_id, (list, tuple, set))
            else ({int(eos_token_id)} if eos_token_id is not None else set())
        )
        for _ in range(self.max_new_tokens):
            outputs = self.model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :].float()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("policy produced non-finite sampling logits")
            probabilities = torch.softmax(logits, dim=-1)
            if not bool(torch.isfinite(probabilities).all()):
                raise FloatingPointError("policy produced non-finite sampling probabilities")
            sampled = torch.multinomial(probabilities, num_samples=1)
            token_id = int(sampled.item())
            completion.append(token_id)
            if token_id in eos_ids:
                break
            past_key_values = outputs.past_key_values
            next_input_ids = sampled
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
        return completion

    def _encode_prompt(self, system_prompt: str, user_prompt: str) -> Tuple[List[int], bool]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        template_options: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            rendered = self.tokenizer.apply_chat_template(messages, **template_options)
        except TypeError:
            template_options.pop("enable_thinking")
            rendered = self.tokenizer.apply_chat_template(messages, **template_options)
        prompt_ids = list(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        available = self.max_seq_length - self.max_new_tokens
        if len(prompt_ids) <= available:
            return prompt_ids, False
        prefix = min(self.preserve_prompt_prefix_tokens, available)
        suffix = available - prefix
        return prompt_ids[:prefix] + (prompt_ids[-suffix:] if suffix else []), True

    def completion_log_probs(
        self,
        prompt_ids: Sequence[int],
        completion_ids: Sequence[int],
        *,
        adapter: str,
        return_entropy: bool = False,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        if not prompt_ids or not completion_ids:
            raise ValueError("prompt_ids and completion_ids must not be empty")
        _activate_adapter(self.model, adapter)
        device = self.model.get_input_embeddings().weight.device
        full_ids = list(prompt_ids) + list(completion_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with self._autocast():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        start = len(prompt_ids) - 1
        completion_logits = logits[:, start : start + len(completion_ids), :]
        targets = input_ids[:, len(prompt_ids) :]
        log_distribution = functional.log_softmax(
            completion_logits.float(),
            dim=-1,
        )
        selected = log_distribution.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).reshape(-1)
        if not return_entropy:
            return selected
        with torch.no_grad():
            detached_log_distribution = log_distribution.detach()
            entropy = -(
                detached_log_distribution.exp() * detached_log_distribution
            ).sum(dim=-1).reshape(-1)
        return selected, entropy

    def _autocast(self) -> Any:
        import torch

        if not torch.cuda.is_available():
            return contextlib.nullcontext()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.compute_dtype]
        return torch.autocast(device_type="cuda", dtype=dtype)


def _load_policy(config: Mapping[str, Any], paths: Mapping[str, Path]) -> Tuple[Any, Any]:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config["model"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[model_config["torch_dtype"]]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured bfloat16, but this GPU does not support it")
    revision_options = (
        {"revision": model_config["revision"]}
        if model_config.get("revision")
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=model_config["trust_remote_code"],
        **revision_options,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config["quant_type"],
            bnb_4bit_use_double_quant=model_config["double_quant"],
            bnb_4bit_compute_dtype=dtype,
        ),
        torch_dtype=dtype,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        trust_remote_code=model_config["trust_remote_code"],
        **revision_options,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(
        base,
        str(paths["init_adapter"]),
        adapter_name="policy",
        is_trainable=True,
    )
    model.load_adapter(
        str(paths["init_adapter"]),
        adapter_name="reference",
        is_trainable=False,
    )
    model._grpo_disabled_dropout_modules = _disable_all_dropout(model)
    _assert_dropout_disabled(model)
    _assert_policy_matches_reference(model)
    _activate_adapter(model, "policy")
    _assert_adapter_training_state(model, active_adapter="policy")
    return model, tokenizer


def _save_policy_adapter(model: Any, tokenizer: Any, output_root: Path) -> Path:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"adapter output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _activate_adapter(model, "policy")
    _assert_adapter_training_state(model, active_adapter="policy")
    parameters = inspect.signature(model.save_pretrained).parameters
    if "selected_adapters" not in parameters:
        raise RuntimeError(
            "installed PEFT cannot save only the policy adapter; PEFT 0.20 is required"
        )
    model.save_pretrained(str(output_root), selected_adapters=["policy"])
    candidates = (output_root / "policy", output_root)
    adapter_path = next(
        (path for path in candidates if (path / "adapter_config.json").is_file()),
        None,
    )
    if adapter_path is None:
        raise RuntimeError(f"saved policy adapter is missing adapter_config.json: {output_root}")
    if not _adapter_artifacts_ready(adapter_path):
        raise RuntimeError(f"saved policy adapter is missing weights: {adapter_path}")
    if (output_root / "reference").exists():
        raise RuntimeError("frozen reference adapter must not be saved with the policy")
    tokenizer.save_pretrained(str(adapter_path))
    return adapter_path


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    resolved = output_dir.resolve()
    dangerous = {
        Path("/").resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
        PROJECT_ROOT.parent.resolve(),
    }
    if resolved in dangerous:
        raise RuntimeError(f"refusing unsafe GRPO output directory: {resolved}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"output directory is not empty: {output_dir}; use --overwrite-output-dir"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _select_tasks(
    tasks: Sequence[AgentRLTask],
    *,
    update: int,
    tasks_per_update: int,
) -> List[AgentRLTask]:
    if not tasks:
        raise ValueError("no training tasks available")
    start = ((update - 1) * tasks_per_update) % len(tasks)
    return [tasks[(start + offset) % len(tasks)] for offset in range(tasks_per_update)]


def run_training(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    overwrite_output_dir: bool,
) -> Dict[str, Any]:
    import torch

    run_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires a CUDA GPU")
    missing_dependencies = [name for name, present in dependency_status().items() if not present]
    if missing_dependencies:
        raise RuntimeError("missing GRPO dependencies: " + ", ".join(missing_dependencies))
    paths = _resolved_paths(config)
    run_provenance = {
        "code": _code_provenance(config_path),
        "base_model": _base_model_provenance(
            str(config["model"]["name_or_path"]),
            revision=config["model"].get("revision"),
        ),
        "runtime_versions": dependency_versions(),
    }
    for key in ("task_file", "bm25", "manifest"):
        if not paths[key].is_file():
            raise RuntimeError(f"missing training artifact: {paths[key]}")
    if not paths["init_adapter"].is_dir():
        raise RuntimeError(f"initial SFT adapter does not exist: {paths['init_adapter']}")
    output_dir = paths["output_dir"]
    _prepare_output_dir(output_dir, overwrite=overwrite_output_dir)

    training = config["training"]
    rollout_config = config["rollout"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tasks = load_tasks(paths["task_file"])
    source_task_count = len(tasks)
    random.Random(seed).shuffle(tasks)
    max_tasks = int(config["data"]["max_tasks"])
    if max_tasks:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise RuntimeError("training task file contains no tasks")

    store = BM25Store.load(str(paths["bm25"]))
    registry = build_minimal_registry(
        lambda query: store.search(
            query=query,
            top_k=int(config["data"]["retrieval_top_k"]),
        )
    )
    model, tokenizer = _load_policy(config, paths)
    reference_sha256_before = _adapter_state_sha256(model, "reference")
    finalizer_config = config["finalizer"]
    finalizer_backend = TransformersChatBackend(
        finalizer_config["name_or_path"],
        device_map="auto",
        dtype=finalizer_config["torch_dtype"],
        enable_thinking=False,
        seed=seed,
        revision=finalizer_config.get("revision"),
    )
    finalizer = FrozenAnswerFinalizer(
        finalizer_backend.make_complete_fn(
            max_new_tokens=int(finalizer_config["max_new_tokens"]),
        )
    )
    from app.agent_rl.env import PersonalRAGEnv

    env = PersonalRAGEnv(
        registry=registry,
        finalize_fn=finalizer,
        allowed_tools=("retrieve_docs",),
        max_steps=int(rollout_config["max_steps"]),
        reward_config=RewardConfig(**config["reward"]),
        seed=seed,
        env_version="personal-rag-hotpotqa-grpo-v1",
    )
    sampler = TrainableGRPOPolicy(
        model,
        tokenizer,
        max_new_tokens=int(rollout_config["max_new_tokens"]),
        temperature=float(rollout_config["temperature"]),
        top_p=float(rollout_config["top_p"]),
        top_k=int(rollout_config["top_k"]),
        max_seq_length=int(training["max_seq_length"]),
        preserve_prompt_prefix_tokens=int(training["preserve_prompt_prefix_tokens"]),
        compute_dtype=str(config["model"]["torch_dtype"]),
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("policy adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    reference_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if ".reference." in name
    }
    if optimizer_parameter_ids & reference_parameter_ids:
        raise RuntimeError("reference adapter parameters entered the optimizer")
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    trajectory_path = output_dir / "train_trajectories.jsonl"
    progress_path = output_dir / "training_progress.json"
    update_reports: List[Dict[str, Any]] = []
    total_episodes = 0
    total_decisions = 0
    total_truncated_prompts = 0

    with trajectory_path.open("w", encoding="utf-8") as trajectory_handle:
        for update in range(1, int(training["max_updates"]) + 1):
            selected_tasks = _select_tasks(
                tasks,
                update=update,
                tasks_per_update=int(rollout_config["tasks_per_update"]),
            )
            runtime_episodes: List[Dict[str, Any]] = []
            group_reports: List[Dict[str, Any]] = []
            update_rollouts: List[PolicyRollout] = []
            for task_index, task in enumerate(selected_tasks):
                group: List[Dict[str, Any]] = []
                env_seed = seed + update * 100_000 + task_index * 1_000
                for rollout_index in range(int(rollout_config["group_size"])):
                    sampling_seed = env_seed + rollout_index * 10
                    sampler.begin_episode(sampling_seed)
                    rollout = run_policy_episode(env, task, sampler, seed=env_seed)
                    entry = {
                        "task_id": task.task_id,
                        "group_index": task_index,
                        "rollout_index": rollout_index,
                        "env_seed": env_seed,
                        "sampling_seed": sampling_seed,
                        "rollout": rollout,
                        "traces": tuple(sampler.traces),
                        "reward": float(rollout.total_reward),
                        "reward_components": sum_reward_components(rollout.transitions),
                    }
                    group.append(entry)
                    runtime_episodes.append(entry)
                    update_rollouts.append(rollout)
                advantages, group_stats = group_relative_advantages(
                    [entry["reward"] for entry in group]
                )
                for entry, advantage in zip(group, advantages):
                    entry["advantage"] = advantage
                group_reports.append({
                    "task_id": task.task_id,
                    "rewards": [entry["reward"] for entry in group],
                    "advantages": advantages,
                    **group_stats,
                })

            optimization_episodes = [
                entry
                for entry in runtime_episodes
                if entry["traces"]
                and (
                    abs(float(entry["advantage"])) > 0.0
                    or float(training["beta"]) > 0.0
                )
            ]
            optimization_epoch_reports: List[Dict[str, Any]] = []
            for optimization_epoch in range(1, int(training["optimization_epochs"]) + 1):
                optimizer.zero_grad(set_to_none=True)
                optimization_rows: List[Dict[str, float]] = []
                backprop_loss = 0.0
                initial_logprob_max_error = 0.0
                policy_entropy_sum = 0.0
                policy_entropy_tokens = 0
                if optimization_episodes:
                    # Dropout probabilities were set to zero at load time.  We
                    # keep train mode for gradient checkpointing without
                    # changing the rollout distribution.
                    model.train()
                    _assert_dropout_disabled(model)
                    for entry in optimization_episodes:
                        weighted_losses = []
                        action_token_counts = []
                        for trace in entry["traces"]:
                            current, entropy = sampler.completion_log_probs(
                                trace.prompt_ids,
                                trace.completion_ids,
                                adapter="policy",
                                return_entropy=True,
                            )
                            old = torch.tensor(
                                trace.old_log_probs,
                                dtype=current.dtype,
                                device=current.device,
                            )
                            reference = torch.tensor(
                                trace.reference_log_probs,
                                dtype=current.dtype,
                                device=current.device,
                            )
                            if optimization_epoch == 1:
                                initial_logprob_max_error = max(
                                    initial_logprob_max_error,
                                    float((current.detach() - old).abs().max().cpu()),
                                )
                            action_loss, diagnostics = grpo_clipped_loss(
                                current,
                                old,
                                reference,
                                advantage=float(entry["advantage"]),
                                clip_epsilon=float(training["clip_epsilon"]),
                                beta=float(training["beta"]),
                            )
                            token_count = int(current.numel())
                            weighted_losses.append(action_loss * token_count)
                            action_token_counts.append(token_count)
                            policy_entropy_sum += float(entropy.sum().cpu())
                            policy_entropy_tokens += int(entropy.numel())
                            optimization_rows.append(diagnostics)
                        episode_loss = torch.stack(weighted_losses).sum() / sum(action_token_counts)
                        backprop_loss += (
                            float(episode_loss.detach().cpu())
                            / len(optimization_episodes)
                        )
                        (episode_loss / len(optimization_episodes)).backward()
                    if optimization_epoch == 1 and initial_logprob_max_error > 0.02:
                        raise RuntimeError(
                            "current policy does not match rollout old policy before update; "
                            f"max logprob error={initial_logprob_max_error:.6f}"
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        float(training["max_grad_norm"]),
                        error_if_nonfinite=True,
                    )
                    grad_norm_value = float(grad_norm.detach().cpu())
                    if not math.isfinite(backprop_loss) or not math.isfinite(grad_norm_value):
                        raise FloatingPointError(
                            f"non-finite optimization state: loss={backprop_loss}, "
                            f"grad_norm={grad_norm_value}"
                        )
                    optimizer.step()
                    _assert_optimizer_finite(optimizer)
                    _activate_adapter(model, "policy")
                optimization_means = mean_dict(optimization_rows)
                action_loss_unweighted_mean = optimization_means.pop("loss", 0.0)
                token_total = sum(row.get("tokens", 0.0) for row in optimization_rows)
                action_loss_token_weighted_mean = (
                    sum(
                        row.get("loss", 0.0) * row.get("tokens", 0.0)
                        for row in optimization_rows
                    )
                    / token_total
                    if token_total
                    else 0.0
                )
                optimization_epoch_reports.append({
                    "epoch": optimization_epoch,
                    "objective_loss": backprop_loss,
                    "action_loss_unweighted_mean": action_loss_unweighted_mean,
                    "action_loss_token_weighted_mean": action_loss_token_weighted_mean,
                    "grad_norm": grad_norm_value if optimization_episodes else 0.0,
                    "initial_logprob_max_error": initial_logprob_max_error,
                    "policy_entropy": (
                        policy_entropy_sum / policy_entropy_tokens
                        if policy_entropy_tokens
                        else 0.0
                    ),
                    **optimization_means,
                })

            reward_rows = [entry["reward_components"] for entry in runtime_episodes]
            evaluation = aggregate_policy_rollouts(update_rollouts)
            update_report = {
                "update": update,
                "task_ids": [task.task_id for task in selected_tasks],
                "episodes": len(runtime_episodes),
                "decisions": sum(len(entry["rollout"].decisions) for entry in runtime_episodes),
                "optimization_episodes": len(optimization_episodes),
                "zero_variance_groups": sum(bool(row["zero_variance"]) for row in group_reports),
                "reward_mean": sum(entry["reward"] for entry in runtime_episodes) / len(runtime_episodes),
                "reward_min": min(entry["reward"] for entry in runtime_episodes),
                "reward_max": max(entry["reward"] for entry in runtime_episodes),
                "reward_components_mean": mean_dict(reward_rows),
                "groups": group_reports,
                "optimization_epochs": optimization_epoch_reports,
                "stop_reason_counts": {
                    reason: sum(row.stop_reason == reason for row in update_rollouts)
                    for reason in sorted({row.stop_reason for row in update_rollouts})
                },
                "evaluation": evaluation,
            }
            update_reports.append(update_report)
            total_episodes += len(runtime_episodes)
            total_decisions += update_report["decisions"]
            total_truncated_prompts += sum(
                int(trace.prompt_truncated)
                for entry in runtime_episodes
                for trace in entry["traces"]
            )
            for entry in runtime_episodes:
                trajectory_handle.write(json.dumps({
                    **entry["rollout"].to_dict(),
                    "grpo": {
                        "update": update,
                        "group_index": entry["group_index"],
                        "rollout_index": entry["rollout_index"],
                        "env_seed": entry["env_seed"],
                        "sampling_seed": entry["sampling_seed"],
                        "episode_reward": entry["reward"],
                        "advantage": entry["advantage"],
                        "reward_components": entry["reward_components"],
                        "action_token_counts": [len(trace.completion_ids) for trace in entry["traces"]],
                        "prompt_truncated": [trace.prompt_truncated for trace in entry["traces"]],
                    },
                }, ensure_ascii=False, sort_keys=True) + "\n")
            trajectory_handle.flush()
            progress_path.write_text(
                json.dumps({
                    "schema_version": "agent-grpo-progress-v1",
                    "completed_updates": update,
                    "updates": update_reports,
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if update % int(training["save_steps"]) == 0:
                _save_policy_adapter(
                    model,
                    tokenizer,
                    output_dir / f"checkpoint-{update}",
                )

    reference_sha256_after = _adapter_state_sha256(model, "reference")
    if reference_sha256_after != reference_sha256_before:
        raise RuntimeError("frozen reference adapter changed during training")
    training_runtime = time.perf_counter() - training_started
    total_runtime = time.perf_counter() - run_started
    peak_memory_allocated = int(torch.cuda.max_memory_allocated())
    peak_memory_reserved = int(torch.cuda.max_memory_reserved())
    adapter_path = _save_policy_adapter(model, tokenizer, output_dir / "adapter")
    report = {
        "schema_version": TRAIN_REPORT_SCHEMA_VERSION,
        "algorithm": "multi-turn-grpo",
        "implementation": "custom-pytorch-clipped-grpo-v1",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "resolved_config": config,
        "resolved_config_sha256": _canonical_sha256(config),
        "provenance": run_provenance,
        "data": {
            "data_dir": str(paths["data_dir"]),
            "task_file": str(paths["task_file"]),
            "available_tasks": len(tasks),
            "input_sha256": {
                "tasks": _sha256(paths["task_file"]),
                "bm25": _sha256(paths["bm25"]),
                "manifest": _sha256(paths["manifest"]),
            },
        },
        "policy": {
            "base_model": config["model"]["name_or_path"],
            "initial_adapter": _adapter_provenance(str(paths["init_adapter"])),
            "reference_policy": "frozen reference adapter copied from initial_adapter",
            "reference_state_sha256_before": reference_sha256_before,
            "reference_state_sha256_after": reference_sha256_after,
            "reference_unchanged": True,
            "disabled_dropout_modules": int(
                getattr(model, "_grpo_disabled_dropout_modules", 0)
            ),
            "trained_adapter": _adapter_provenance(str(adapter_path)),
            "trained_adapter_path": str(adapter_path),
        },
        "finalizer": {
            "model": finalizer_config["name_or_path"],
            "resolved_commit": finalizer_backend.resolved_commit,
            "frozen": True,
        },
        "objective": {
            "group_scope": "same task and deterministic initial environment state",
            "loss_scope": "generated action tokens only",
            "clip_epsilon": training["clip_epsilon"],
            "beta": training["beta"],
            "optimization_epochs_per_rollout_batch": training["optimization_epochs"],
            "sampling_distribution": "raw policy softmax (temperature=1, top_p=1, top_k=0)",
        },
        "training": {
            "completed_updates": len(update_reports),
            "source_task_count": source_task_count,
            "training_pool_tasks": len(tasks),
            "episodes": total_episodes,
            "decisions": total_decisions,
            "truncated_prompts": total_truncated_prompts,
            "runtime_seconds": training_runtime,
            "total_runtime_seconds": total_runtime,
            "peak_cuda_memory_allocated_bytes": peak_memory_allocated,
            "peak_cuda_memory_reserved_bytes": peak_memory_reserved,
            "updates": update_reports,
        },
        "artifacts": {
            "trajectories": {
                "path": str(trajectory_path),
                "sha256": _sha256(trajectory_path),
            },
            "progress": {
                "path": str(progress_path),
                "sha256": _sha256(progress_path),
            },
        },
        "dependencies": dependency_status(),
    }
    report_path = output_dir / "train_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.model:
        config["model"]["name_or_path"] = args.model
        if Path(args.model).expanduser().is_dir():
            config["model"]["revision"] = None
    if args.init_adapter:
        config["model"]["init_adapter"] = args.init_adapter
    if args.finalizer_model:
        config["finalizer"]["name_or_path"] = args.finalizer_model
        if Path(args.finalizer_model).expanduser().is_dir():
            config["finalizer"]["revision"] = None
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    if args.max_updates is not None:
        config["training"]["max_updates"] = args.max_updates
    if args.max_tasks is not None:
        config["data"]["max_tasks"] = args.max_tasks
    if args.group_size is not None:
        config["rollout"]["group_size"] = args.group_size
    if args.tasks_per_update is not None:
        config["rollout"]["tasks_per_update"] = args.tasks_per_update
    if args.max_steps is not None:
        config["rollout"]["max_steps"] = args.max_steps
    if args.optimization_epochs is not None:
        config["training"]["optimization_epochs"] = args.optimization_epochs
    validate_config(config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-GPU multi-turn GRPO for the strict-JSON retrieval controller"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--init-adapter", default="")
    parser.add_argument("--finalizer-model", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--tasks-per-update", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--optimization-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default="")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"config does not exist: {config_path}")
    try:
        config = load_config(config_path)
        _apply_overrides(config, args)
        if args.dry_run:
            report = make_dry_run_report(config, config_path=config_path)
            if args.report:
                report_path = Path(args.report).expanduser()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        else:
            report = run_training(
                config,
                config_path=config_path,
                overwrite_output_dir=args.overwrite_output_dir,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
