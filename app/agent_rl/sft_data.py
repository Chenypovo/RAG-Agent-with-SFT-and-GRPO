from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SFT_DATA_SCHEMA_VERSION = "agent-sft-example-v1"
SFT_REPORT_SCHEMA_VERSION = "agent-sft-build-report-v1"
OUTPUT_FORMATS = ("messages", "prompt_response")


@dataclass(frozen=True)
class SFTBuildConfig:
    """Filtering and serialization options for successful policy rollouts."""

    success_metric: str = "JointSuccess"
    min_success: float = 1.0
    output_format: str = "messages"
    require_clean_transitions: bool = True
    evidence_metric: Optional[str] = None
    min_evidence: float = 1.0
    pre_final_tool_repeat: int = 1

    def __post_init__(self) -> None:
        if not self.success_metric.strip():
            raise ValueError("success_metric must not be empty")
        if not math.isfinite(self.min_success):
            raise ValueError("min_success must be finite")
        if self.evidence_metric is not None and not self.evidence_metric.strip():
            raise ValueError("evidence_metric must not be empty when provided")
        if not math.isfinite(self.min_evidence):
            raise ValueError("min_evidence must be finite")
        if (
            isinstance(self.pre_final_tool_repeat, bool)
            or not isinstance(self.pre_final_tool_repeat, int)
            or self.pre_final_tool_repeat < 1
        ):
            raise ValueError("pre_final_tool_repeat must be a positive integer")
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {', '.join(OUTPUT_FORMATS)}"
            )


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _decision_is_parse_clean(decision: Any) -> bool:
    if not isinstance(decision, Mapping):
        return False
    return decision.get("parse_error") in (None, "")


def _decision_is_complete(decision: Any) -> bool:
    if not isinstance(decision, Mapping):
        return False
    return all(
        isinstance(decision.get(key), str) and bool(decision.get(key).strip())
        for key in ("system_prompt", "user_prompt", "raw_output")
    ) and isinstance(decision.get("action"), Mapping)


def _tool_event_for_transition(
    transition: Mapping[str, Any],
    action: Mapping[str, Any],
    step: int,
) -> Optional[Mapping[str, Any]]:
    observation_after = transition.get("observation_after")
    if not isinstance(observation_after, Mapping):
        return None
    history = observation_after.get("history")
    if not isinstance(history, Sequence) or isinstance(
        history, (str, bytes, bytearray)
    ):
        return None
    candidates = [
        event
        for event in history
        if isinstance(event, Mapping)
        and event.get("step_index") == step
        and event.get("tool") == action.get("tool")
        and event.get("args") == action.get("args")
    ]
    return candidates[-1] if candidates else None


def _unsafe_transition_reason(
    decisions: Sequence[Any],
    transitions: Any,
) -> Optional[str]:
    if not isinstance(transitions, Sequence) or isinstance(
        transitions, (str, bytes, bytearray)
    ):
        return "missing_transitions"
    if len(transitions) != len(decisions):
        return "transition_count_mismatch"

    for step, (decision, transition) in enumerate(zip(decisions, transitions)):
        if not isinstance(decision, Mapping) or not isinstance(transition, Mapping):
            return "invalid_transition"
        if transition.get("step_index") != step:
            return "transition_step_mismatch"

        decision_action = decision.get("action")
        transition_action = transition.get("action")
        if not isinstance(transition_action, Mapping) or transition_action != decision_action:
            return "transition_action_mismatch"

        reward_breakdown = transition.get("reward_breakdown")
        if not isinstance(reward_breakdown, Mapping):
            return "missing_reward_breakdown"
        invalid_reward = _as_float(reward_breakdown.get("invalid_action", 0.0))
        duplicate_reward = _as_float(reward_breakdown.get("duplicate_call", 0.0))
        if invalid_reward is None or duplicate_reward is None:
            return "invalid_reward_breakdown"
        if invalid_reward < 0.0:
            return "invalid_action_transition"
        if duplicate_reward < 0.0:
            return "duplicate_call_transition"

        if bool(transition_action.get("final_answer")):
            # Final-answer transitions legitimately create no new tool event.
            continue
        if not transition_action.get("tool"):
            return "invalid_transition_action"
        event = _tool_event_for_transition(transition, transition_action, step)
        if event is None:
            return "missing_tool_event"
        if event.get("ok") is not True or event.get("error") not in (None, ""):
            return "unsuccessful_tool_event"
    return None


def _safe_source_provenance(value: Any) -> Optional[Dict[str, Any]]:
    """Copy verifier metadata without carrying answer-bearing task fields.

    Verifier provenance is metadata, not model input.  Keys which could contain
    labels or reference answers are still excluded so downstream trainers cannot
    accidentally serialize them into a prompt.
    """

    if not isinstance(value, Mapping):
        return None
    excluded_fragments = ("gold", "answer", "label", "reference", "target")
    safe: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if any(fragment in key.lower() for fragment in excluded_fragments):
            continue
        if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            safe[key] = raw_value
        elif isinstance(raw_value, Mapping):
            nested = _safe_source_provenance(raw_value)
            if nested:
                safe[key] = nested
        elif isinstance(raw_value, Sequence) and not isinstance(
            raw_value, (str, bytes, bytearray)
        ):
            scalar_values = [
                item
                for item in raw_value
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
            if len(scalar_values) == len(raw_value):
                safe[key] = scalar_values
    return safe or None


def _verifier_provenance(
    rollout: Mapping[str, Any],
    verification: Mapping[str, Any],
    config: SFTBuildConfig,
    score: float,
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "source_field": "PolicyRollout.verification",
        "metric": config.success_metric,
        "score": score,
        "min_success": config.min_success,
        "verification_metrics": {
            str(key): value
            for key, value in verification.items()
            if _as_float(value) is not None
        },
    }
    if config.evidence_metric is not None:
        evidence_score = _as_float(verification.get(config.evidence_metric))
        provenance["evidence_gate"] = {
            "metric": config.evidence_metric,
            "score": evidence_score,
            "min_evidence": config.min_evidence,
        }
    source = _safe_source_provenance(rollout.get("verifier_provenance"))
    if source is not None:
        provenance["source_provenance"] = source
    for key in ("verifier_name", "verifier_version"):
        value = rollout.get(key)
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            provenance[key] = value
    return provenance


def _training_example(
    rollout: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    step: int,
    verification: Mapping[str, Any],
    score: float,
    config: SFTBuildConfig,
    augmentation_replica: int = 0,
) -> Dict[str, Any]:
    system_prompt = str(decision["system_prompt"])
    user_prompt = str(decision["user_prompt"])
    response = str(decision["raw_output"])
    policy_version = str(
        decision.get("policy_version") or rollout.get("policy_version") or "unknown"
    )
    example: Dict[str, Any] = {
        "schema_version": SFT_DATA_SCHEMA_VERSION,
        "task_id": str(rollout.get("task_id", "")),
        "step": step,
        "policy_version": policy_version,
        "parsed_action": dict(decision["action"]),
        "verifier_provenance": _verifier_provenance(
            rollout,
            verification,
            config,
            score,
        ),
    }
    if config.output_format == "messages":
        example["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
    else:
        example["prompt"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        example["response"] = response
    if augmentation_replica:
        example["augmentation"] = {
            "kind": "pre_final_tool_repeat",
            "replica": augmentation_replica,
            "total_repeats": config.pre_final_tool_repeat,
        }
    return example


def _is_final_action(decision: Mapping[str, Any]) -> bool:
    action = decision.get("action")
    return isinstance(action, Mapping) and bool(action.get("final_answer"))


def _is_pre_final_tool_decision(
    decisions: Sequence[Any],
    step: int,
) -> bool:
    """Return true for the continuation action immediately before a clean stop.

    Repeating only this row strengthens the supervision at the exact state where
    the SFT-741 controller stopped too early.  Earlier retrievals and final-answer
    rows retain their original frequency.
    """

    if step + 1 >= len(decisions):
        return False
    decision = decisions[step]
    next_decision = decisions[step + 1]
    if not isinstance(decision, Mapping) or not isinstance(next_decision, Mapping):
        return False
    action = decision.get("action")
    return (
        isinstance(action, Mapping)
        and bool(action.get("tool"))
        and _is_final_action(next_decision)
    )


def _action_category(decision: Mapping[str, Any]) -> str:
    if _is_final_action(decision):
        return "final_answer"
    action = decision.get("action")
    if isinstance(action, Mapping) and action.get("tool"):
        return "tool"
    return "other"


def build_sft_examples(
    rollouts: Iterable[Mapping[str, Any]],
    *,
    config: Optional[SFTBuildConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter successful episodes and return one SFT row per policy decision.

    Gold answers and task metadata are deliberately not read or copied.  Model
    inputs come only from the prompts already recorded on each PolicyDecision.
    """

    resolved_config = config or SFTBuildConfig()
    examples: List[Dict[str, Any]] = []
    filter_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    base_action_counts: Counter[str] = Counter()
    output_action_counts: Counter[str] = Counter()
    episodes_seen = 0
    decisions_seen = 0
    episodes_kept = 0

    for rollout in rollouts:
        episodes_seen += 1
        if not isinstance(rollout, Mapping):
            filter_counts["invalid_episode"] += 1
            continue

        decisions = rollout.get("decisions")
        if not isinstance(decisions, Sequence) or isinstance(
            decisions, (str, bytes, bytearray)
        ) or not decisions:
            filter_counts["missing_decisions"] += 1
            continue
        decisions_seen += len(decisions)

        if not all(_decision_is_parse_clean(decision) for decision in decisions):
            filter_counts["parse_error"] += 1
            continue
        if not all(_decision_is_complete(decision) for decision in decisions):
            filter_counts["incomplete_decision"] += 1
            continue
        if resolved_config.require_clean_transitions:
            transition_reason = _unsafe_transition_reason(
                decisions,
                rollout.get("transitions"),
            )
            if transition_reason is not None:
                filter_counts[transition_reason] += 1
                continue

        verification = rollout.get("verification")
        if not isinstance(verification, Mapping):
            filter_counts["missing_verification"] += 1
            continue
        score = _as_float(verification.get(resolved_config.success_metric))
        if score is None:
            filter_counts["missing_success_metric"] += 1
            continue
        if score < resolved_config.min_success:
            filter_counts["below_success_threshold"] += 1
            continue
        if resolved_config.evidence_metric is not None:
            evidence_score = _as_float(
                verification.get(resolved_config.evidence_metric)
            )
            if evidence_score is None:
                filter_counts["missing_evidence_metric"] += 1
                continue
            if evidence_score < resolved_config.min_evidence:
                filter_counts["below_evidence_threshold"] += 1
                continue

        episodes_kept += 1
        rollout_policy = str(rollout.get("policy_version") or "unknown")
        policy_counts[rollout_policy] += 1
        for step, decision in enumerate(decisions):
            category = _action_category(decision)
            base_action_counts[category] += 1
            repeats = (
                resolved_config.pre_final_tool_repeat
                if _is_pre_final_tool_decision(decisions, step)
                else 1
            )
            for replica in range(repeats):
                examples.append(
                    _training_example(
                        rollout,
                        decision,
                        step=step,
                        verification=verification,
                        score=score,
                        config=resolved_config,
                        augmentation_replica=replica,
                    )
                )
                output_action_counts[category] += 1

    report: Dict[str, Any] = {
        "schema_version": SFT_REPORT_SCHEMA_VERSION,
        "config": {
            "success_metric": resolved_config.success_metric,
            "min_success": resolved_config.min_success,
            "output_format": resolved_config.output_format,
            "require_clean_transitions": resolved_config.require_clean_transitions,
            "evidence_metric": resolved_config.evidence_metric,
            "min_evidence": resolved_config.min_evidence,
            "pre_final_tool_repeat": resolved_config.pre_final_tool_repeat,
        },
        "episodes_seen": episodes_seen,
        "episodes_kept": episodes_kept,
        "episodes_filtered": episodes_seen - episodes_kept,
        "decisions_seen": decisions_seen,
        "examples_written": len(examples),
        "base_examples": sum(base_action_counts.values()),
        "augmented_examples": len(examples) - sum(base_action_counts.values()),
        "base_action_counts": dict(sorted(base_action_counts.items())),
        "output_action_counts": dict(sorted(output_action_counts.items())),
        "filter_counts": dict(sorted(filter_counts.items())),
        "kept_policy_versions": dict(sorted(policy_counts.items())),
    }
    return examples, report


def load_rollout_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} must contain a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_sft_jsonl(path: Path, examples: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_sft_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    report_path: Optional[Path] = None,
    config: Optional[SFTBuildConfig] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path) if report_path is not None else None
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    resolved_report = report_path.resolve() if report_path is not None else None
    if resolved_input == resolved_output:
        raise ValueError("output must not overwrite the input trajectory file")
    if resolved_report in (resolved_input, resolved_output):
        raise ValueError("report must be different from input and output")
    existing = [path for path in (output_path, report_path) if path is not None and path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing artifact(s): "
            + ", ".join(str(path) for path in existing)
        )

    examples, report = build_sft_examples(
        load_rollout_jsonl(input_path),
        config=config,
    )
    written = write_sft_jsonl(output_path, examples)
    report_payload = {
        **report,
        "artifacts": {
            "input": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
            },
            "output": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            },
        },
    }
    if written != report_payload["examples_written"]:
        raise RuntimeError("written example count does not match the build report")
    report = {
        **report_payload,
        "artifacts": {
            **report_payload["artifacts"],
            "report": {
                "path": str(report_path) if report_path is not None else None,
                "sha256": _canonical_json_sha256(report_payload),
                "sha256_scope": "canonical JSON before the artifacts.report checksum field",
            },
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
