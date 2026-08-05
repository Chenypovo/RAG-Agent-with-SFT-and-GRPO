from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from app.agent_rl.artifacts import sha256_file


COMPARISON_SCHEMA_VERSION = "agent-policy-eval-comparison-v1"

# Metrics copied directly from the official aggregate report.  These preserve
# the evaluator's exact denominators (notably per-decision invalid/duplicate
# rates) and are therefore the authoritative headline deltas.
CORE_REPORT_METRICS = (
    "AnswerEM",
    "AnswerF1",
    "JointSuccess",
    "CompleteSentenceEvidence",
    "CompleteDocumentEvidence",
    "MeanToolCalls",
    "MeanEnvironmentReturn",
    "DuplicateCallRate",
    "InvalidActionRate",
    "BudgetStopRate",
)

# These metrics have one value per task, so their differences can receive a
# paired bootstrap interval and an exact improved/regressed/tied task count.
PAIRED_METRICS = (
    "AnswerEM",
    "AnswerF1",
    "JointSuccess",
    "CompleteSentenceEvidence",
    "CompleteDocumentEvidence",
    "MeanToolCalls",
    "MeanEnvironmentReturn",
    "BudgetStopRate",
    "DuplicateCallCount",
    "InvalidActionCount",
)

LOWER_IS_BETTER = {
    "MeanToolCalls",
    "BudgetStopRate",
    "DuplicateCallCount",
    "InvalidActionCount",
}

_REPORT_CONSISTENCY_METRICS = (
    "AnswerEM",
    "AnswerF1",
    "JointSuccess",
    "CompleteSentenceEvidence",
    "CompleteDocumentEvidence",
    "MeanToolCalls",
    "MeanEnvironmentReturn",
    "BudgetStopRate",
)


def compare_evaluation_runs(
    run_specs: Sequence[tuple[str, str | Path, str | Path]],
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare two or more official policy evaluations on identical tasks."""

    if len(run_specs) < 2:
        raise ValueError("comparison requires at least two evaluation runs")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    labels = [str(spec[0]).strip() for spec in run_specs]
    if any(not label for label in labels):
        raise ValueError("run labels must not be empty")
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")

    loaded = [
        _load_run(label, Path(report_path), Path(trajectory_path))
        for label, report_path, trajectory_path in run_specs
    ]
    reference_ids = set(loaded[0]["tasks"])
    task_set_differences: Dict[str, Any] = {}
    for run in loaded[1:]:
        task_ids = set(run["tasks"])
        if task_ids != reference_ids:
            task_set_differences[run["label"]] = {
                "missing_from_run": sorted(reference_ids - task_ids),
                "extra_in_run": sorted(task_ids - reference_ids),
            }
    if task_set_differences:
        raise ValueError(
            "evaluation runs must contain identical task_id sets; differences: "
            + json.dumps(task_set_differences, ensure_ascii=False, sort_keys=True)
        )

    ordered_task_ids = sorted(reference_ids)
    comparisons = []
    for pair_index, (baseline, candidate) in enumerate(itertools.combinations(loaded, 2)):
        comparisons.append(
            _compare_pair(
                baseline,
                candidate,
                ordered_task_ids,
                bootstrap_samples=bootstrap_samples,
                seed=seed + pair_index,
            )
        )

    manifest_hashes = [run["data_manifest_sha256"] for run in loaded]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "config": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": 0.95,
            "bootstrap_method": "paired percentile bootstrap over task_id",
            "seed": seed,
        },
        "comparability": {
            "task_ids_identical": True,
            "n_shared_tasks": len(ordered_task_ids),
            "data_manifest_sha256_identical": (
                all(value == manifest_hashes[0] for value in manifest_hashes)
                if manifest_hashes and all(manifest_hashes)
                else None
            ),
        },
        "runs": [_public_run(run) for run in loaded],
        "comparisons": comparisons,
    }


def _load_run(label: str, report_path: Path, trajectory_path: Path) -> Dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read report for {label}: {report_path}: {exc}") from exc
    if not isinstance(report, Mapping):
        raise ValueError(f"report for {label} must be a JSON object")
    evaluation = report.get("evaluation")
    metrics = evaluation.get("metrics") if isinstance(evaluation, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ValueError(f"report for {label} must contain evaluation.metrics")
    report_metrics = {
        key: _finite_number(metrics[key], f"{label} report metric {key}")
        for key in CORE_REPORT_METRICS
        if key in metrics and metrics[key] is not None
    }

    tasks: Dict[str, Dict[str, float]] = {}
    try:
        handle = trajectory_path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot read trajectories for {label}: {trajectory_path}: {exc}"
        ) from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid trajectory JSON for {label} at line {line_number}: {exc}"
                ) from exc
            task_id = str(row.get("task_id", "")).strip() if isinstance(row, Mapping) else ""
            if not task_id:
                raise ValueError(
                    f"trajectory for {label} at line {line_number} has no task_id"
                )
            if task_id in tasks:
                raise ValueError(f"duplicate task_id for {label}: {task_id}")
            tasks[task_id] = _task_metrics(label, task_id, row)
    if not tasks:
        raise ValueError(f"trajectory file for {label} is empty")

    declared_n_tasks = evaluation.get("n_tasks") if isinstance(evaluation, Mapping) else None
    if declared_n_tasks is not None and int(declared_n_tasks) != len(tasks):
        raise ValueError(
            f"report/trajectory task count mismatch for {label}: "
            f"report={declared_n_tasks}, trajectories={len(tasks)}"
        )
    consistency: Dict[str, Any] = {}
    for metric_name in _REPORT_CONSISTENCY_METRICS:
        if metric_name not in report_metrics:
            continue
        trajectory_mean = _mean([row[metric_name] for row in tasks.values()])
        difference = trajectory_mean - report_metrics[metric_name]
        consistent = math.isclose(difference, 0.0, abs_tol=1e-9, rel_tol=1e-9)
        consistency[metric_name] = {
            "report": report_metrics[metric_name],
            "trajectory_mean": trajectory_mean,
            "difference": difference,
            "consistent": consistent,
        }
        if not consistent:
            raise ValueError(
                f"report/trajectory metric mismatch for {label} {metric_name}: "
                f"report={report_metrics[metric_name]}, trajectories={trajectory_mean}"
            )

    return {
        "label": label,
        "report_path": str(report_path),
        "trajectory_path": str(trajectory_path),
        "report_sha256": sha256_file(report_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "schema_version": report.get("schema_version"),
        "data_manifest_sha256": report.get("data_manifest_sha256"),
        "report_metrics": report_metrics,
        "aggregate_consistency": consistency,
        "tasks": tasks,
    }


def _task_metrics(label: str, task_id: str, row: Mapping[str, Any]) -> Dict[str, float]:
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        raise ValueError(f"trajectory {label}/{task_id} has no verification object")
    metrics = {
        name: _finite_number(verification.get(name), f"{label}/{task_id} {name}")
        for name in PAIRED_METRICS[:5]
    }
    transitions = row.get("transitions", [])
    if not isinstance(transitions, list):
        raise ValueError(f"trajectory {label}/{task_id} transitions must be a list")
    metrics.update({
        "MeanToolCalls": float(sum(
            1
            for transition in transitions
            if isinstance(transition, Mapping)
            and isinstance(transition.get("action"), Mapping)
            and bool(transition["action"].get("tool"))
        )),
        "MeanEnvironmentReturn": _finite_number(
            row.get("total_reward"), f"{label}/{task_id} total_reward"
        ),
        "BudgetStopRate": float(row.get("stop_reason") == "budget"),
        "DuplicateCallCount": float(sum(
            _negative_reward_component(transition, "duplicate_call")
            for transition in transitions
        )),
        "InvalidActionCount": float(sum(
            _negative_reward_component(transition, "invalid_action")
            for transition in transitions
        )),
    })
    return metrics


def _negative_reward_component(transition: Any, name: str) -> int:
    if not isinstance(transition, Mapping):
        return 0
    breakdown = transition.get("reward_breakdown")
    if not isinstance(breakdown, Mapping):
        return 0
    value = breakdown.get(name, 0.0)
    return int(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) < 0.0
    )


def _compare_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_ids: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    core_differences: Dict[str, Any] = {}
    for metric_name in CORE_REPORT_METRICS:
        if (
            metric_name not in baseline["report_metrics"]
            or metric_name not in candidate["report_metrics"]
        ):
            continue
        baseline_value = baseline["report_metrics"][metric_name]
        candidate_value = candidate["report_metrics"][metric_name]
        core_differences[metric_name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "difference": candidate_value - baseline_value,
        }

    paired_metrics: Dict[str, Any] = {}
    rng = random.Random(seed)
    for metric_name in PAIRED_METRICS:
        baseline_values = [baseline["tasks"][task_id][metric_name] for task_id in task_ids]
        candidate_values = [candidate["tasks"][task_id][metric_name] for task_id in task_ids]
        differences = [
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(baseline_values, candidate_values)
        ]
        low, high = _paired_bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            rng=rng,
        )
        higher_is_better = metric_name not in LOWER_IS_BETTER
        counts = _paired_counts(
            baseline_values,
            candidate_values,
            higher_is_better=higher_is_better,
        )
        paired_metrics[metric_name] = {
            "baseline": _mean(baseline_values),
            "candidate": _mean(candidate_values),
            "difference": _mean(differences),
            "higher_is_better": higher_is_better,
            "bootstrap_95_ci": {
                "low": low,
                "high": high,
                "samples": bootstrap_samples,
                "method": "paired percentile bootstrap over task_id",
            },
            "paired_counts": counts,
        }
    return {
        "baseline": baseline["label"],
        "candidate": candidate["label"],
        "n_paired_tasks": len(task_ids),
        "bootstrap_seed": seed,
        "core_metric_differences": core_differences,
        "paired_metrics": paired_metrics,
    }


def _paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    n = len(differences)
    estimates = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    estimates.sort()
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _paired_counts(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    higher_is_better: bool,
) -> Dict[str, int]:
    candidate_higher = baseline_higher = ties = 0
    for baseline_value, candidate_value in zip(baseline, candidate):
        if math.isclose(baseline_value, candidate_value, abs_tol=1e-12, rel_tol=1e-12):
            ties += 1
        elif candidate_value > baseline_value:
            candidate_higher += 1
        else:
            baseline_higher += 1
    return {
        "candidate_better": candidate_higher if higher_is_better else baseline_higher,
        "baseline_better": baseline_higher if higher_is_better else candidate_higher,
        "ties": ties,
        "candidate_higher": candidate_higher,
        "baseline_higher": baseline_higher,
    }


def _public_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: run[key]
        for key in (
            "label",
            "report_path",
            "trajectory_path",
            "report_sha256",
            "trajectory_sha256",
            "schema_version",
            "data_manifest_sha256",
            "report_metrics",
            "aggregate_consistency",
        )
    } | {"n_tasks": len(run["tasks"])}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
