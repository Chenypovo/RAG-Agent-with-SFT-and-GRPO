"""Run the frozen, auditable Agent orchestration benchmark.

Scripted mode is a deterministic regression check, not an effectiveness claim:

    python scripts/eval_agent_frozen.py --mode scripted --agent both

Real mode uses the configured OpenAI-compatible LLM for routing/control and
answer synthesis while keeping documents, memory, and tools frozen:

    python scripts/eval_agent_frozen.py --mode real --model GLM-4.6 --agent both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.eval.agent_benchmark import detect_real_environment, run_benchmark  # noqa: E402


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_environment(environment: Dict[str, bool]) -> None:
    print("Real-model configuration (values are never printed):")
    for key in (
        "api_key_configured",
        "base_url_configured",
        "llm_model_configured",
        "ready_for_real_mode",
    ):
        print(f"  {key}: {environment[key]}")


def _print_report(report: Dict[str, Any]) -> None:
    print(
        f"\nBenchmark={report['benchmark_id']} mode={report['mode']} "
        f"dataset_sha256={report['dataset_sha256'][:12]}..."
    )
    if report["mode"] == "scripted":
        print("WARNING: scripted results are regression checks, not resume effectiveness numbers.")

    for agent_name, payload in report["agents"].items():
        print(f"\n[{agent_name}] main comparison (shared_core only)")
        main = payload["main_comparison"]
        for key in (
            "n_tasks",
            "task_success",
            "answer_accuracy",
            "evidence_accuracy",
            "tool_call_precision",
            "tool_call_recall",
            "tool_multiset_exact_match",
            "refusal_accuracy",
            "avg_tool_calls",
            "avg_steps",
            "parse_failure_rate",
        ):
            print(f"  {key}: {_fmt(main[key])}")

        extended = payload["extended_capabilities"]
        print(
            "  extended capabilities: "
            f"n={extended['n_tasks']} coverage={_fmt(extended['capability_coverage'])} "
            f"success={_fmt(extended['task_success'])}"
        )
        robust = payload["robustness"]
        print(
            "  robustness: "
            f"n={robust['n_tasks']} success={_fmt(robust['task_success'])} "
            f"parse_failure_rate={_fmt(robust['parse_failure_rate'])}"
        )

    if "main_comparison_delta_loop_minus_baseline" in report:
        delta = report["main_comparison_delta_loop_minus_baseline"]
        print("\n[loop - baseline] shared_core delta")
        for key, value in delta.items():
            print(f"  {key}: {value:+.3f}")

    if report["reserved_tasks"]:
        print(f"\nReserved tasks excluded from metrics: {len(report['reserved_tasks'])}")
        for task in report["reserved_tasks"]:
            print(f"  {task['task_id']}: {task['skip_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen baseline-vs-ToolAgent benchmark with strict, auditable scoring"
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/agent_benchmark_v1.json",
        help="Frozen benchmark JSON",
    )
    parser.add_argument("--mode", choices=["scripted", "real"], default="scripted")
    parser.add_argument("--agent", choices=["both", "baseline", "loop"], default="both")
    parser.add_argument(
        "--model",
        default="",
        help="Real-mode model override, for example GLM-4.6; defaults to LLM_MODEL",
    )
    parser.add_argument("--output", default="", help="Optional JSON report path")
    parser.add_argument(
        "--check-real-config",
        action="store_true",
        help="Only report whether required real-mode configuration exists; never print values",
    )
    args = parser.parse_args()

    if args.check_real_config:
        _print_environment(detect_real_environment())
        return

    agents = ("baseline", "loop") if args.agent == "both" else (args.agent,)
    report = run_benchmark(
        args.dataset,
        mode=args.mode,
        agents=agents,
        model=args.model,
    )
    _print_report(report)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved report: {path}")


if __name__ == "__main__":
    main()
