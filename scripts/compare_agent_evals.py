"""Compare prompt, SFT, and GRPO official evaluation outputs by task_id."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.artifacts import ensure_outputs_available  # noqa: E402
from app.agent_rl.comparison import compare_evaluation_runs  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official Agent RL evaluation reports and paired task trajectories"
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "REPORT_JSON", "TRAJECTORIES_JSONL"),
        required=True,
        help="repeat for prompt, SFT, GRPO, or any other comparable policy",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/agent_rl/eval_comparison.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if len(args.run) < 2:
        parser.error("provide at least two --run entries")
    output_path = Path(args.output)
    ensure_outputs_available((output_path,), overwrite=args.overwrite)
    report = compare_evaluation_runs(
        [(label, report_path, trajectory_path) for label, report_path, trajectory_path in args.run],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
