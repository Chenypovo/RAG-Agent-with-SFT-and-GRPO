"""Build filtered decision-level SFT JSONL from PolicyRollout trajectories."""

import argparse
import json
import os
import sys
from pathlib import Path


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.sft_data import SFTBuildConfig, build_sft_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter successful PolicyRollout episodes into decision-level SFT JSONL"
    )
    parser.add_argument("--input", required=True, help="PolicyRollout JSONL input")
    parser.add_argument("--output", required=True, help="SFT JSONL output")
    parser.add_argument("--report", required=True, help="JSON build report output")
    parser.add_argument("--success-metric", default="JointSuccess")
    parser.add_argument("--min-success", type=float, default=1.0)
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("messages", "prompt_response"),
        default="messages",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output and report files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input does not exist: {input_path}")
    output_path = Path(args.output)
    report_path = Path(args.report)
    if output_path.resolve() == input_path.resolve():
        parser.error("output must not overwrite the input trajectory file")
    if report_path.resolve() in (input_path.resolve(), output_path.resolve()):
        parser.error("report must be different from input and output")

    try:
        config = SFTBuildConfig(
            success_metric=args.success_metric,
            min_success=args.min_success,
            output_format=args.output_format,
        )
        report = build_sft_jsonl(
            input_path,
            output_path,
            report_path=report_path,
            config=config,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
