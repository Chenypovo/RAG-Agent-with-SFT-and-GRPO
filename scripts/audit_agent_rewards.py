"""Run deterministic reward-hacking probes and write a JSON report."""

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

from app.agent_rl.artifacts import ensure_outputs_available, sha256_file  # noqa: E402
from app.agent_rl.reward_hacking import (  # noqa: E402
    reward_config_from_mapping,
    run_reward_hacking_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Agent RL rewards for deterministic reward-hacking regressions"
    )
    parser.add_argument(
        "--config",
        default="configs/agent_rl/qwen3_1.7b_grpo.json",
        help="GRPO JSON config whose reward section is audited",
    )
    parser.add_argument(
        "--output",
        default="results/agent_rl/reward_hacking_audit.json",
        help="machine-readable JSON report path",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    ensure_outputs_available((output_path,), overwrite=args.overwrite)
    config_path = Path(args.config)
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read reward config {config_path}: {exc}") from exc
    if not isinstance(raw_config, dict) or not isinstance(raw_config.get("reward"), dict):
        raise ValueError(f"config must contain a reward object: {config_path}")
    reward_config = reward_config_from_mapping(raw_config["reward"])
    report = run_reward_hacking_audit(reward_config)
    report["reward_config_source"] = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
