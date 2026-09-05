"""Inspect an existing remote experiment; never launch or retry a stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    results = root / "results"
    status = json.loads((results / "status.json").read_text())
    stage = status.get("stage", {})
    name = stage.get("name", "")
    progress_path = None
    if name == "full_teacher1617":
        progress_path = results / "e1_teacher/rollouts.candidates/progress.json"
    elif name == "grpo20":
        progress_path = results / "e2_grpo/training_progress.json"
    elif name.startswith("eval_"):
        progress_path = results / "evaluation" / (name.removeprefix("eval_") + ".episodes/progress.json")
    progress = None
    if progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if name == "grpo20":
            progress = {"completed_updates": progress["completed_updates"], "total_updates": 20}
    prefix = " ".join(stage.get("command", [])[:2])
    processes = subprocess.run(["pgrep", "-f", "^" + re.escape(prefix) + "( |$)"],
        capture_output=True, text=True).stdout.split() if prefix else []
    log = results / "logs" / (name + ".log")
    row = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "stage": name or None, "status": stage.get("status", status.get("status")),
        "stage_elapsed_seconds": round(time.time() - stage["started_unix"]) if stage.get("started_unix") else None,
        "completed_stages": [{"name": item["name"], "seconds": item.get("seconds"), "status": item["status"]}
            for item in status.get("completed_stages", [])],
        "progress": progress,
        "progress_age_seconds": round(time.time() - progress_path.stat().st_mtime)
            if progress_path is not None and progress_path.exists() else None,
        "active_stage_processes": processes, "disk_free_bytes": shutil.disk_usage(root).free,
        "log_tail": log.read_text()[-800:] if log.exists() else None,
    }
    teacher = results / "e1_teacher/rollouts.candidates/progress.json"
    if teacher.exists():
        row["teacher_progress"] = json.loads(teacher.read_text())
    with (root / "audit/monitor_checks.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    latest = root / "audit/latest_monitor_check.json"
    temporary = latest.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(latest)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
