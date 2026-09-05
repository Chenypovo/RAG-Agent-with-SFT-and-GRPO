"""Paired bootstrap for retrieval attempts and behaviour with exact denominators."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent_rl.artifacts import sha256_file
from app.agent_rl.run_journal import atomic_json


METRICS = {
    "MeanRetrievalCalls": ("retrieval", "episode"),
    "PrematureStopRate": ("premature", "episode"),
    "BudgetExhaustionRate": ("budget", "episode"),
    "JSONInvalidRate": ("json_invalid", "decisions"),
    "UnsupportedToolActionRate": ("unsupported", "decisions"),
    "DuplicateCallRatePerDecision": ("duplicate", "decisions"),
    "Retrieval0Rate": ("retrieval0", "episode"),
    "Retrieval1Rate": ("retrieval1", "episode"),
    "Retrieval2Rate": ("retrieval2", "episode"),
    "Retrieval3Rate": ("retrieval3", "episode"),
    "Retrieval4PlusRate": ("retrieval4plus", "episode"),
}


def features(row):
    decisions = row["decisions"]
    transitions = row["transitions"]
    calls = sum((d.get("action") or {}).get("tool") == "retrieve_docs" for d in decisions)
    return {
        "episode": 1, "decisions": len(decisions), "retrieval": calls,
        "premature": int(any(t.get("reward_breakdown", {}).get("premature_final_answer", 0) < 0 for t in transitions)),
        "budget": int(row.get("stop_reason") == "budget"),
        "json_invalid": sum(bool(d.get("parse_error")) for d in decisions),
        "unsupported": sum((d.get("action") or {}).get("tool") not in (None, "retrieve_docs") for d in decisions),
        "duplicate": sum(t.get("reward_breakdown", {}).get("duplicate_call", 0) < 0 for t in transitions),
        **{"retrieval" + str(n): int(calls == n) for n in range(4)},
        "retrieval4plus": int(calls >= 4),
    }


def paired_summary(bn, bd, cn, cd, indices):
    bn, bd, cn, cd = (np.asarray(x, dtype=float) for x in (bn, bd, cn, cd))
    if not (np.all(bd > 0) and np.all(cd > 0)):
        raise ValueError("every task must have a positive metric denominator")
    baseline = float(bn.sum() / bd.sum())
    candidate = float(cn.sum() / cd.sum())
    deltas = cn[indices].sum(axis=1) / cd[indices].sum(axis=1) - bn[indices].sum(axis=1) / bd[indices].sum(axis=1)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"baseline": baseline, "candidate": candidate, "difference": candidate - baseline,
        "ci95_low": float(low), "ci95_high": float(high)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root) / "results"
    output = root / "paired_behaviour_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    runs = {}
    artifacts = {}
    for name in ("prompt", "e1_sft", "e2_grpo"):
        path = root / "evaluation" / (name + ".jsonl")
        data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(data) == 1000
        runs[name] = {row["task_id"]: features(row) for row in data}
        assert len(runs[name]) == 1000
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    ids = sorted(runs["prompt"])
    assert all(sorted(run) == ids for run in runs.values())
    pairs = []
    for pair_index, (baseline, candidate) in enumerate(itertools.combinations(runs, 2)):
        rng = random.Random(42 + pair_index)
        indices = np.asarray([rng.choices(range(1000), k=1000) for _ in range(2000)], dtype=np.int64)
        metrics = {}
        for name, (numerator, denominator) in METRICS.items():
            arrays = [[runs[label][task_id][field] for task_id in ids]
                for label, field in ((baseline, numerator), (baseline, denominator), (candidate, numerator), (candidate, denominator))]
            metrics[name] = paired_summary(*arrays, indices)
        pairs.append({"baseline": baseline, "candidate": candidate, "seed": 42 + pair_index, "metrics": metrics})
    report = {"bootstrap_samples": 2000, "seed": 42, "shared_tasks": 1000, "paired_task_ids_identical": True,
        "method": "paired percentile bootstrap over task IDs; the same sampled IDs feed both policies; ratio-of-sums for per-decision rates",
        "metric_definitions": METRICS, "artifacts": artifacts, "comparisons": pairs,
        "scope": "supplement to the original headline comparison; MeanRetrievalCalls excludes unsupported tool names; CIs spanning zero do not establish equivalence"}
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
