"""Evaluate the no-GPU one-shot BM25 baseline on prepared HotpotQA tasks."""

import argparse
import json
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.evaluation import evaluate_retrieval_baseline  # noqa: E402
from app.agent_rl.tasks import load_tasks  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prepared HotpotQA BM25 retrieval")
    parser.add_argument("--data-dir", default="data/agent_rl/hotpotqa_smoke")
    parser.add_argument("--partition", choices=["all", "train", "validation", "test"], default="all")
    parser.add_argument("--ks", default="4,8,20")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    task_name = "tasks.jsonl" if args.partition == "all" else f"tasks_{args.partition}.jsonl"
    tasks = load_tasks(data_dir / task_name)
    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]
    store = BM25Store.load(str(data_dir / "bm25.json"))
    report = evaluate_retrieval_baseline(
        tasks,
        lambda query, top_k: store.search(query=query, top_k=top_k),
        ks=ks,
    )
    report["retriever"] = "BM25Okapi"
    report["partition"] = args.partition
    report["data_dir"] = str(data_dir)

    output_path = Path(args.output) if args.output else data_dir / f"bm25_baseline_{args.partition}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
