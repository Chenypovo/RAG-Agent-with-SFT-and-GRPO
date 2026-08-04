"""Prepare a small, verifier-ready HotpotQA environment without GPU dependencies.

Example:
    python3 scripts/prepare_hotpotqa_agent_rl.py \
        --limit 200 --output-dir data/agent_rl/hotpotqa_smoke
"""

import argparse
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.datasets.hotpotqa import (  # noqa: E402
    convert_hotpot_rows,
    download_hotpot_rows,
    write_prepared_hotpotqa,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA tasks, corpus and a BM25 smoke index")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--output-dir", default="data/agent_rl/hotpotqa_smoke")
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()

    rows = download_hotpot_rows(limit=args.limit, offset=args.offset, page_size=args.page_size)
    if len(rows) != args.limit:
        raise RuntimeError(f"requested {args.limit} rows but downloaded {len(rows)}")
    prepared = convert_hotpot_rows(rows, offset=args.offset)
    manifest = write_prepared_hotpotqa(
        prepared,
        args.output_dir,
        build_bm25=not args.skip_bm25,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
