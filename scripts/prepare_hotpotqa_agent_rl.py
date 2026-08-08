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
    DATASET_CONFIG,
    DATASET_SPLIT,
    INVALID_ROW_POLICIES,
    PARTITION_MODES,
    convert_hotpot_rows,
    download_hotpot_rows,
    load_hotpot_parquet,
    write_prepared_hotpotqa,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA tasks, corpus and a BM25 smoke index")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--source-split", default=DATASET_SPLIT)
    parser.add_argument("--partition-mode", choices=PARTITION_MODES, default="gold_document_hash")
    parser.add_argument(
        "--invalid-row-policy",
        choices=INVALID_ROW_POLICIES,
        default="error",
        help="error (default) rejects invalid rows; skip records and omits them",
    )
    parser.add_argument("--input-parquet", default="")
    parser.add_argument("--output-dir", default="data/agent_rl/hotpotqa_smoke")
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()

    if args.input_parquet:
        rows = load_hotpot_parquet(args.input_parquet, limit=args.limit, offset=args.offset)
    else:
        rows = download_hotpot_rows(
            limit=args.limit,
            offset=args.offset,
            page_size=args.page_size,
            config=args.dataset_config,
            split=args.source_split,
        )
    if len(rows) != args.limit:
        raise RuntimeError(f"requested {args.limit} rows but downloaded {len(rows)}")
    prepared = convert_hotpot_rows(
        rows,
        config=args.dataset_config,
        split=args.source_split,
        offset=args.offset,
        partition_mode=args.partition_mode,
        invalid_row_policy=args.invalid_row_policy,
    )
    manifest = write_prepared_hotpotqa(
        prepared,
        args.output_dir,
        build_bm25=not args.skip_bm25,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
