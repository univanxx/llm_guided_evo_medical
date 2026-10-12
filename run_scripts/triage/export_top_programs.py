#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
from pathlib import Path

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)


async def export(args: argparse.Namespace) -> None:
    storage = RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=f"redis://{args.redis_host}:{args.redis_port}/{args.redis_db}",
            key_prefix=args.redis_prefix,
            read_only=True,
        )
    )
    try:
        programs = await storage.get_all()
    finally:
        await storage.close()

    eligible = [
        p
        for p in programs
        if args.metric in p.metrics
        and p.is_complete
        and not p.is_failed
        and (args.max_generation is None or p.generation <= args.max_generation)
    ]
    eligible.sort(
        key=lambda p: (float(p.metrics[args.metric]), p.generation, p.atomic_counter),
        reverse=True,
    )
    selected = []
    seen_code: set[str] = set()
    for program in eligible:
        digest = hashlib.sha256(program.code.encode()).hexdigest()
        if digest in seen_code:
            continue
        seen_code.add(digest)
        selected.append((program, digest))
        if len(selected) >= args.top_n:
            break
    if not selected:
        raise RuntimeError(
            f"No eligible programs found in db={args.redis_db}, prefix={args.redis_prefix}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.output_dir.glob("*.py"):
        old.unlink()
    rows = []
    for rank, (program, digest) in enumerate(selected, 1):
        metric = float(program.metrics[args.metric])
        path = args.output_dir / f"{rank:03d}_{metric:.4f}_{program.id[:8]}.py"
        path.write_text(program.code, encoding="utf-8")
        rows.append(
            {
                "rank": rank,
                "program_id": program.id,
                "path": str(path),
                "generation": program.generation,
                "code_sha256": digest,
                **{f"metric_{k}": v for k, v in sorted(program.metrics.items())},
            }
        )
    with (args.output_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] exported {len(rows)} unique programs to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, required=True)
    parser.add_argument("--redis-prefix", required=True)
    parser.add_argument("--metric", default="fitness")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--max-generation",
        type=int,
        help="Only export candidates created at or before this generation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    asyncio.run(export(parser.parse_args()))


if __name__ == "__main__":
    main()
