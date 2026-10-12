from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = ROOT / "problems" / "triage"
MIMIC_SEED_STEMS = (
    "article_prompt_clinical",
    "article_prompt_general",
    "decision_tree_esi",
    "kb_full_context",
    "kb_tfidf",
    "resource_scoring",
    "self_consistency_vote",
    "structured_assessment",
)
METRICS = (
    "fitness",
    "exact_accuracy",
    "range_accuracy",
    "undertriage_rate",
    "overtriage_rate",
    "severe_undertriage_rate",
    "recall_1",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
