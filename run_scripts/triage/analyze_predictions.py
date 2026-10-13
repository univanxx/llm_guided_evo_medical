#!/usr/bin/env python3
"""Per-case triage analysis: weighted kappa, confusion matrices, paired bootstrap.

Each input is a predictions JSONL written by ``evaluate_programs.py``.  The first
``--pred`` is the reference (baseline); paired bootstrap CIs are reported for
the difference of every other system against it.

Example:
  python analyze_predictions.py \
    --pred base=.../base/predictions/article_prompt_clinical_r1.jsonl \
    --pred run1=.../luna/predictions/luna_run1_r1.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random

LEVELS = [1, 2, 3, 4, 5]


def load(path: Path) -> dict[str, tuple[int, int | None]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        try:
            pred = int(r["predicted_acuity"])
            pred = pred if 1 <= pred <= 5 else None
        except (TypeError, ValueError):
            pred = None
        case_id = str(r["case_id"])
        if case_id in rows:
            raise ValueError(f"{path}: duplicate case_id {case_id!r}")
        true = int(r["true_acuity"])
        if true not in LEVELS:
            raise ValueError(f"{path}: invalid true_acuity {true!r} for {case_id}")
        rows[case_id] = (true, pred)
    return rows


def weighted_kappa(pairs: list[tuple[int, int | None]], power: int) -> float:
    # Cohen's weighted kappa is defined on the ordinal ESI labels. Invalid
    # outputs are excluded here and exposed separately through valid_rate;
    # all accuracy/error-rate metrics still count them as incorrect.
    pairs = [(t, p) for t, p in pairs if p is not None]
    if not pairs:
        return float("nan")
    k = len(LEVELS)
    obs = [[0.0] * k for _ in range(k)]
    for t, p in pairs:
        assert p is not None
        obs[t - 1][p - 1] += 1
    n = sum(map(sum, obs))
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (abs(i - j) / (k - 1)) ** power
            num += w * obs[i][j]
            den += w * row[i] * col[j] / n
    return 1 - num / den if den else 0.0


def metrics(pairs: list[tuple[int, int | None]]) -> dict[str, float]:
    n = len(pairs)
    if not n:
        raise ValueError("Cannot compute metrics for zero cases")
    exact = sum(t == p for t, p in pairs) / n
    under = sum(p is not None and p > t for t, p in pairs) / n
    over = sum(p is not None and p < t for t, p in pairs) / n
    severe = sum(p is not None and p - t >= 2 for t, p in pairs) / n
    rng = sum(p is not None and (p == t or (t > 1 and p == t - 1)) for t, p in pairs) / n
    recalls = {}
    for lvl in (1, 2, 3):
        cls = [(t, p) for t, p in pairs if t == lvl]
        recalls[lvl] = sum(t == p for t, p in cls) / len(cls) if cls else float("nan")
    hi = [(t, p) for t, p in pairs if t <= 2]
    hi_sens = sum(p is not None and p <= 2 for _, p in hi) / len(hi) if hi else float("nan")
    return {
        "valid_rate": sum(p is not None for _, p in pairs) / n,
        "exact": exact,
        "range": rng,
        "balanced_acc_1_3": sum(recalls.values()) / 3,
        "qwk": weighted_kappa(pairs, 2),
        "lwk": weighted_kappa(pairs, 1),
        "under": under,
        "over": over,
        "severe_under": severe,
        "recall_1": recalls[1],
        "high_acuity_sens": hi_sens,
    }


def confusion_matrix(pairs: list[tuple[int, int | None]]) -> dict[str, list[int]]:
    matrix = {}
    for t in LEVELS:
        row = [sum(1 for tt, pp in pairs if tt == t and pp == p) for p in LEVELS]
        row.append(sum(1 for tt, pp in pairs if tt == t and pp is None))
        if sum(row):
            matrix[str(t)] = row
    return matrix


def confusion(pairs: list[tuple[int, int | None]]) -> str:
    lines = ["true\\pred " + " ".join(f"{p:>5}" for p in LEVELS) + "  invalid"]
    for t, row in confusion_matrix(pairs).items():
        lines.append(
            f"{int(t):>9} "
            + " ".join(f"{c:>5}" for c in row[:-1])
            + f"  {row[-1]:>7}"
        )
    return "\n".join(lines)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, matching common statistical tools."""
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return float("nan")
    pos = (len(clean) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path.jsonl")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    systems = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        systems[name] = load(Path(path))
    ref_name = next(iter(systems))
    ref_ids = set(systems[ref_name])
    for name, rows in systems.items():
        ids_here = set(rows)
        if ids_here != ref_ids:
            missing = sorted(ref_ids - ids_here)[:5]
            extra = sorted(ids_here - ref_ids)[:5]
            raise SystemExit(
                f"{name}: case_id set differs from {ref_name}; "
                f"missing={len(ref_ids-ids_here)} {missing}, "
                f"extra={len(ids_here-ref_ids)} {extra}"
            )
        mismatched = [
            case_id
            for case_id in ref_ids
            if rows[case_id][0] != systems[ref_name][case_id][0]
        ]
        if mismatched:
            raise SystemExit(
                f"{name}: true labels differ from {ref_name} for "
                f"{len(mismatched)} cases, e.g. {mismatched[:5]}"
            )
    ids = sorted(ref_ids)
    if not ids:
        raise SystemExit("Prediction files contain no cases")
    aligned = {name: [s[i] for i in ids] for name, s in systems.items()}

    report: dict = {
        "n_cases": len(ids),
        "reference": ref_name,
        "confusion_columns": ["1", "2", "3", "4", "5", "invalid"],
        "systems": {},
        "deltas_vs_reference": {},
    }
    for name, pairs in aligned.items():
        report["systems"][name] = {
            "metrics": metrics(pairs),
            "confusion_matrix": confusion_matrix(pairs),
        }
        print(f"\n== {name} (n={len(pairs)})")
        for k, v in report["systems"][name]["metrics"].items():
            print(f"  {k:>18}: {v:.4f}")
        print(confusion(pairs))

    ref = aligned[ref_name]
    rng = random.Random(args.seed)
    for name, pairs in aligned.items():
        if name == ref_name:
            continue
        ref_metrics = report["systems"][ref_name]["metrics"]
        cur_metrics = report["systems"][name]["metrics"]
        point = {k: v - ref_metrics[k] for k, v in cur_metrics.items()}
        samples: dict[str, list[float]] = {k: [] for k in point}
        for _ in range(args.bootstrap):
            idx = [rng.randrange(len(ids)) for _ in ids]
            a = metrics([pairs[i] for i in idx])
            b = metrics([ref[i] for i in idx])
            for k in samples:
                samples[k].append(a[k] - b[k])
        report["deltas_vs_reference"][name] = {}
        print(f"\n== {name} - {ref_name}: delta [95% paired bootstrap CI]")
        for k, vals in samples.items():
            lo = percentile(vals, 0.025)
            hi = percentile(vals, 0.975)
            report["deltas_vs_reference"][name][k] = {"delta": point[k], "ci95": [lo, hi]}
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"  {k:>18}: {point[k]:+.4f} [{lo:+.4f}, {hi:+.4f}] {sig}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        csv_path = args.out.with_suffix(".metrics.csv")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            metric_names = list(next(iter(report["systems"].values()))["metrics"])
            writer = csv.DictWriter(f, fieldnames=["system", *metric_names])
            writer.writeheader()
            for name, payload in report["systems"].items():
                writer.writerow({"system": name, **payload["metrics"]})


if __name__ == "__main__":
    main()
