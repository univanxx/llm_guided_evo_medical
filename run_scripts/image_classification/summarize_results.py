"""Aggregate held-out paired evaluations without rerunning inference or selecting by test."""
import argparse
import csv
import json
from pathlib import Path
import statistics

METRICS = ("accuracy", "balanced_accuracy", "f1_pneumonia", "macro_f1", "sensitivity", "specificity", "invalid_rate")


def score(rows):
    """Count invalid responses as false negatives of their true class for recall and F1."""
    if not rows or any(row.get("true_id") not in (0, 1) for row in rows):
        raise ValueError("Prediction rows need nonempty binary ground truth")
    recalls, f1 = [], []
    for label in (0, 1):
        count = sum(row["true_id"] == label for row in rows)
        if not count:
            raise ValueError("Both classes are required for balanced metrics")
        tp = sum(row["true_id"] == label and row.get("pred_id") == label for row in rows)
        fp = sum(row["true_id"] != label and row.get("pred_id") == label for row in rows)
        fn = count - tp
        recalls.append(tp / count)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": sum(row.get("pred_id") == row["true_id"] for row in rows) / len(rows), "balanced_accuracy": statistics.mean(recalls), "f1_pneumonia": f1[1], "macro_f1": statistics.mean(f1), "sensitivity": recalls[1], "specificity": recalls[0], "invalid_rate": sum(row.get("pred_id") not in (0, 1) for row in rows) / len(rows)}


def read_json(path):
    """Read a saved UTF-8 artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run_dirs):
    """Average test repetitions within searches before calculating sample SD across searches."""
    records, identities, groups = [], {}, {}
    seen = set()
    for directory in run_dirs:
        config = read_json(directory / "run_config.json")
        if read_json(directory / "run_status.json").get("status") != "completed":
            raise ValueError("Only completed runs can be summarized")
        if config.get("evolution_source_runs"):
            raise ValueError("Use individual search runs for across-search statistics")
        identity = (config["model"], config["resolution"], config["seed"])
        if identity in seen:
            raise ValueError("Duplicate model/resolution/search seed")
        seen.add(identity)
        condition = identity[:2]
        from med_evolve.final_selection import experiment_contract

        contract = experiment_contract(config, directory)
        if condition in identities and identities[condition] != contract:
            raise ValueError("Incompatible search protocols in one condition")
        identities[condition] = contract
        summary = read_json(directory / "paired_evaluation/summary.json")
        for role in ("initial", "evolved"):
            saved = summary["roles"][role]["test"]["results"]
            if len(saved) != config["final_repeats"]:
                raise ValueError("Incomplete test repetitions")
            folder = directory / "paired_evaluation" / role / "final_evaluation"
            passes = {}
            for metric_path in folder.glob("*/metrics.json"):
                metric = read_json(metric_path)
                passes[metric["evaluation_id"]] = metric_path.parent
            repeated = []
            for result in saved:
                output = passes[result["evaluation_id"]]
                if read_json(output / "status.json").get("status") != "complete":
                    raise ValueError("Incomplete image evaluation")
                rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines() if line.strip()]
                if len(rows) != result["total"] or len(rows) != config["final_max_images"] or len({row["rel_path"] for row in rows}) != len(rows):
                    raise ValueError("Missing or duplicate image predictions")
                repeated.append(score(rows))
            record = {"model": condition[0], "resolution": condition[1], "seed": config["seed"], "role": role, "test_repeats": len(repeated), **{name: statistics.mean(item[name] for item in repeated) for name in METRICS}}
            records.append(record)
            groups.setdefault((*condition, role), []).append(record)
    conditions = []
    for (model, resolution, role), values in sorted(groups.items()):
        record = {"model": model, "resolution": resolution, "role": role, "searches": len(values)}
        for name in METRICS:
            measurements = [item[name] for item in values]
            record[name + "_mean"] = statistics.mean(measurements)
            record[name + "_sd"] = statistics.stdev(measurements) if len(values) > 1 else None
        conditions.append(record)
    return {"per_search": records, "conditions": conditions}


def main():
    """Write per-search and condition-level CSVs from explicitly supplied run directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in result.items():
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
