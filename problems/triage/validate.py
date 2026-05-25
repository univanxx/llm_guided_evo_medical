import sys
from collections import Counter
from typing import Any


MIN_EXACT_ACCURACY = 0.20


def _parse_acuity(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    raw = result.get("predicted_acuity")
    try:
        acuity = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return str(acuity) if 1 <= acuity <= 5 else None


def _is_range_match(true_acuity: str, predicted_acuity: str | None) -> bool:
    if predicted_acuity is None:
        return False
    true_level = int(true_acuity)
    predicted_level = int(predicted_acuity)
    return predicted_level == true_level or (
        true_level > 1 and predicted_level == true_level - 1
    )


def validate(patients: list[dict], data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results", [])
    execution_time = float(data.get("execution_time_sec", 999999.0))
    n_patients = len(patients)

    if not isinstance(results, list):
        raise ValueError(f"`results` must be a list, got {type(results).__name__}")
    if len(results) != n_patients:
        raise ValueError(
            f"`results` length must match patients length: "
            f"got {len(results)}, expected {n_patients}"
        )

    true_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    exact_correct = 0
    range_correct = 0
    undertriage = 0
    overtriage = 0
    severe_undertriage = 0

    for patient_row, result in zip(patients, results, strict=True):
        true_acuity = str(patient_row["patient"]["true_acuity"])
        predicted_acuity = _parse_acuity(result)
        true_counts[true_acuity] += 1

        if predicted_acuity is not None:
            predicted_counts[predicted_acuity] += 1

        if predicted_acuity == true_acuity:
            exact_correct += 1
            correct_counts[true_acuity] += 1
        if _is_range_match(true_acuity, predicted_acuity):
            range_correct += 1

        if predicted_acuity is None:
            continue
        true_level = int(true_acuity)
        predicted_level = int(predicted_acuity)
        if predicted_level > true_level:
            undertriage += 1
            if predicted_level - true_level >= 2:
                severe_undertriage += 1
        elif predicted_level < true_level:
            overtriage += 1

    exact_accuracy = exact_correct / n_patients if n_patients else 0.0
    range_accuracy = range_correct / n_patients if n_patients else 0.0
    undertriage_rate = undertriage / n_patients if n_patients else 0.0
    overtriage_rate = overtriage / n_patients if n_patients else 0.0
    severe_undertriage_rate = severe_undertriage / n_patients if n_patients else 0.0
    mean_time = execution_time / n_patients if n_patients else execution_time

    recall_by_level = {
        level: correct_counts[level] / true_counts[level]
        for level in sorted(true_counts)
    }
    recall_1 = recall_by_level.get("1", 1.0)
    fitness = 0.65 * exact_accuracy + 0.25 * range_accuracy + 0.10 * recall_1
    is_valid = int(exact_accuracy >= MIN_EXACT_ACCURACY)

    print("-" * 50, file=sys.stderr)
    print(f"[VALIDATE] Patients: {n_patients}", file=sys.stderr)
    print(f"[VALIDATE] Exact accuracy: {exact_accuracy:.3f}", file=sys.stderr)
    print(f"[VALIDATE] Range accuracy: {range_accuracy:.3f}", file=sys.stderr)
    print(f"[VALIDATE] Undertriage rate: {undertriage_rate:.3f}", file=sys.stderr)
    print(f"[VALIDATE] Overtriage rate: {overtriage_rate:.3f}", file=sys.stderr)
    print(
        f"[VALIDATE] Severe undertriage rate: {severe_undertriage_rate:.3f}",
        file=sys.stderr,
    )
    print(f"[VALIDATE] Recall ESI-1: {recall_1:.3f}", file=sys.stderr)
    print(f"[VALIDATE] Mean time per patient: {mean_time:.3f}s", file=sys.stderr)
    print(f"[VALIDATE] True counts: {dict(sorted(true_counts.items()))}", file=sys.stderr)
    print(
        f"[VALIDATE] Predicted counts: {dict(sorted(predicted_counts.items()))}",
        file=sys.stderr,
    )
    print(f"[VALIDATE] Fitness: {fitness:.3f}", file=sys.stderr)
    print(f"[VALIDATE] Valid: {'YES' if is_valid else 'NO'}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    return {
        "fitness": float(fitness),
        "exact_accuracy": float(exact_accuracy),
        "range_accuracy": float(range_accuracy),
        "undertriage_rate": float(undertriage_rate),
        "overtriage_rate": float(overtriage_rate),
        "severe_undertriage_rate": float(severe_undertriage_rate),
        "recall_1": float(recall_1),
        "is_valid": is_valid,
    }
