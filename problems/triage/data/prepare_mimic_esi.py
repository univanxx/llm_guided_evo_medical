#!/usr/bin/env python3
"""Build an article-protocol ESI dataset from raw MIMIC source files.

This script mirrors the source-side preprocessing described in the paper:
- join ED triage, ED stays, first diagnosis, demographics, and discharge notes;
- reproduce the paper code's source-side HPI, test, and diagnosis filtering;
- reserve the first 2200 cleaned rows as the article candidate pool used by the
  public paper code and expose the first 2000 of them as an article-like benchmark;
- use only rows after that 2200-row pool as the train pool for evolution.

The paper's public repository performs a later LLM-dependent postprocessing step
before selecting its final 2000 rows. Without those intermediate predictions or
their final CSV, source files alone can only reproduce an article-like benchmark,
not a guaranteed row-for-row copy of the published set.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = 20260515
DEFAULT_BENCHMARK_SIZE = 2000
DEFAULT_ARTICLE_CANDIDATE_SIZE = 2200
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[2] / "triage" / "data"

TRIAGE_CSV = "mimic-iv-ed-2.2/ed/triage.csv.gz"
EDSTAYS_CSV = "mimic-iv-ed-2.2/ed/edstays.csv.gz"
DIAGNOSIS_CSV = "mimic-iv-ed-2.2/ed/diagnosis.csv.gz"

COMPLAINT_PATTERNS = {
    "complaint:abdominal": (
        "abdominal",
        "abdomen",
        "belly",
        "epigastr",
        "flank",
        "nausea",
        "vomit",
        "diarrhea",
    ),
    "complaint:cardiac": (
        "chest pain",
        "substernal",
        "palpitation",
        "syncope",
        "heart failure",
        "myocardial",
    ),
    "complaint:respiratory": (
        "shortness of breath",
        "dyspnea",
        "wheez",
        "cough",
        "hypoxia",
        "respiratory",
    ),
    "complaint:neurologic": (
        "altered mental",
        "confusion",
        "unresponsive",
        "seizure",
        "stroke",
        "weakness",
        "numbness",
        "headache",
        "dizziness",
    ),
    "complaint:infection": (
        "fever",
        "sepsis",
        "cellulitis",
        "infection",
        "pneumonia",
        "abscess",
    ),
    "complaint:trauma": (
        "fall",
        "trauma",
        "fracture",
        "injury",
        "struck",
        "laceration",
    ),
    "complaint:urinary": (
        "dysuria",
        "urinary",
        "hematuria",
        "urine",
        "kidney",
        "renal",
    ),
    "complaint:skin": (
        "rash",
        "wound",
        "ulcer",
        "swelling",
        "erythema",
        "edema",
    ),
    "complaint:psychiatric": (
        "suicidal",
        "depression",
        "anxiety",
        "psychosis",
        "overdose",
    ),
}

ED_RE = re.compile(r"\bED", flags=re.IGNORECASE)


def normalize_spaces(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize_acuity(value: str | None) -> str | None:
    raw = normalize_spaces(value)
    if not raw:
        return None
    try:
        acuity = int(float(raw))
    except ValueError:
        return None
    return str(acuity) if 1 <= acuity <= 5 else None


def open_zip_csv(zip_path: Path, member_name: str):
    archive = zipfile.ZipFile(zip_path)
    raw = archive.open(member_name)
    text = gzip.open(raw, "rt", encoding="utf-8", newline="")
    return archive, raw, text


def load_triage_rows(zip_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    archive, raw, text = open_zip_csv(zip_path, TRIAGE_CSV)
    try:
        rows: dict[tuple[str, str], dict[str, str]] = {}
        for row in csv.DictReader(text):
            acuity = normalize_acuity(row.get("acuity"))
            stay_id = normalize_spaces(row.get("stay_id"))
            subject_id = normalize_spaces(row.get("subject_id"))
            if not stay_id or not subject_id or acuity is None:
                continue
            rows[(subject_id, stay_id)] = {
                "stay_id": stay_id,
                "subject_id": subject_id,
                "acuity": acuity,
                "temperature": normalize_spaces(row.get("temperature")),
                "heartrate": normalize_spaces(row.get("heartrate")),
                "resprate": normalize_spaces(row.get("resprate")),
                "o2sat": normalize_spaces(row.get("o2sat")),
                "sbp": normalize_spaces(row.get("sbp")),
                "dbp": normalize_spaces(row.get("dbp")),
                "pain": normalize_spaces(row.get("pain")),
                "chief_complaint": normalize_spaces(row.get("chiefcomplaint")),
            }
        return rows
    finally:
        text.close()
        raw.close()
        archive.close()


def load_primary_diagnoses(zip_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    archive, raw, text = open_zip_csv(zip_path, DIAGNOSIS_CSV)
    try:
        rows: dict[tuple[str, str], dict[str, str]] = {}
        for row in csv.DictReader(text):
            subject_id = normalize_spaces(row.get("subject_id"))
            stay_id = normalize_spaces(row.get("stay_id"))
            if (
                normalize_spaces(row.get("seq_num")) != "1"
                or not subject_id
                or not stay_id
                or not normalize_spaces(row.get("icd_code"))
            ):
                continue
            rows[(subject_id, stay_id)] = {
                "icd_code": normalize_spaces(row.get("icd_code")),
                "icd_title": normalize_spaces(row.get("icd_title")),
                "icd_version": normalize_spaces(row.get("icd_version")),
            }
        return rows
    finally:
        text.close()
        raw.close()
        archive.close()


def load_patients(path: Path) -> dict[str, dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return {
            normalize_spaces(row.get("subject_id")): {
                "anchor_age": normalize_spaces(row.get("anchor_age")),
                "anchor_year": normalize_spaces(row.get("anchor_year")),
            }
            for row in csv.DictReader(f)
            if normalize_spaces(row.get("subject_id"))
        }


def format_gender(value: str) -> str:
    value = normalize_spaces(value).upper()
    if value == "F":
        return "Female"
    if value == "M":
        return "Male"
    return value or "Unknown"


def format_float(value: str) -> str:
    raw = normalize_spaces(value)
    if not raw:
        return ""
    try:
        return f"{float(raw):.1f}"
    except ValueError:
        return raw


def build_patient_info(row: dict[str, str], patient: dict[str, str]) -> str:
    parts = [
        f"Gender: {format_gender(row.get('gender', ''))}",
        f"Race: {normalize_spaces(row.get('race')) or 'Unknown'}",
    ]
    if patient.get("anchor_age"):
        parts.append(f"Age: {patient['anchor_age']}")
    return ", ".join(parts)


def build_initial_vitals(row: dict[str, str]) -> str:
    fields = [
        ("Temperature", "temperature"),
        ("Heartrate", "heartrate"),
        ("resprate", "resprate"),
        ("o2sat", "o2sat"),
        ("sbp", "sbp"),
        ("dbp", "dbp"),
    ]
    return ", ".join(
        f"{label}: {value}"
        for label, key in fields
        if (value := format_float(row.get(key, "")))
    )


def load_edstays(zip_path: Path) -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    dict[str, tuple[str, str]],
]:
    archive, raw, text = open_zip_csv(zip_path, EDSTAYS_CSV)
    try:
        rows: dict[tuple[str, str, str], dict[str, str]] = {}
        first_stay_by_hadm: dict[str, tuple[str, str]] = {}
        for row in csv.DictReader(text):
            subject_id = normalize_spaces(row.get("subject_id"))
            hadm_id = normalize_spaces(row.get("hadm_id"))
            stay_id = normalize_spaces(row.get("stay_id"))
            if not subject_id or not hadm_id or not stay_id:
                continue
            rows[(subject_id, stay_id, hadm_id)] = row
            first_stay_by_hadm.setdefault(hadm_id, (subject_id, stay_id))
        return rows, first_stay_by_hadm
    finally:
        text.close()
        raw.close()
        archive.close()


def get_tests(text: str) -> str | None:
    lower_text = text.lower()
    try:
        body = lower_text.split("pertinent results:")[1].split("brief hospital course:")[0]
    except IndexError:
        return None
    if "discharge labs" in body:
        return body.split("discharge labs")[0]
    return body


def get_hpi_section(text: str) -> str:
    return text.split("History of Present Illness:")[-1].split("Physical Exam:")[0]


def extract_hpi(text: str) -> str:
    lower_text = text.lower()
    pos_past_med_hist = lower_text.find("past medical history:")
    pos_soc_hist = lower_text.find("social history:")
    pos_fam_hist = lower_text.find("family history:")
    if pos_past_med_hist != -1:
        return text[:pos_past_med_hist].strip()
    if pos_soc_hist != -1:
        return text[:pos_soc_hist].strip()
    if pos_soc_hist != -1:
        return text[:pos_fam_hist].strip()
    return text


def remove_ed_course(text: str) -> str:
    patterns = [
        r"in the ED, initial vital.*",
        r"in the ED initial vital.*",
        r"\bED Course.*",
        r"\bIn ED initial VS.*",
        r"in the ED, initial VS.*",
        r"\binitial VS.*",
        r"in the ED.*",
    ]
    for pattern in patterns:
        text = re.sub(re.compile(pattern, re.IGNORECASE | re.DOTALL), "", text)
    return text


def extract_diagnosis(text: str) -> str:
    diagnosis = text.split("Discharge Diagnosis:")[-1].split("Discharge Condition:")[0]
    diagnosis = re.sub(re.compile("discharge diagnosis:", re.IGNORECASE), "", diagnosis)
    diagnosis = re.sub(
        re.compile(r".*Facility:\n___", re.IGNORECASE | re.DOTALL),
        "",
        diagnosis,
    )
    diagnosis = re.sub(
        re.compile(r".*___ Diagnosis:", re.IGNORECASE | re.DOTALL),
        "",
        diagnosis,
    )
    diagnosis = re.sub(re.compile(r"PMH.*", re.IGNORECASE | re.DOTALL), "", diagnosis)
    return diagnosis


def diagnosis_is_usable(diagnosis: str) -> bool:
    lower = diagnosis.lower()
    if any(term in lower for term in ("deceased", "died", "history of present illness")):
        return False

    has_primary = "primary" in lower
    valid_primary = bool(re.search(r"^\s*\nprimary", diagnosis, flags=re.IGNORECASE))
    if has_primary and not valid_primary:
        return False

    has_secondary = "secondary" in lower
    valid_secondary = bool(re.search(r"\nsecondary", diagnosis, flags=re.IGNORECASE))
    if has_secondary and not valid_secondary:
        return False

    index = lower.find("secondary")
    primary_diagnosis = diagnosis[:index] if index != -1 else diagnosis
    primary_diagnosis = re.sub(
        re.compile(r"___ Condition:.*", re.IGNORECASE | re.DOTALL),
        "",
        primary_diagnosis,
    )
    single_newlines = re.findall(r"(?<!\n)\n(?!\n)", primary_diagnosis)
    return len(single_newlines) < 16


def build_records_from_discharge(
    discharge_path: Path,
    triage_rows: dict[tuple[str, str], dict[str, str]],
    edstays: dict[tuple[str, str, str], dict[str, str]],
    first_stay_by_hadm: dict[str, tuple[str, str]],
    diagnoses: dict[tuple[str, str], dict[str, str]],
    patients: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_subjects: set[str] = set()

    with gzip.open(discharge_path, "rt", encoding="utf-8", newline="") as f:
        for note in csv.DictReader(f):
            hadm_id = normalize_spaces(note.get("hadm_id"))
            subject_id = normalize_spaces(note.get("subject_id"))
            stay_ref = first_stay_by_hadm.get(hadm_id)
            if not hadm_id or stay_ref is None:
                continue

            mapped_subject_id, stay_id = stay_ref
            if mapped_subject_id != subject_id or subject_id in seen_subjects:
                continue

            triage = triage_rows.get((subject_id, stay_id))
            edstay = edstays.get((subject_id, stay_id, hadm_id))
            if triage is None or edstay is None:
                continue

            # The paper code drops duplicate subjects after the first inner joins,
            # before later patient/diagnosis filtering.
            seen_subjects.add(subject_id)

            patient = patients.get(subject_id)
            diagnosis_row = diagnoses.get((subject_id, stay_id))
            if patient is None or diagnosis_row is None:
                continue
            text = note.get("text", "")
            if "history of present illness" not in text.lower():
                continue

            tests = get_tests(text)
            if tests is None or len(tests) >= 3000:
                continue

            hpi = remove_ed_course(extract_hpi(get_hpi_section(text)))
            if not 50 < len(hpi) < 2000:
                continue
            if (
                ED_RE.search(hpi)
                or " er " in hpi.lower()
                or "emergency room" in hpi.lower()
                or "emergency department" in hpi.lower()
                or "impression" in hpi.lower()
            ):
                continue

            diagnosis = extract_diagnosis(text)
            if not diagnosis_is_usable(diagnosis):
                continue

            records.append(
                {
                    "stay_id": stay_id,
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "hpi": hpi,
                    "patient_info": build_patient_info(edstay, patient),
                    "initial_vitals": build_initial_vitals(triage),
                    "chief_complaint": triage["chief_complaint"],
                    "acuity": triage["acuity"],
                }
            )
    return records


def _parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _age_bin(age: float | None) -> str:
    if age is None:
        return "age:unknown"
    if age < 18:
        return "age:pediatric"
    if age < 40:
        return "age:young_adult"
    if age < 65:
        return "age:adult"
    if age < 80:
        return "age:older_adult"
    return "age:very_old"


def _length_bin(text: str) -> str:
    length = len(text)
    if length < 400:
        return "length:short"
    if length < 800:
        return "length:medium"
    if length < 1200:
        return "length:long"
    return "length:very_long"


def feature_tokens(record: dict[str, str]) -> set[str]:
    text = " ".join(
        [
            record["patient_info"],
            record["initial_vitals"],
            record["chief_complaint"],
            record["hpi"],
        ]
    )
    text_lower = text.lower()
    features = {
        f"acuity:{record['acuity']}",
        _length_bin(record["hpi"]),
    }

    if "female" in record["patient_info"].lower():
        features.add("gender:female")
    elif "male" in record["patient_info"].lower():
        features.add("gender:male")
    else:
        features.add("gender:unknown")

    age = _parse_float(r"Age:\s*([0-9.]+)", record["patient_info"])
    features.add(_age_bin(age))

    temperature = _parse_float(r"Temperature:\s*([0-9.]+)", record["initial_vitals"])
    heart_rate = _parse_float(r"Heartrate:\s*([0-9.]+)", record["initial_vitals"])
    resp_rate = _parse_float(r"resprate:\s*([0-9.]+)", record["initial_vitals"])
    o2sat = _parse_float(r"o2sat:\s*([0-9.]+)", record["initial_vitals"])
    sbp = _parse_float(r"sbp:\s*([0-9.]+)", record["initial_vitals"])

    if temperature is not None and temperature >= 100.4:
        features.add("vital:fever")
    if temperature is not None and temperature < 96.0:
        features.add("vital:hypothermia")
    if heart_rate is not None and heart_rate >= 120:
        features.add("vital:tachycardia")
    if heart_rate is not None and heart_rate < 50:
        features.add("vital:bradycardia")
    if resp_rate is not None and resp_rate >= 22:
        features.add("vital:tachypnea")
    if o2sat is not None and o2sat <= 93:
        features.add("vital:hypoxemia")
    if sbp is not None and sbp < 90:
        features.add("vital:hypotension")
    if sbp is not None and sbp >= 180:
        features.add("vital:severe_hypertension")

    matched_complaints = 0
    for feature, patterns in COMPLAINT_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            features.add(feature)
            matched_complaints += 1
    if matched_complaints == 0:
        features.add("complaint:other")

    return features


def _stable_jitter(record: dict[str, str], seed: int) -> float:
    rng = random.Random(f"{seed}:{record['stay_id']}")
    return rng.random() * 1e-6


def diverse_subset_for_label(
    records: list[dict[str, str]],
    *,
    target_count: int,
    seed: int,
) -> list[dict[str, str]]:
    if len(records) <= target_count:
        return list(records)

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    feature_counts: Counter[str] = Counter()
    feature_frequency: Counter[str] = Counter()
    features_by_id: dict[str, set[str]] = {}

    for record in records:
        features = feature_tokens(record)
        features_by_id[record["stay_id"]] = features
        feature_frequency.update(features)

    while len(selected) < target_count:
        best_record = None
        best_score = -math.inf

        for record in records:
            stay_id = record["stay_id"]
            if stay_id in selected_ids:
                continue

            score = 0.0
            for feature in features_by_id[stay_id]:
                rarity_weight = 1.0 / math.sqrt(feature_frequency[feature])
                coverage_weight = 1.0 / math.sqrt(1.0 + feature_counts[feature])
                score += rarity_weight * coverage_weight
            score += _stable_jitter(record, seed)

            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None:
            break

        selected.append(best_record)
        selected_ids.add(best_record["stay_id"])
        feature_counts.update(features_by_id[best_record["stay_id"]])

    return selected


def balanced_evolution_subset(
    records: list[dict[str, str]],
    *,
    seed: int,
) -> list[dict[str, str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_label[record["acuity"]].append(record)

    if not by_label:
        return []

    per_label = min(len(items) for items in by_label.values())
    rng = random.Random(seed)
    subset: list[dict[str, str]] = []
    for label in sorted(by_label):
        subset.extend(
            diverse_subset_for_label(
                by_label[label],
                target_count=per_label,
                seed=seed,
            )
        )
    rng.shuffle(subset)
    return subset


def to_jsonl_record(record: dict[str, str]) -> dict[str, str]:
    return {
        "case_id": record["stay_id"],
        "source": "mimic-iv-ed-2.2",
        "subject_id": record["subject_id"],
        "hadm_id": record["hadm_id"],
        "hpi": record["hpi"],
        "patient_info": record["patient_info"],
        "initial_vitals": record["initial_vitals"],
        "chief_complaint": record["chief_complaint"],
        "true_acuity": record["acuity"],
    }


def write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonl_record(record), ensure_ascii=False) + "\n")


def write_source_csv(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "hpi",
        "patient_info",
        "initial_vitals",
        "chief_complaint",
        "acuity",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def counts(records: Iterable[dict[str, str]]) -> Counter[str]:
    return Counter(record["acuity"] for record in records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing mimic-iv-ed-2.2.zip, mimic-iv-3.0/, and mimic-iv-note-2.2/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated CSV/JSONL files.",
    )
    parser.add_argument("--benchmark-size", type=int, default=DEFAULT_BENCHMARK_SIZE)
    parser.add_argument(
        "--article-candidate-size",
        type=int,
        default=DEFAULT_ARTICLE_CANDIDATE_SIZE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    ed_zip = source_dir / "mimic-iv-ed-2.2.zip"
    patients_csv = source_dir / "mimic-iv-3.0" / "patients.csv.gz"
    discharge_csv = source_dir / "mimic-iv-note-2.2" / "discharge.csv.gz"

    for required in (ed_zip, patients_csv, discharge_csv):
        if not required.exists():
            raise FileNotFoundError(f"Required source file not found: {required}")

    triage_rows = load_triage_rows(ed_zip)
    diagnoses = load_primary_diagnoses(ed_zip)
    patients = load_patients(patients_csv)
    edstays, first_stay_by_hadm = load_edstays(ed_zip)
    records = build_records_from_discharge(
        discharge_csv,
        triage_rows,
        edstays,
        first_stay_by_hadm,
        diagnoses,
        patients,
    )

    if len(records) <= args.article_candidate_size:
        raise ValueError(
            f"Need more than {args.article_candidate_size} cleaned rows, got {len(records)}"
        )

    article_candidates = records[: args.article_candidate_size]
    benchmark = article_candidates[: args.benchmark_size]
    train = records[args.article_candidate_size :]
    train_evolution = balanced_evolution_subset(train, seed=args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_source_csv(out_dir / "mimic_esi_source.csv", records)
    write_jsonl(out_dir / "mimic_esi_all.jsonl", records)
    write_jsonl(out_dir / "mimic_esi_article_candidate_2200.jsonl", article_candidates)
    write_jsonl(out_dir / "mimic_esi_article_like_test.jsonl", benchmark)
    write_jsonl(out_dir / "mimic_esi_train.jsonl", train)
    write_jsonl(out_dir / "mimic_esi_train_evolution.jsonl", train_evolution)

    print(f"usable rows: {len(records)} counts={dict(sorted(counts(records).items()))}")
    print(
        "mimic_esi_article_candidate_2200.jsonl: "
        f"n={len(article_candidates)} "
        f"counts={dict(sorted(counts(article_candidates).items()))}"
    )
    print(
        "mimic_esi_article_like_test.jsonl: "
        f"n={len(benchmark)} counts={dict(sorted(counts(benchmark).items()))}"
    )
    print(
        f"mimic_esi_train.jsonl: n={len(train)} "
        f"counts={dict(sorted(counts(train).items()))}"
    )
    print(
        "mimic_esi_train_evolution.jsonl: "
        f"n={len(train_evolution)} counts={dict(sorted(counts(train_evolution).items()))}"
    )


if __name__ == "__main__":
    main()
