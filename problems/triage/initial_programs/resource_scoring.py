"""Hybrid ESI classifier: LLM extracts dimensions, rules choose the level."""

import asyncio

from helper import ask_llm_json, measure_time


SYSTEM = """You are an emergency triage nurse.
Score the case and return JSON only:
{
  "needs_lifesaving_intervention": true,
  "high_risk": true,
  "altered_mental_status": true,
  "severe_pain_or_distress": true,
  "resource_count": 0,
  "danger_zone_vitals": true
}"""


def patient_text(patient: dict) -> str:
    return (
        f"HPI: {patient['hpi']}\n"
        f"Patient info: {patient['patient_info']}\n"
        f"Initial vitals: {patient['initial_vitals']}\n"
        f"Chief complaint: {patient.get('chief_complaint', '')}"
    )


def truthy(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "yes", "1"}


def count(value) -> int:
    try:
        return max(0, min(6, int(value)))
    except (TypeError, ValueError):
        return 2


def classify_from_features(features: dict) -> str:
    if truthy(features.get("needs_lifesaving_intervention")):
        return "1"
    if any(
        truthy(features.get(key))
        for key in ("high_risk", "altered_mental_status", "severe_pain_or_distress")
    ):
        return "2"
    if count(features.get("resource_count")) >= 2:
        return "3"
    if count(features.get("resource_count")) == 1:
        return "4"
    return "5"


async def classify(patient: dict) -> dict:
    features = await ask_llm_json(patient_text(patient), SYSTEM)
    return {"predicted_acuity": classify_from_features(features)}


@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    return {
        "results": await asyncio.gather(
            *(classify(row["patient"]) for row in patients)
        )
    }


def entrypoint(patients: list[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))
