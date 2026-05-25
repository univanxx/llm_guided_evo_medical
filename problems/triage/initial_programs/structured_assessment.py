"""Two-stage ESI classifier with structured extraction then decision."""

import asyncio

from helper import ask_llm_json, measure_time


ANALYZE_SYSTEM = """You are an emergency triage nurse.
Extract only findings supported by the case.
Return JSON with:
{
  "red_flags": ["..."],
  "abnormal_vitals": ["..."],
  "mental_status_concern": true,
  "severe_pain_or_distress": true,
  "likely_resources": ["labs", "imaging", "ekg", "iv_medication", "procedure", "none"],
  "resource_count": 0
}"""

CLASSIFY_SYSTEM = """You are a senior ESI triage nurse.
Choose ESI 1-5 from the structured assessment and original case.
1 = immediate life-saving intervention.
2 = high risk, altered mental status, or severe pain/distress.
3 = stable but likely needs two or more resources.
4 = likely needs one resource.
5 = likely needs no resources beyond exam.
Return JSON only: {"acuity": 1}"""


def patient_text(patient: dict) -> str:
    return (
        f"HPI: {patient['hpi']}\n"
        f"Patient info: {patient['patient_info']}\n"
        f"Initial vitals: {patient['initial_vitals']}\n"
        f"Chief complaint: {patient.get('chief_complaint', '')}"
    )


def normalize_acuity(value) -> str:
    try:
        acuity = int(value)
    except (TypeError, ValueError):
        return "3"
    return str(acuity) if 1 <= acuity <= 5 else "3"


async def classify(patient: dict) -> dict:
    text = patient_text(patient)
    assessment = await ask_llm_json(text, ANALYZE_SYSTEM)
    decision = await ask_llm_json(
        f"Original case:\n{text}\n\nStructured assessment:\n{assessment}",
        CLASSIFY_SYSTEM,
    )
    return {"predicted_acuity": normalize_acuity(decision.get("acuity"))}


@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    return {
        "results": await asyncio.gather(
            *(classify(row["patient"]) for row in patients)
        )
    }


def entrypoint(patients: list[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))
