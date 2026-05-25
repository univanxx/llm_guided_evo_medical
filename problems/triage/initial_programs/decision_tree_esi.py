"""ESI-style branching classifier driven by short LLM questions."""

import asyncio

from helper import ask_llm_json, measure_time


YES_NO_SYSTEM = (
    "You are an emergency triage nurse. Answer the clinical question from the "
    'case as JSON only: {"answer": "yes" or "no"}'
)
RESOURCE_SYSTEM = (
    "You are an emergency triage nurse. Estimate how many distinct ED resources "
    "this patient is likely to need: labs, imaging, ECG, IV medication, specialty "
    'consult, or procedure. Return JSON only: {"resource_count": 0}'
)


def patient_text(patient: dict) -> str:
    return (
        f"HPI: {patient['hpi']}\n"
        f"Patient info: {patient['patient_info']}\n"
        f"Initial vitals: {patient['initial_vitals']}\n"
        f"Chief complaint: {patient.get('chief_complaint', '')}"
    )


async def yes(text: str, question: str) -> bool:
    result = await ask_llm_json(f"{text}\n\nQuestion: {question}", YES_NO_SYSTEM)
    return str(result.get("answer", "no")).strip().lower().startswith("y")


def parse_count(value) -> int:
    try:
        return max(0, min(6, int(value)))
    except (TypeError, ValueError):
        return 2


async def classify(patient: dict) -> dict:
    text = patient_text(patient)
    if await yes(
        text,
        "Does the patient require immediate life-saving intervention now?",
    ):
        return {"predicted_acuity": "1"}

    if await yes(
        text,
        "Is this a high-risk situation, altered mental status, or severe pain/distress?",
    ):
        return {"predicted_acuity": "2"}

    resources = await ask_llm_json(text, RESOURCE_SYSTEM)
    count = parse_count(resources.get("resource_count"))
    if count >= 2:
        return {"predicted_acuity": "3"}
    if count == 1:
        return {"predicted_acuity": "4"}
    return {"predicted_acuity": "5"}


@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    return {
        "results": await asyncio.gather(
            *(classify(row["patient"]) for row in patients)
        )
    }


def entrypoint(patients: list[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))
