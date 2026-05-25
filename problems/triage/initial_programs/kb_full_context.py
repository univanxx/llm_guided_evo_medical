"""ESI classifier that injects the local reference KB as full context."""

import asyncio

from helper import ask_llm_json, load_esi_kb_text, measure_time


SYSTEM = """You are an ESI triage nurse.
Use the supplied reference material to assign ESI 1-5.
Return JSON only: {"acuity": 1}"""
KB_TEXT = load_esi_kb_text(max_chars=45_000)


def patient_text(patient: dict) -> str:
    return (
        f"HPI: {patient['hpi']}\n"
        f"Patient info: {patient['patient_info']}\n"
        f"Initial vitals: {patient['initial_vitals']}\n"
        f"Chief complaint: {patient.get('chief_complaint', '')}"
    )


def normalize(value) -> str:
    try:
        acuity = int(value)
    except (TypeError, ValueError):
        return "3"
    return str(acuity) if 1 <= acuity <= 5 else "3"


async def classify(patient: dict) -> dict:
    prompt = f"Reference material:\n{KB_TEXT}\n\nPatient case:\n{patient_text(patient)}"
    result = await ask_llm_json(prompt, SYSTEM)
    return {"predicted_acuity": normalize(result.get("acuity"))}


@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    return {
        "results": await asyncio.gather(
            *(classify(row["patient"]) for row in patients)
        )
    }


def entrypoint(patients: list[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))
