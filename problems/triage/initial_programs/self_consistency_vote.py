"""Prompt-diverse ESI self-consistency vote."""

import asyncio
from collections import Counter

from helper import ask_llm_json, measure_time


SYSTEMS = [
    (
        "You are an experienced ESI triage nurse. Focus on correct ESI level. "
        'Return JSON only: {"acuity": 1}'
    ),
    (
        "You are a safety-focused ED nurse. Avoid dangerous undertriage, but "
        "do not assign unnecessary high acuity. "
        'Return JSON only: {"acuity": 1}'
    ),
    (
        "You are an ESI specialist. First identify life-saving need, then "
        "high-risk features, then expected resource count. "
        'Return JSON only: {"acuity": 1}'
    ),
]


def patient_text(patient: dict) -> str:
    return (
        "Assign ESI 1-5.\n"
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


async def one_vote(text: str, system: str) -> str:
    result = await ask_llm_json(text, system)
    return normalize(result.get("acuity"))


def choose(votes: list[str]) -> str:
    counts = Counter(votes)
    best_count = max(counts.values())
    tied = sorted(int(label) for label, count in counts.items() if count == best_count)
    return str(min(tied))


async def classify(patient: dict) -> dict:
    text = patient_text(patient)
    votes = await asyncio.gather(*(one_vote(text, system) for system in SYSTEMS))
    return {"predicted_acuity": choose(votes)}


@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    return {
        "results": await asyncio.gather(
            *(classify(row["patient"]) for row in patients)
        )
    }


def entrypoint(patients: list[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))
