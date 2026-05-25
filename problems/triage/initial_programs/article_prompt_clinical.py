"""Article-style clinical-user ESI prompt baseline."""

import asyncio
import re

from helper import ask_llm, measure_time


SYSTEM_PROMPT = "You are a helpful assistant."

ACUITY_RE = re.compile(r"<acuity>\s*([1-5])\s*</acuity>", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\b([1-5])\b")
SEMAPHORE = asyncio.Semaphore(20)


def build_prompt(patient: dict) -> str:
    return (
        "You are a nurse with emergency and triage experience. Using the "
        "patient's history of present illness, his information and initial "
        "vitals, determine the triage level based on the Emergency Severity "
        "Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 "
        "(lowest acuity): 1: Assign if the patient requires immediate "
        "lifesaving intervention. 2: Assign if the patient is in a high-risk "
        "situation (e.g., confused, lethargic, disoriented, or experiencing "
        "severe pain/distress)  3: Assign if the patient requires two or more "
        "diagnostic or therapeutic interventions and their vital signs are "
        "within acceptable limits for non-urgent care. 4: Assign if the "
        "patient requires one diagnostic or therapeutic intervention (e.g., "
        "lab test, imaging, or EKG). 5: Assign if the patient does not require "
        "any diagnostic or therapeutic interventions beyond a physical exam "
        "(e.g., no labs, imaging, or wound care).\n"
        f"History of present illness: {patient['hpi']}, patient info: "
        f"{patient['patient_info']} and initial vitals: "
        f"{patient['initial_vitals']}. Respond with the level in an <acuity> "
        "tag."
    )


def parse_acuity(raw: str) -> str:
    match = ACUITY_RE.search(raw or "")
    if match:
        return match.group(1)
    fallback = FALLBACK_RE.search(raw or "")
    return fallback.group(1) if fallback else "3"


async def classify(patient: dict) -> dict:
    async with SEMAPHORE:
        raw = await ask_llm(build_prompt(patient), SYSTEM_PROMPT)
    return {"predicted_acuity": parse_acuity(raw)}


@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    results = await asyncio.gather(
        *(classify(row["patient"]) for row in patients)
    )
    return {"results": results}


def entrypoint(patients: list[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))
