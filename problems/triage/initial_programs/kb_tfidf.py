"""ESI classifier with TF-IDF retrieval from the local reference KB."""

import asyncio
import math
import re
from collections import Counter

from helper import ask_llm_json, load_esi_kb, measure_time


TOP_K = 3
KB_CACHE = None
IDF_CACHE = None
VECTORS = None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def patient_text(patient: dict) -> str:
    return (
        f"HPI: {patient['hpi']}\n"
        f"Patient info: {patient['patient_info']}\n"
        f"Initial vitals: {patient['initial_vitals']}\n"
        f"Chief complaint: {patient.get('chief_complaint', '')}"
    )


def load_kb():
    global KB_CACHE
    if KB_CACHE is None:
        KB_CACHE = load_esi_kb()
    return KB_CACHE


def build_idf(docs: list[list[str]]) -> dict[str, float]:
    global IDF_CACHE
    if IDF_CACHE is not None:
        return IDF_CACHE
    counts = Counter()
    for tokens in docs:
        counts.update(set(tokens))
    n_docs = len(docs)
    IDF_CACHE = {
        token: math.log((n_docs + 1) / (freq + 1)) + 1
        for token, freq in counts.items()
    }
    return IDF_CACHE


def vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {token: (freq / total) * idf.get(token, 1.0) for token, freq in tf.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    common = set(left) & set(right)
    if not common:
        return 0.0
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve(text: str) -> str:
    global VECTORS
    kb = load_kb()
    docs = [tokenize(f"{row['title']} {row['text']}") for row in kb]
    idf = build_idf(docs)
    if VECTORS is None:
        VECTORS = [vector(tokens, idf) for tokens in docs]
    query = vector(tokenize(text), idf)
    ranked = sorted(
        enumerate(VECTORS),
        key=lambda item: cosine(query, item[1]),
        reverse=True,
    )
    parts = []
    for index, vec in ranked[:TOP_K]:
        if cosine(query, vec) <= 0:
            continue
        row = kb[index]
        parts.append(f"### {row['title']}\n{row['text'][:1800]}")
    return "\n\n".join(parts)


SYSTEM = """You are an ESI triage nurse.
Use the reference material when it is helpful, but decide from the patient case.
Return JSON only: {"acuity": 1}"""


def normalize(value) -> str:
    try:
        acuity = int(value)
    except (TypeError, ValueError):
        return "3"
    return str(acuity) if 1 <= acuity <= 5 else "3"


async def classify(patient: dict) -> dict:
    text = patient_text(patient)
    context = retrieve(text)
    result = await ask_llm_json(
        f"Reference material:\n{context}\n\nPatient case:\n{text}",
        SYSTEM,
    )
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
