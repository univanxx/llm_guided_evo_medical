"""Guided‑innovation ESI classifier: refined TF‑IDF retrieval + lightweight ESI‑1 safety."""

import asyncio
import math
import re
from collections import Counter

from helper import (
    ask_llm_json,
    load_esi_kb,
    measure_time,
)

# ---------- Retrieval configuration ----------
MAX_K = 3               # limit snippets to avoid dilution
SIM_THRESHOLD = 0.30    # stricter similarity to keep only relevant KB parts

KB_CACHE = None
IDF_CACHE = None
VECTORS = None


def tokenize(text: str) -> list[str]:
    """Keep alphanumerics (e.g., vitals like 120/80) and lowercase."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def patient_text(patient: dict) -> str:
    """Flatten patient record for prompting."""
    return (
        f"HPI: {patient.get('hpi', '')}\n"
        f"Patient info: {patient.get('patient_info', '')}\n"
        f"Initial vitals: {patient.get('initial_vitals', '')}\n"
        f"Chief complaint: {patient.get('chief_complaint', '')}"
    )


def load_kb():
    """Lazy load the full ESI knowledge base."""
    global KB_CACHE
    if KB_CACHE is None:
        KB_CACHE = load_esi_kb()
    return KB_CACHE


def build_idf(docs: list[list[str]]) -> dict[str, float]:
    """Add‑one smoothed IDF."""
    global IDF_CACHE
    if IDF_CACHE is not None:
        return IDF_CACHE
    doc_freq = Counter()
    for tokens in docs:
        doc_freq.update(set(tokens))
    n_docs = len(docs)
    IDF_CACHE = {
        token: math.log((n_docs + 1) / (freq + 1)) for token, freq in doc_freq.items()
    }
    return IDF_CACHE


def vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Sub‑linear TF (1+log(tf)) weighting."""
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {
        t: ((1 + math.log(tf[t])) / total) * idf.get(t, 0.0)
        for t in tf
    }


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(text: str) -> str:
    """Return up to MAX_K most similar KB entries above SIM_THRESHOLD."""
    global VECTORS
    kb = load_kb()
    docs = [tokenize(f"{row['title']} {row['text']}") for row in kb]
    idf = build_idf(docs)

    if VECTORS is None:
        VECTORS = [vector(toks, idf) for toks in docs]

    query_vec = vector(tokenize(text), idf)

    sims = [(idx, cosine(query_vec, vec)) for idx, vec in enumerate(VECTORS)]
    filtered = [
        (idx, sim) for idx, sim in sims if sim >= SIM_THRESHOLD
    ]
    # If nothing passes the threshold, fall back to the single best match
    if not filtered:
        filtered = sorted(sims, key=lambda x: x[1], reverse=True)[:1]

    # Sort descending by similarity and keep top MAX_K
    filtered.sort(key=lambda x: x[1], reverse=True)
    snippets = []
    for idx, _ in filtered[:MAX_K]:
        row = kb[idx]
        snippet = row["text"][:1200]  # keep within token budget
        snippets.append(f"### {row['title']}\n{snippet}")

    return "\n\n".join(snippets)


# ---------- Prompt templates ----------
SYSTEM = """You are an ESI triage nurse.
Use the reference material when helpful, but decide from the patient case.
Return JSON only: {"acuity": <int>}"""

FEATURE_SYSTEM = """You are an emergency triage nurse.
Extract the following boolean flags and an integer resource count from the case.
Return JSON only:
{
  "needs_lifesaving_intervention": false,
  "high_risk": false,
  "altered_mental_status": false,
  "severe_pain_or_distress": false,
  "danger_zone_vitals": false,
  "resource_count": 0
}"""


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def safe_int(value, default=0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def classify_from_features(f: dict) -> str:
    """Lightweight rule‑based mapping mirroring core ESI logic."""
    if truthy(f.get("needs_lifesaving_intervention")):
        return "1"
    if any(
        truthy(f.get(k))
        for k in (
            "high_risk",
            "altered_mental_status",
            "severe_pain_or_distress",
            "danger_zone_vitals",
        )
    ):
        return "2"
    rc = safe_int(f.get("resource_count"))
    if rc >= 3:
        return "3"
    if rc == 2:
        return "4"
    return "5"


def normalize(value) -> str | None:
    """Parse model output; return None on any failure."""
    try:
        acuity = int(value)
    except (TypeError, ValueError):
        return None
    return str(acuity) if 1 <= acuity <= 5 else None


async def classify(patient: dict) -> dict:
    txt = patient_text(patient)

    # 1) rule‑based safety check (ESI‑1 priority)
    features = await ask_llm_json(txt, FEATURE_SYSTEM)
    rule_pred = classify_from_features(features)
    if rule_pred == "1":
        return {"predicted_acuity": "1"}

    # 2) retrieve relevant KB snippets
    context = retrieve(txt)

    # 3) LLM prediction with retrieved context
    llm_res = await ask_llm_json(
        f"Reference material:\n{context}\n\nPatient case:\n{txt}",
        SYSTEM,
    )
    llm_pred = normalize(llm_res.get("acuity"))
    # If LLM fails, fall back to rule‑based result
    final = llm_pred if llm_pred is not None else rule_pred
    return {"predicted_acuity": final}


@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    return {
        "results": await asyncio.gather(
            *(classify(row["patient"]) for row in patients)
        )
    }


def entrypoint(patients: list[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))