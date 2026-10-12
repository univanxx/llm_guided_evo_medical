"""Guided‑innovation ESI classifier with refined KB relevance, stricter vitals,
expanded life‑threatening keywords, and safer fallback handling."""
import asyncio
import re
from collections import Counter
from helper import ask_llm, measure_time, load_esi_kb_text

# ----------------------------------------------------------------------
# Prompt & KB handling
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an experienced emergency nurse responsible for triage decisions. "
    "Determine the Emergency Severity Index (ESI) level (1‑5) for the patient and "
    "respond ONLY with the level enclosed in a single <acuity> tag, e.g. "
    "<acuity>3</acuity>. "
    "If the patient is life‑threatening and needs immediate lifesaving intervention, "
    "output <acuity>1</acuity>."
)

KB_TEXT = load_esi_kb_text().splitlines()


def _top_keywords(text: str, top_n: int = 5) -> set[str]:
    """Very light TF‑IDF‑like extraction: most frequent words (>=3 chars)."""
    words = re.findall(r"\b\w{3,}\b", text.lower())
    freq = Counter(words)
    return {w for w, _ in freq.most_common(top_n)}


def _relevant_kb(patient: dict, max_lines: int = 15) -> str:
    """Select KB lines that share at least two top keywords with the patient."""
    patient_blob = f"{patient.get('hpi','')} {patient.get('chief_complaint','')}"
    keywords = _top_keywords(patient_blob)
    relevant = [
        line for line in KB_TEXT
        if sum(1 for kw in keywords if kw in line.lower()) >= 2
    ]
    if not relevant:
        relevant = KB_TEXT[:max_lines]
    return "\n".join(relevant[:max_lines])


def build_prompt(patient: dict) -> str:
    """Compose a concise prompt enriched with a context‑specific KB excerpt."""
    parts = [
        "You are a nurse with emergency and triage experience. Using the "
        "patient's history of present illness and his information, determine "
        "the triage level based on the Emergency Severity Index (ESI), ranging "
        "from 1 (most urgent) to 5 (least urgent).",
        "Relevant ESI guidance:",
        _relevant_kb(patient),
        f"History of present illness: {patient.get('hpi', '')}",
        f"Patient info: {patient.get('patient_info', '')}",
    ]
    if patient.get("initial_vitals"):
        parts.append(f"Initial vitals: {patient['initial_vitals']}")
    if patient.get("chief_complaint"):
        parts.append(f"Chief complaint: {patient['chief_complaint']}")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Regex extraction
# ----------------------------------------------------------------------
ACUITY_RE = re.compile(r"<\s*acuity\s*>\s*([1-5])\s*<\/\s*acuity\s*>", re.IGNORECASE)

def _extract_acuity(text: str) -> str | None:
    """Return a validated ESI digit or None."""
    if not text:
        return None
    m = ACUITY_RE.search(text)
    return m.group(1) if m else None


# ----------------------------------------------------------------------
# Rule‑based urgent detection
# ----------------------------------------------------------------------
LIFE_THREATENING_KEYWORDS = {
    "cardiac arrest", "respiratory failure", "severe bleeding", "unstable angina",
    "shock", "unconscious", "coma", "cardiogenic shock", "hypovolemic shock",
    "septic shock", "massive hemorrhage", "hemorrhagic shock", "loss of consciousness",
    "unresponsive", "cyanosis", "pulseless", "apnea", "life‑threatening",
    "pulmonary embolism", "myocardial infarction", "stroke", "severe sepsis",
    "anaphylaxis", "tension pneumothorax", "cardiopulmonary arrest",
    "pulseless electrical activity", "altered mental status", "hypotension",
    "severe dyspnea", "hypoxia", "bradycardia",
}


def contains_life_threatening(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in LIFE_THREATENING_KEYWORDS)


VITAL_PATTERNS = {
    "hr": re.compile(r"\b(?:HR|heart\s*rate)\s*[:=]?\s*([0-9]{2,3})\b", re.I),
    "sbp": re.compile(r"\b(?:SBP|systolic)\s*[:=]?\s*([0-9]{2,3})\b", re.I),
    "spo2": re.compile(r"\b(?:SpO2|spo2)\s*[:=]?\s*([0-9]{1,3})\b", re.I),
    "rr": re.compile(r"\b(?:RR|respiratory\s*rate)\s*[:=]?\s*([0-9]{2,3})\b", re.I),
    "gcs": re.compile(r"\bGCS\s*[:=]?\s*([0-9]{1,2})\b", re.I),
}


def vitals_flag(vitals_str: str) -> bool:
    """Detect physiologic danger signs with stricter thresholds."""
    for name, pat in VITAL_PATTERNS.items():
        m = pat.search(vitals_str)
        if not m:
            continue
        val = int(m.group(1))
        if name == "hr" and val >= 130:          # stricter than previous 120
            return True
        if name == "sbp" and val <= 90:
            return True
        if name == "spo2" and val <= 90:
            return True
        if name == "rr" and val >= 30:
            return True
        if name == "gcs" and val < 14:
            return True
    return False


def is_esi_one(patient: dict) -> bool:
    """Fast rule‑based detection for cases that must be ESI‑1."""
    if contains_life_threatening(patient.get("hpi", "")):
        return True
    if vitals_flag(str(patient.get("initial_vitals", ""))):
        return True
    return False


# ----------------------------------------------------------------------
# Secondary minimal prompt (truly minimal)
# ----------------------------------------------------------------------
MINIMAL_PROMPT = (
    "Provide ONLY the ESI level (1‑5) wrapped in an <acuity> tag, no extra text."
)


async def _secondary_prompt(patient: dict) -> str:
    """Retry LLM call with a minimal instruction to coax a clean tag."""
    return await ask_llm(build_prompt(patient) + "\n\n" + MINIMAL_PROMPT, SYSTEM_PROMPT)


# ----------------------------------------------------------------------
# Core classification
# ----------------------------------------------------------------------
SEMAPHORE = asyncio.Semaphore(40)


async def classify(patient: dict) -> dict:
    # 1️⃣ Rule‑based urgent check
    if is_esi_one(patient):
        return {"predicted_acuity": "1"}

    async with SEMAPHORE:
        raw = await ask_llm(build_prompt(patient), SYSTEM_PROMPT)
        acuity = _extract_acuity(raw)

        if acuity is None:
            # 2️⃣ Retry with a minimal prompt
            raw2 = await _secondary_prompt(patient)
            acuity = _extract_acuity(raw2)

        if acuity is None:
            # 3️⃣ Heuristic fallback based on vitals severity
            vitals = str(patient.get("initial_vitals", ""))
            if vitals_flag(vitals):
                acuity = "2"
            else:
                acuity = "4"

    return {"predicted_acuity": acuity}


# ----------------------------------------------------------------------
# Async entrypoint
# ----------------------------------------------------------------------
@measure_time()
async def entrypoint_async(patients: list[dict]) -> dict:
    results = await asyncio.gather(
        *(classify(row["patient"]) for row in patients)
    )
    return {"results": results}


def entrypoint(patients: list[dict]) -> dict:
    """Synchronous wrapper required by the evaluation harness."""
    return asyncio.run(entrypoint_async(patients))