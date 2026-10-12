"""Guided‑Innovation ESI triage (Hybrid – refined thresholds & voting)
   - Retrieval: COSINE_THRESHOLD lowered to 0.30 for richer context.
   - Critical detection: relaxed vitals (HR>130, RR>30, SBP<90, SpO2<90) and expanded keywords.
   - Majority vote: tie‑break prefers more urgent (lower) level.
   - Resource vote: cnt≥2 → "3" (or "2" when high‑acuity tokens present) to avoid excessive downgrading.
   - parse_count returns None on failure; fallback relies on other votes.
   - Semaphore increased to 12 for parallelism.
"""

import asyncio
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from helper import ask_llm_json, load_esi_kb, measure_time

# ----------------------------------------------------------------------
# 1. Retrieval hyper‑parameters
# ----------------------------------------------------------------------
TOP_K = 6
COSINE_THRESHOLD = 0.30          # looser cut‑off → more useful KB snippets
SNIPPET_MAX_CHARS = 600

# ----------------------------------------------------------------------
# 2. Tokenisation
# ----------------------------------------------------------------------
TOKEN_PATTERN = r"\d+/\d+|\d+|\w+"

def tokenize(text: str) -> List[str]:
    return re.findall(TOKEN_PATTERN, text.lower())

# ----------------------------------------------------------------------
# 3. TF‑IDF cache (unchanged core)
# ----------------------------------------------------------------------
_KB_CACHE: Optional[List[Dict[str, Any]]] = None
_IDF_CACHE: Optional[Dict[str, float]] = None
_VECTORS_CACHE: Optional[List[Dict[str, float]]] = None
_NORMS_CACHE: Optional[List[float]] = None

def _ensure_kb_cache() -> None:
    global _KB_CACHE, _IDF_CACHE, _VECTORS_CACHE, _NORMS_CACHE
    if _KB_CACHE is None:
        _KB_CACHE = load_esi_kb()
        docs = [tokenize(f"{row['title']} {row['text']}") for row in _KB_CACHE]
        _IDF_CACHE = _build_idf(docs)
        _VECTORS_CACHE = [_vector(d, _IDF_CACHE) for d in docs]
        _NORMS_CACHE = [math.sqrt(sum(v * v for v in vec.values())) for vec in _VECTORS_CACHE]

def _build_idf(docs: List[List[str]]) -> Dict[str, float]:
    df = Counter()
    for tokens in docs:
        df.update(set(tokens))
    n = len(docs)
    return {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}

def _vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {tok: (cnt / total) * idf.get(tok, 1.0) for tok, cnt in tf.items()}

def _cosine(a: Dict[str, float], b: Dict[str, float], norm_b: float) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def retrieve(text: str) -> str:
    """Return up to TOP_K KB snippets whose cosine similarity ≥ COSINE_THRESHOLD."""
    _ensure_kb_cache()
    query_vec = _vector(tokenize(text), _IDF_CACHE)

    ranked = sorted(
        enumerate(_VECTORS_CACHE),
        key=lambda itm: _cosine(query_vec, itm[1], _NORMS_CACHE[itm[0]]),
        reverse=True,
    )

    parts = []
    for idx, _ in ranked:
        sim = _cosine(query_vec, _VECTORS_CACHE[idx], _NORMS_CACHE[idx])
        if sim < COSINE_THRESHOLD:
            continue
        snippet = _KB_CACHE[idx]["text"][:SNIPPET_MAX_CHARS]
        parts.append(f"### {_KB_CACHE[idx]['title']}\n{snippet}")
        if len(parts) >= TOP_K:
            break
    return "\n\n".join(parts)

# ----------------------------------------------------------------------
# 4. Patient‑text flattening
# ----------------------------------------------------------------------
def patient_text(p: Dict) -> str:
    return "\n".join(
        [
            f"HPI: {p.get('hpi', '')}",
            f"Info: {p.get('patient_info', '')}",
            f"Vitals: {p.get('initial_vitals', '')}",
            f"Chief: {p.get('chief_complaint', '')}",
        ]
    )

# ----------------------------------------------------------------------
# 5. Critical (ESI‑1) detection – relaxed but still safe
# ----------------------------------------------------------------------
CRITICAL_KEYWORDS = {
    "cardiac arrest", "unresponsive", "airway obstruction",
    "massive hemorrhage", "shock", "respiratory distress",
    "severe pain", "altered mental status", "hypotension",
    "low blood pressure", "hypoxia", "severe abdominal pain",
    "major trauma", "burns >30% body surface", "pulseless",
    "asystole", "ventricular fibrillation", "septic shock",
    "anaphylaxis", "tamponade"
}

def is_critical(p: Dict) -> bool:
    """Detect obvious life‑threatening presentations."""
    # Keyword scan first (fast)
    hpi = p.get("hpi", "").lower()
    if any(kw in hpi for kw in CRITICAL_KEYWORDS):
        return True

    # Vitals parsing – tolerant to dict or raw string
    vitals_raw = p.get("initial_vitals", {})
    if isinstance(vitals_raw, dict):
        sbp = int(vitals_raw.get("SBP", 0) or 0)
        hr  = int(vitals_raw.get("HR", 0) or 0)
        rr  = int(vitals_raw.get("RR", 0) or 0)
        spo2 = int(vitals_raw.get("SpO2", 100) or 100)
    else:
        txt = str(vitals_raw).lower()
        sbp = int(re.search(r"\b(\d{2,3})\s*/\s*\d{2,3}\b", txt).group(1)) if re.search(r"\b(\d{2,3})\s*/\s*\d{2,3}\b", txt) else 0
        hr  = int(re.search(r"(?:hr|heart\s*rate)[^\d]{0,10}(\d{2,3})", txt).group(1)) if re.search(r"(?:hr|heart\s*rate)[^\d]{0,10}(\d{2,3})", txt) else 0
        rr  = int(re.search(r"(?:rr|respiratory\s*rate)[^\d]{0,10}(\d{2,3})", txt).group(1)) if re.search(r"(?:rr|respiratory\s*rate)[^\d]{0,10}(\d{2,3})", txt) else 0
        spo2 = int(re.search(r"\b(8[0-9]|[0-7]\d)%\b", txt).group(1)) if re.search(r"\b(8[0-9]|[0-7]\d)%\b", txt) else 100

    # Relaxed but still urgent thresholds
    if sbp and sbp < 90:
        return True
    if hr and (hr > 130 or hr < 40):
        return True
    if rr and (rr > 30 or rr < 8):
        return True
    if spo2 and spo2 < 90:
        return True

    return False

# ----------------------------------------------------------------------
# 6. Normalisation of LLM output
# ----------------------------------------------------------------------
_WORD_TO_DIGIT = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

def normalize_acuity(val) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        acu = int(val)
    else:
        s = str(val).lower()
        s = re.sub(r"[^\w\s-]", " ", s)
        m = re.search(r"\b([1-5])\b", s)
        if m:
            acu = int(m.group(1))
        else:
            m = re.search(r"\b([1-5])\s*[-–]\s*([1-5])\b", s)
            if m:
                acu = int(m.group(1))  # lower (more urgent) side
            else:
                for w, d in _WORD_TO_DIGIT.items():
                    if re.search(rf"\b{w}\b", s):
                        acu = int(d)
                        break
                else:
                    return None
    return str(acu) if 1 <= acu <= 5 else None

# ----------------------------------------------------------------------
# 7. Yes/No helper (robust)
# ----------------------------------------------------------------------
YES_NO_SYSTEM = "You are an emergency triage nurse. Answer with JSON: {\"answer\": \"yes\" or \"no\"}"

async def yes(text: str, question: str) -> bool:
    try:
        res = await ask_llm_json(f"{text}\n\nQuestion: {question}", YES_NO_SYSTEM)
        ans = str(res.get("answer", "")).strip().lower()
        ans = re.sub(r"[^\w]", "", ans)
        return ans in {"yes", "y", "true", "1"}
    except Exception:
        return False

# ----------------------------------------------------------------------
# 8. Resource‑count handling (balanced)
# ----------------------------------------------------------------------
RESOURCE_SYSTEM = (
    "You are an emergency triage nurse. Estimate how many distinct ED resources "
    "(labs, imaging, ECG, IV meds, consult, procedure) the patient will need. "
    "Return JSON only: {\"resource_count\": <0‑6>}"
)

_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6,
}

def parse_count(v) -> Optional[int]:
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).lower()
    m = re.search(r"\b([0-6])\b", s)
    if m:
        return int(m.group(1))
    for w, n in _WORD_NUM.items():
        if re.search(rf"\b{w}\b", s):
            return n
    return None  # keep None to avoid forced low urgency

HIGH_ACUITY_TOKENS = {
    "severe pain", "altered mental status", "unresponsive", "cardiac arrest",
    "airway obstruction", "massive hemorrhage", "shock", "respiratory distress",
    "severe abdominal pain", "major trauma", "burns >30% body surface"
}

def resource_vote(txt: str, cnt: Optional[int]) -> Optional[str]:
    """Map resource count + high‑acuity context to an ESI level."""
    if cnt is None:
        return None
    low = txt.lower()
    high = any(tok in low for tok in HIGH_ACUITY_TOKENS)

    if cnt >= 3:
        return "2" if high else "3"
    if cnt == 2:
        return "2" if high else "3"
    if cnt == 1:
        return "3" if high else "4"
    return "5"

# ----------------------------------------------------------------------
# 9. Prompt templates
# ----------------------------------------------------------------------
LEVEL_EXAMPLES = """
Examples for level‑1:
- Massive chest trauma with airway compromise → {"acuity": 1}
- Unresponsive patient in cardiac arrest → {"acuity": 1}
Examples for level‑2:
- Severe abdominal pain with tachycardia → {"acuity": 2}
- Altered mental status, GCS 13 → {"acuity": 2}
Examples for level‑3:
- Stable vitals, needs labs and X‑ray → {"acuity": 3}
Examples for level‑4:
- Simple laceration, requires suturing only → {"acuity": 4}
Examples for level‑5:
- Routine wellness visit, no resources needed → {"acuity": 5}
"""

BASE_SYSTEM = """You are an experienced emergency‑department triage nurse.
Using the patient's history, demographics, and initial vitals, determine the
Emergency Severity Index (ESI) level (1‑5).

{kb}

Key criteria:
1 – Immediate life‑saving intervention required.
2 – High‑risk situation (altered mental status, severe pain, rapid deterioration).
3 – Stable vitals, likely needs ≥2 resources.
4 – Likely needs exactly 1 resource.
5 – No resources beyond focused exam.

Return ONLY a JSON object: {{"acuity": <int 1‑5>}}"""

SAFETY_SYSTEM = """You are a safety‑focused ED nurse. Avoid dangerous under‑triage while not over‑triaging unnecessarily.
{kb}
Return ONLY a JSON object: {{"acuity": <int 1‑5>}}"""

# ----------------------------------------------------------------------
# 10. Voting utilities (tie‑break prefers more urgent)
# ----------------------------------------------------------------------
def majority_vote(votes: List[str]) -> str:
    """Prefer level‑1 if present; otherwise most common; ties → lower (more urgent) level."""
    if "1" in votes:
        return "1"
    cnt = Counter(votes)
    max_c = max(cnt.values())
    tied = [int(v) for v, c in cnt.items() if c == max_c]
    return str(min(tied))   # lower number = more urgent

# ----------------------------------------------------------------------
# 11. Core classification
# ----------------------------------------------------------------------
SEMAPHORE = asyncio.Semaphore(12)

async def one_vote(txt: str, system: str) -> Optional[str]:
    try:
        res = await ask_llm_json(txt, system)
        return normalize_acuity(res.get("acuity"))
    except Exception:
        return None

async def classify(patient: dict) -> dict:
    # Fast‑track critical cases
    if is_critical(patient):
        return {"predicted_acuity": "1"}

    txt = patient_text(patient)
    kb = retrieve(txt)

    systems = [
        BASE_SYSTEM.format(kb=kb + "\n" + LEVEL_EXAMPLES),
        SAFETY_SYSTEM.format(kb=kb + "\n" + LEVEL_EXAMPLES),
        BASE_SYSTEM.format(kb=kb)  # no examples
    ]

    async with SEMAPHORE:
        raw_votes = await asyncio.gather(*(one_vote(txt, sys) for sys in systems))

        # Resource‑count vote
        try:
            rc_res = await ask_llm_json(txt, RESOURCE_SYSTEM)
            cnt = parse_count(rc_res.get("resource_count"))
        except Exception:
            cnt = None
        res_vote = resource_vote(txt, cnt)

    votes = [v for v in raw_votes if v is not None]
    if res_vote:
        votes.append(res_vote)

    # Need at least two independent votes; otherwise fall back to simple heuristics
    if len(votes) >= 2:
        pred = majority_vote(votes)
    else:
        # Simple fallback based on resource count alone
        if cnt == 0:
            pred = "5"
        elif cnt == 1:
            pred = "4"
        else:
            pred = "3"

    return {"predicted_acuity": pred}

# ----------------------------------------------------------------------
# 12. Entry points
# ----------------------------------------------------------------------
@measure_time()
async def entrypoint_async(patients: List[dict]) -> dict:
    results = await asyncio.gather(*(classify(p["patient"]) for p in patients))
    return {"results": results}

def entrypoint(patients: List[dict]) -> dict:
    return asyncio.run(entrypoint_async(patients))