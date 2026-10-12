"""Guided‑Innovation ESI classifier (refined):
- Precise red‑flag matching via word‑boundary regex
- Always query LLM for verification (no bypass for high‑resource decisions)
- Expanded TOP_K to 6 for richer KB context
- Lower unknown‑token TF‑IDF weight (0.005)
- Adjusted tachypnea threshold (>35) in rule fallback
"""

import asyncio
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional

from helper import (
    ask_llm_json,
    ask_llm,
    load_esi_kb,
    measure_time,
)

# ---------- Retrieval configuration ----------
TOP_K = 6                     # more snippets for better coverage
MAX_SNIPPET_CHARS = 800       # unchanged limit per snippet
SEMAPHORE = asyncio.Semaphore(30)

KB_CACHE: Optional[List[Dict[str, Any]]] = None
IDF_CACHE: Optional[Dict[str, float]] = None
VECTORS: Optional[List[Dict[str, float]]] = None
TOKENIZED_DOCS: Optional[List[List[str]]] = None
NORM_CACHE: Optional[List[float]] = None
CACHE_LOCK = asyncio.Lock()   # protect concurrent init

# ---------- Red‑flag handling ----------
BASE_RED_FLAGS = {
    "hypotension", "shock", "cardiac arrest", "unresponsive",
    "severe bleeding", "respiratory distress", "seizure",
    "altered mental status", "cyanosis", "airway obstruction",
    "pulseless", "asystole", "ventricular fibrillation",
}
# generate whitelist with simple variants
RED_FLAG_WHITELIST = {
    term for base in BASE_RED_FLAGS
    for term in {
        base,
        base + "s",
        base.replace(" ", "_"),
        base.replace(" ", ""),
    }
}
# pre‑compile regex patterns for word‑boundary matching
_RED_FLAG_PATTERNS = [re.compile(rf"\b{re.escape(w)}\b") for w in RED_FLAG_WHITELIST]

def _clean_flag(flag: str) -> str:
    """Normalize flag text for matching."""
    return re.sub(r"[^\w\s]", "", flag).lower()

def filter_red_flags(flags: List[str]) -> List[str]:
    """Keep flags that match any whitelist term as a whole word."""
    kept = []
    for f in flags:
        clean = _clean_flag(f)
        if any(p.search(clean) for p in _RED_FLAG_PATTERNS):
            kept.append(f)
    return kept

# ---------- Text utilities ----------
def tokenize(text: str) -> List[str]:
    """Extract alphabetic words (≥2 letters) and numbers."""
    return re.findall(r"[a-zA-Z]{2,}|\d+", text.lower())

def patient_text(patient: Dict[str, Any]) -> str:
    """Serialize patient fields for retrieval / prompting."""
    return (
        f"HPI: {patient.get('hpi', '').strip()}\n"
        f"Patient info: {patient.get('patient_info', '').strip()}\n"
        f"Initial vitals: {patient.get('initial_vitals', '').strip()}\n"
        f"Chief complaint: {patient.get('chief_complaint', '').strip()}"
    )

# ---------- KB loading & TF‑IDF ----------
async def _ensure_kb_loaded():
    async with CACHE_LOCK:
        global KB_CACHE, IDF_CACHE, VECTORS, TOKENIZED_DOCS, NORM_CACHE
        if KB_CACHE is None:
            KB_CACHE = load_esi_kb()
        if TOKENIZED_DOCS is None:
            TOKENIZED_DOCS = [tokenize(row["text"]) for row in KB_CACHE]
        if VECTORS is None:
            idf = build_idf(TOKENIZED_DOCS)
            VECTORS = [vector(toks, idf) for toks in TOKENIZED_DOCS]
            NORM_CACHE = [vec_norm(v) for v in VECTORS]

def load_kb():
    return KB_CACHE  # after _ensure_kb_loaded() has run

def build_idf(docs: List[List[str]]) -> Dict[str, float]:
    """Compute smoothed IDF; cache result."""
    global IDF_CACHE
    if IDF_CACHE is not None:
        return IDF_CACHE
    doc_occ = Counter()
    for tokens in docs:
        doc_occ.update(set(tokens))
    n = len(docs)
    IDF_CACHE = {
        token: math.log((n + 1) / (freq + 1)) + 1
        for token, freq in doc_occ.items()
    }
    return IDF_CACHE

def vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Sparse TF‑IDF vector; unknown tokens get modest weight (0.005)."""
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (cnt / total) * idf.get(t, 0.005) for t, cnt in tf.items()}

def vec_norm(vec: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))

def cosine(a: Dict[str, float], b: Dict[str, float],
           norm_a: float, norm_b: float) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

async def retrieve(text: str) -> str:
    """Return concatenated top‑K most similar KB snippets (truncated)."""
    await _ensure_kb_loaded()
    kb = load_kb()
    query_vec = vector(tokenize(text), IDF_CACHE)  # type: ignore
    query_norm = vec_norm(query_vec)

    ranked = sorted(
        enumerate(VECTORS),  # type: ignore
        key=lambda it: cosine(query_vec, it[1], query_norm, NORM_CACHE[it[0]]),  # type: ignore
        reverse=True,
    )

    pieces = []
    cur_len = 0
    for idx, _ in ranked[:TOP_K]:
        row = kb[idx]
        snippet = row["text"]
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
        addition = f"### {row['title']}\n{snippet}"
        if cur_len + len(addition) > MAX_SNIPPET_CHARS * TOP_K:
            break
        pieces.append(addition)
        cur_len += len(addition)

    return "\n\n".join(pieces)

# ---------- Prompt templates ----------
ANALYZE_SYSTEM = """You are an emergency triage nurse.
Extract only findings supported by the case.
Return JSON with the fields:
{
  "red_flags": ["..."],
  "abnormal_vitals": ["..."],
  "mental_status_concern": true,
  "severe_pain_or_distress": true,
  "likely_resources": ["labs","imaging","ekg","iv_medication","procedure","none"],
  "resource_count": 0
}
Now analyze the following case."""

CLASSIFY_SYSTEM = """You are a senior ESI triage nurse.
Using the original case text and the structured assessment provided,
choose the correct ESI level (1‑5) based on these rules:

1 = any red‑flag present.
2 = high‑risk condition, altered mental status, or severe pain/distress (no red flag).
3 = stable but needs two or more resources.
4 = likely needs one resource.
5 = likely needs no resources beyond exam.

Return JSON only: {"acuity": 1}
Now make your decision for the case below."""

# ---------- LLM helpers ----------
async def ask_json_with_retry(prompt: str, system: str, attempts: int = 5) -> Dict[str, Any]:
    """Retry JSON extraction; on persistent failure return empty dict."""
    for _ in range(attempts):
        try:
            resp = await ask_llm_json(prompt, system)
            if isinstance(resp, dict):
                return resp
        except Exception:
            continue
    return {}

def coerce_acuity(val) -> str:
    """Extract a valid integer 1‑5 from varied LLM outputs."""
    if not val:
        return ""
    try:
        acuity = int(float(val))
        if 1 <= acuity <= 5:
            return str(acuity)
    except Exception:
        pass
    word_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
    s = str(val).lower()
    for w, d in word_map.items():
        if w in s:
            return d
    m = re.search(r"\b([1-5])\b", s)
    return m.group(1) if m else ""

async def ask_with_retry(context: str, case: str) -> Dict[str, str]:
    """Primary LLM call; second stricter prompt on failure."""
    prompt = f"Reference material:\n{context}\n\nPatient case:\n{case}"
    res = await ask_llm_json(prompt, CLASSIFY_SYSTEM)
    acuity = coerce_acuity(res.get("acuity"))
    if acuity:
        return {"acuity": acuity}
    # second attempt – ask for plain number
    simple = f"Patient case:\n{case}\nAnswer with only the integer acuity (1‑5)."
    res2 = await ask_llm_json(simple, CLASSIFY_SYSTEM)
    acuity2 = coerce_acuity(res2.get("acuity"))
    return {"acuity": acuity2 or ""}

# ---------- Rule‑based fallback ----------
def extract_numbers(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\b\d+\b", text)]

def rule_fallback(patient: Dict[str, Any]) -> str:
    """Conservative vitals‑driven fallback when LLM yields nothing."""
    vitals = patient.get("initial_vitals", "").lower()
    life_threat = any(tok in vitals for tok in ["shock", "cardiac arrest", "unresponsive", "pulseless"])
    if life_threat:
        return "1"

    nums = extract_numbers(vitals)
    # softened thresholds (tuned)
    if any(n < 60 for n in nums):          # severe hypotension
        return "1"
    if any(n > 150 for n in nums):         # extreme tachycardia
        return "1"
    if any(n > 35 for n in nums):          # severe tachypnea (raised from 30)
        return "2"
    if any(n < 90 for n in nums):          # moderate hypoxia / low temp
        return "2"
    return "3"

# ---------- Assessment‑driven decision ----------
def decision_from_assessment(assessment: Dict[str, Any]) -> Optional[str]:
    """Derive ESI from structured assessment when red‑flags are absent."""
    if assessment.get("mental_status_concern"):
        return "2"
    if assessment.get("severe_pain_or_distress"):
        return "2"

    rc = int(assessment.get("resource_count", 0))
    if rc >= 2:
        return "3"
    if rc == 1:
        return "4"
    return "5"

# ---------- Core classification ----------
async def classify(patient: Dict[str, Any]) -> Dict[str, str]:
    async with SEMAPHORE:
        case = patient_text(patient)
        context = await retrieve(case)

        # 1️⃣ Structured assessment
        assess_prompt = f"{ANALYZE_SYSTEM}\n\nCase:\n{case}"
        assessment = await ask_json_with_retry(assess_prompt, ANALYZE_SYSTEM)

        # Ensure all expected keys
        for key, default in [
            ("red_flags", []),
            ("abnormal_vitals", []),
            ("mental_status_concern", False),
            ("severe_pain_or_distress", False),
            ("likely_resources", []),
            ("resource_count", 0),
        ]:
            assessment.setdefault(key, default)

        # Precise red‑flag filtering
        assessment["red_flags"] = filter_red_flags(assessment["red_flags"])

        # Immediate ESI‑1 if any red flag survives
        if assessment["red_flags"]:
            return {"predicted_acuity": "1"}

        # 2️⃣ Decision from resources (no bypass)
        resource_decision = decision_from_assessment(assessment)
        if resource_decision:
            # Verify with LLM; if LLM supplies a value, trust it
            primary = await ask_with_retry(context, case)
            if primary["acuity"]:
                return {"predicted_acuity": primary["acuity"]}
            # otherwise fall back to the rule‑based resource decision
            return {"predicted_acuity": resource_decision}

        # 3️⃣ Primary LLM classification (fallback when assessment gave nothing)
        primary = await ask_with_retry(context, case)
        acuity = primary["acuity"]

        # 4️⃣ Final fallback
        final_acuity = acuity or rule_fallback(patient)
        return {"predicted_acuity": final_acuity}

@measure_time()
async def entrypoint_async(patients: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    results = await asyncio.gather(
        *(classify(row["patient"]) for row in patients)
    )
    return {"results": results}

def entrypoint(patients: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    return asyncio.run(entrypoint_async(patients))