# Conservative decision threshold to reduce false positives.

PROMPT = """You are a medical image classifier for chest X-rays.

Choose class id:
- 0: normal
- 1: pneumonia

Decision rule (use silently):
- Output pneumonia ONLY if you see a clear focal/lobar airspace opacity or convincing infiltrates
  consistent with infection (e.g., consolidation +/- air bronchograms).
- If findings are subtle/ambiguous or quality is poor, prefer normal.

Return ONLY JSON:
{"class_id": int, "class_name": str}
"""

def entrypoint():
    """Return the initial prompt with the canonical greedy sampling configuration."""
    return {"prompt": PROMPT, "sampling": {"temperature": 0.0, "top_p": 1.0}}
