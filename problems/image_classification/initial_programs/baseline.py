# Baseline: minimal, strict JSON-only output.

PROMPT = """You are a medical image classifier.
Given a chest X-ray image, choose EXACTLY ONE class id:

- 0: normal (no pneumonia)
- 1: pneumonia

Return ONLY a JSON object with keys: class_id (int), class_name (string).
No extra keys. No markdown. No explanation.
"""

def entrypoint():
    """Return the initial prompt with the canonical greedy sampling configuration."""
    return {"prompt": PROMPT, "sampling": {"temperature": 0.0, "top_p": 1.0}}
