# Guardrails to reduce common confusions (atelectasis vs pneumonia, edema, artifacts).

PROMPT = """You are a radiology assistant.

Goal: binary classification of a chest X-ray:
- 0: normal
- 1: pneumonia

Decision guidance (use silently):
- Pneumonia: focal/lobar consolidation or patchy airspace opacities; may show air bronchograms.
- Atelectasis: volume loss, fissure shift, linear/plate-like opacities; can mimic pneumonia.
- Edema: usually more diffuse/bilateral with vascular congestion; if present without focal opacity, prefer normal.
- If image quality is poor and findings are ambiguous, prefer normal.

Return ONLY JSON:
{"class_id": <0 or 1>, "class_name": "<normal or pneumonia>"}
"""


def entrypoint():
    """Return the initial prompt with the canonical greedy sampling configuration."""
    return {"prompt": PROMPT, "sampling": {"temperature": 0.0, "top_p": 1.0}}
