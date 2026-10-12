# Inspired by the MedGemma technical report zero-shot CXR finding prompt style.
# Adapted to our binary PneumoniaMNIST task with strict JSON-only output.

PROMPT = """You are a radiology assistant.

Question: Is there pneumonia in this chest X-ray image?

You may reason silently. Then give a final very short, definitive answer as STRICT JSON ONLY.

Classes:
- 0: normal
- 1: pneumonia

Output ONLY this JSON object (no other text):
{"class_id": <0 or 1>, "class_name": "<normal or pneumonia>"}
"""


def entrypoint():
    """Return the initial prompt with the canonical greedy sampling configuration."""
    return {"prompt": PROMPT, "sampling": {"temperature": 0.0, "top_p": 1.0}}
