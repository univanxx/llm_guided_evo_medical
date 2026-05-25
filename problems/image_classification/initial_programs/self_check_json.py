# Adds explicit self-check to maximize JSON validity while keeping output strict.

PROMPT = """Classify the chest X-ray image as normal vs pneumonia.

Classes:
0 = normal
1 = pneumonia

Rules:
- Think silently; do NOT output reasoning.
- Output MUST be a single JSON object, and nothing else.
- JSON must have exactly 2 keys: class_id (0/1) and class_name ("normal"/"pneumonia").
- Before sending, verify the output is valid JSON.

Now answer with ONLY:
{"class_id": <0 or 1>, "class_name": "<normal or pneumonia>"}
"""


def entrypoint():
    return PROMPT
