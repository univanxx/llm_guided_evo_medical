# Add explicit "format repair" instruction to maximize parse_success_rate.

PROMPT = """Task: binary classification of chest X-ray.

Classes:
0 = normal
1 = pneumonia

Output MUST be valid JSON (single object), EXACTLY these two keys:
- class_id (integer 0 or 1)
- class_name (string "normal" or "pneumonia")

If you accidentally start writing non-JSON text, STOP and output only the JSON object.

Now classify the image and output ONLY the JSON object.
"""

def entrypoint():
    return PROMPT
