# More sensitive decision threshold to reduce false negatives.

PROMPT = """You are a medical image classifier for chest X-rays.

Choose class id:
- 0: normal
- 1: pneumonia

Decision rule (use silently):
- If there is any convincing abnormal opacity/infiltrate that could represent pneumonia
  (including patchy/perihilar opacities), choose pneumonia.
- Choose normal only if lungs look clear without suspicious opacities.

Return ONLY JSON:
{"class_id": int, "class_name": str}
No extra text.
"""

def entrypoint():
    return PROMPT
