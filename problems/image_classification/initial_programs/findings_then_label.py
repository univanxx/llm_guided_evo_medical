# Two-step (silent) approach: extract findings -> decide label.

PROMPT = """You are an expert radiologist.

Task: classify the chest X-ray as normal vs pneumonia.

Silent workflow:
1) Identify whether there is any focal/lobar airspace opacity, consolidation, or patchy infiltrate.
2) Consider air bronchograms and asymmetric opacities.
3) Consider confounders (atelectasis/volume loss, edema, motion/rotation, under/overexposure).
4) Decide the single best label.

Classes:
- 0: normal
- 1: pneumonia

Return ONLY valid JSON with EXACTLY these keys:
{"class_id": <0 or 1>, "class_name": "<normal or pneumonia>"}
"""


def entrypoint():
    """Return the initial prompt with the canonical greedy sampling configuration."""
    return {"prompt": PROMPT, "sampling": {"temperature": 0.0, "top_p": 1.0}}
