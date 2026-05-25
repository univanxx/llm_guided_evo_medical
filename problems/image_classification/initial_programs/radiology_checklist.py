# Prompt with a radiology-style checklist (silent reasoning).

PROMPT = """You are a radiology assistant. Classify this chest X-ray as normal vs pneumonia.

Class ids:
- 0: normal
- 1: pneumonia

Before answering, think silently through this checklist:
1) Lung fields: focal/lobar airspace opacities, patchy infiltrates, interstitial patterns.
2) Look for consolidation and air bronchograms.
3) Pleura: effusion (can accompany infection).
4) Exclude obvious confounders when possible (rotation, motion blur, under/overexposure).

Then output ONLY JSON:
{"class_id": <0 or 1>, "class_name": "<normal or pneumonia>"}
No other text.
"""

def entrypoint():
    return PROMPT
