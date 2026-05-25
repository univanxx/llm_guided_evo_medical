"""
Computes accuracy and token usage from doctor-patient dialogues.

When called with an evolution-aware context (ancestor_metrics), applies:
  - ``fitness`` is raw diagnostic accuracy; ``lineage_blended_fitness`` blends
    current accuracy with exponentially weighted ancestor ``fitness`` (for logging).
  - Delta-based validity gate: if raw fitness is below the weighted ancestor
    average by more than FITNESS_DELTA_THRESHOLD, the solution is marked invalid
    (is_valid=0) to suppress noise.
  - Forced-final cap: the fraction of cases ending in a forced final answer.
"""

from typing import Dict, List, Optional
import os
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
EXP_DECAY: float = 0.7
"""Decay factor for exponential ancestor weighting (0 < decay <= 1).
   Weight of ancestor at depth *i* is ``decay ** i``."""

ANCESTOR_WEIGHT: float = 0.3
"""Weight on the weighted ancestor average in ``lineage_blended_fitness``.
   ``blend = (1 - ANCESTOR_WEIGHT) * raw + ANCESTOR_WEIGHT * ancestor_avg``."""

FITNESS_DELTA_THRESHOLD: float = 0.15
"""Maximum allowed absolute difference between current fitness and the
   weighted ancestor average.  Exceeding this marks the mutant as invalid."""

# FORCED_FINAL_THRESHOLD: float = 0.5
# """Maximum allowed fraction of forced final answers."""

# ---------------------------------------------------------------------------
# Tokenizer (lazy-loaded)
# ---------------------------------------------------------------------------
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        model_name = os.environ.get("MEDIQ_MODEL_NAME")
        if model_name and model_name.lower().startswith("meta-llama/"):
            _, suffix = model_name.split("/", 1)
            model_name = f"meta-llama/{suffix}"
        # Use HF_TOKEN for gated models (e.g. Meta-Llama); set in run.sh or env.
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
    return _TOKENIZER


# ---------------------------------------------------------------------------
# Ancestor weighting helpers
# ---------------------------------------------------------------------------

def _weighted_ancestor_fitness(
    ancestor_metrics: List[dict],
    key: str = "fitness",
    decay: float = EXP_DECAY,
) -> Optional[float]:
    """Return the exponentially weighted average of *key* across ancestors.

    ``ancestor_metrics[0]`` is the immediate parent, ``[1]`` the grandparent, etc.
    Returns ``None`` when no ancestor carries *key*.
    """
    if not ancestor_metrics:
        return None

    total_weight = 0.0
    weighted_sum = 0.0
    for i, m in enumerate(ancestor_metrics):
        val = m.get(key)
        if val is None:
            continue
        w = decay ** i
        weighted_sum += val * w
        total_weight += w

    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def _compute_lineage_blended_fitness(
    raw: float,
    ancestor_avg: Optional[float],
    ancestor_weight: float = ANCESTOR_WEIGHT,
) -> float:
    """Blend raw accuracy with weighted ancestor average of fitness."""
    if ancestor_avg is None:
        return raw
    return (1.0 - ancestor_weight) * raw + ancestor_weight * ancestor_avg


# ---------------------------------------------------------------------------
# Main validation entry-point
# ---------------------------------------------------------------------------

def validate(context_or_data, data=None) -> Dict[str, float]:
    """Compute accuracy, token cost, and validity.

    Supports two call signatures (auto-detected):
      - ``validate(data)``           – no evolution context
      - ``validate(context, data)``  – with evolution-aware context
    """
    if data is None:
        data = context_or_data
        context: Optional[dict] = None
    else:
        context = context_or_data

    dialogues, diagnoses, _case_ids, ground_truth, run_metadata = data

    if len(ground_truth) != len(diagnoses):
        raise ValueError(
            f"Number of cases for predictions and ground truth must match! "
            f"Got {len(diagnoses)} for predictions, should be {len(ground_truth)}"
        )

    n_cases = len(ground_truth)

    tokenizer = _get_tokenizer()
    correct_answs = 0
    total_expert_tokens = 0
    expert_responses = run_metadata["expert_responses"]

    token_cases = 0
    for case_responses, diagnosis_i, ground_truth_i in zip(
        expert_responses, diagnoses, ground_truth
    ):
        if diagnosis_i == ground_truth_i:
            correct_answs += 1
        if case_responses:
            token_cases += 1
            for text in case_responses:
                total_expert_tokens += len(
                    tokenizer.encode(str(text), add_special_tokens=False)
                )

    raw_fitness = correct_answs / len(ground_truth)
    tokens_count = (
        total_expert_tokens / token_cases if token_cases > 0 else 0.0
    )

    # --- Evolution-aware adjustments ------------------------------------------
    ancestor_metrics: List[dict] = []
    if context is not None:
        ancestor_metrics = context.get("ancestor_metrics", [])

    ancestor_avg = _weighted_ancestor_fitness(ancestor_metrics, key="fitness")

    lineage_blended = _compute_lineage_blended_fitness(raw_fitness, ancestor_avg)

    ancestor_fitness_gap = 0.0
    if ancestor_avg is not None:
        ancestor_fitness_gap = ancestor_avg - raw_fitness

    is_valid = 1.0
    # if forced_final_rate > FORCED_FINAL_THRESHOLD:
    #     is_valid = 0.0
    if ancestor_avg is not None and ancestor_fitness_gap > FITNESS_DELTA_THRESHOLD:
        is_valid = 0.0

    return {
        "fitness": float(raw_fitness),
        "lineage_blended_fitness": float(lineage_blended),
        "tokens_count": float(tokens_count),
        "ancestor_fitness_gap": float(ancestor_fitness_gap),
        "is_valid": is_valid,
    }
