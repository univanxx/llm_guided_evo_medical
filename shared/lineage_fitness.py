"""
Lineage-aware fitness scoring for evolutionary optimization.

When a program is evaluated in an evolutionary context, its raw fitness score
can be noisy due to batch sampling variance. Lineage blending and validity
gating reduce this noise by incorporating information from the program's
ancestor chain.

Concepts
--------
ancestor_metrics : list[dict]
    Metrics from the program's ancestors. ``[0]`` is the immediate parent,
    ``[1]`` the grandparent, and so on. Each dict must have at least a
    ``"fitness"`` key (float).

lineage_blended_fitness
    A weighted blend of the current raw fitness and the exponentially
    weighted average of ancestor fitness values:

        blended = (1 - w) * raw + w * ancestor_avg

    Useful for logging and smoothed comparisons. Equals ``raw`` when no
    ancestor data is available.

is_valid
    A binary flag (1.0 / 0.0) that gates out runs with an anomalously large
    fitness drop relative to the ancestor lineage. Specifically, if
    ``ancestor_avg - raw > FITNESS_DELTA_THRESHOLD``, the run is marked
    invalid (likely noise or a broken mutation).

Usage
-----
    from lineage_fitness import compute_lineage_metrics

    raw_fitness = 0.72
    ancestor_metrics = context.get("ancestor_metrics", [])

    result = compute_lineage_metrics(raw_fitness, ancestor_metrics)
    # result = {
    #     "lineage_blended_fitness": 0.731,
    #     "ancestor_fitness_gap":    0.04,
    #     "is_valid":                1.0,
    # }
"""

from __future__ import annotations

from typing import Dict, List, Optional


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
   weighted ancestor average before the run is marked invalid."""


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def weighted_ancestor_fitness(
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


def lineage_blended_fitness(
    raw: float,
    ancestor_avg: Optional[float],
    ancestor_weight: float = ANCESTOR_WEIGHT,
) -> float:
    """Blend raw fitness with the weighted ancestor average.

    Returns ``raw`` when ``ancestor_avg`` is ``None`` (no ancestor data).
    """
    if ancestor_avg is None:
        return raw
    return (1.0 - ancestor_weight) * raw + ancestor_weight * ancestor_avg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_lineage_metrics(
    raw_fitness: float,
    ancestor_metrics: List[dict],
    *,
    key: str = "fitness",
    decay: float = EXP_DECAY,
    ancestor_weight: float = ANCESTOR_WEIGHT,
    delta_threshold: float = FITNESS_DELTA_THRESHOLD,
) -> Dict[str, float]:
    """Compute lineage-aware fitness metrics from a raw score.

    Parameters
    ----------
    raw_fitness:
        Raw fitness value for the current run (e.g. accuracy).
    ancestor_metrics:
        List of ancestor metric dicts, ordered from nearest to furthest.
    key:
        Which metric key to use from ancestor dicts (default: ``"fitness"``).
    decay:
        Exponential decay for ancestor weighting.
    ancestor_weight:
        Blend weight for ancestor average in ``lineage_blended_fitness``.
    delta_threshold:
        Fitness drop threshold above which the run is marked invalid.

    Returns
    -------
    dict with:
        - ``lineage_blended_fitness`` : float
        - ``ancestor_fitness_gap``    : float (0.0 when no ancestors)
        - ``is_valid``                : float (1.0 or 0.0)
    """
    ancestor_avg = weighted_ancestor_fitness(ancestor_metrics, key=key, decay=decay)

    blended = lineage_blended_fitness(raw_fitness, ancestor_avg, ancestor_weight)

    gap = 0.0
    if ancestor_avg is not None:
        gap = ancestor_avg - raw_fitness

    is_valid = 1.0
    if ancestor_avg is not None and gap > delta_threshold:
        is_valid = 0.0

    return {
        "lineage_blended_fitness": float(blended),
        "ancestor_fitness_gap": float(gap),
        "is_valid": is_valid,
    }
