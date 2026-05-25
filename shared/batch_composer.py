"""
Adaptive batch composition for evolutionary evaluation.

BatchComposer builds evaluation batches from three parts to provide stable,
informative fitness signals across generations:

  1. Anchor set  — small fixed subset refreshed every ``anchor_refresh_interval``
                   generations. Provides a stable baseline for cross-generation
                   fitness comparison.
  2. Hard buffer — samples with the highest accumulated error rates across
                   recent evaluations. Maintains selective pressure on genuinely
                   difficult cases.
  3. Random fill — fresh random samples drawn per generation for exploration
                   diversity.

All candidates within the same generation receive identical batches (the batch
is deterministic given ``iteration`` and the current persisted sample stats).

Usage
-----
    composer = BatchComposer(
        total_data_size=len(dataset),
        batch_size=50,
        state_path="data/batch_state.json",
    )
    indices = composer.compose_batch(iteration=seed)
    subset = [dataset[i] for i in indices]
    # ... evaluate subset ...
    composer.update_results(indices, correct_mask, iteration=seed)
"""

from __future__ import annotations

import fcntl
import json
import os
import random
from typing import List


class BatchComposer:
    """Composes evaluation batches from anchor + hard-buffer + random portions.

    Parameters
    ----------
    total_data_size:
        Total number of samples in the full dataset.
    batch_size:
        Number of samples returned per :meth:`compose_batch` call.
    state_path:
        Path to the JSON file persisting per-sample statistics. The parent
        directory is created automatically. Concurrent processes are safe:
        reads use ``LOCK_SH``, writes use ``LOCK_EX``.
    anchor_size:
        Number of anchor samples per batch.
    hard_buffer_max_size:
        Maximum number of hard-buffer (high-error) samples per batch.
    anchor_refresh_interval:
        Anchor set is refreshed every this many iterations.
    anchor_base_seed:
        RNG seed base for anchor selection.
    hard_buffer_base_seed:
        RNG seed base for hard-buffer weighted sampling.
    """

    def __init__(
        self,
        total_data_size: int,
        batch_size: int = 50,
        state_path: str = "data/batch_state.json",
        anchor_size: int = 20,
        hard_buffer_max_size: int = 30,
        anchor_refresh_interval: int = 5,
        anchor_base_seed: int = 42,
        hard_buffer_base_seed: int = 123456,
    ) -> None:
        self.total_data_size = total_data_size
        self.batch_size = batch_size
        self.state_path = state_path
        self.anchor_size = anchor_size
        self.hard_buffer_max_size = hard_buffer_max_size
        self.anchor_refresh_interval = anchor_refresh_interval
        self.anchor_base_seed = anchor_base_seed
        self.hard_buffer_base_seed = hard_buffer_base_seed

    # -- state persistence ------------------------------------------------

    @staticmethod
    def _default_state() -> dict:
        return {"sample_stats": {}}

    def _read_state(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _write_state(self, state: dict) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, self.state_path)

    # -- batch part builders ----------------------------------------------

    def _compute_anchors(self, iteration: int) -> List[int]:
        anchor_epoch = iteration // self.anchor_refresh_interval
        rng = random.Random(self.anchor_base_seed + anchor_epoch)
        return sorted(
            rng.sample(
                range(self.total_data_size),
                k=min(self.anchor_size, self.total_data_size),
            )
        )

    def _compute_hard_buffer(
        self, sample_stats: dict, exclude: set, iteration: int
    ) -> List[int]:
        indices: List[int] = []
        weights: List[float] = []
        for idx_str, s in sample_stats.items():
            idx = int(idx_str)
            if idx in exclude or idx >= self.total_data_size:
                continue
            evals = s.get("evals", 0)
            if evals < 1:
                continue
            error_rate = s.get("errors", 0) / evals
            if error_rate > 0:
                indices.append(idx)
                weights.append(error_rate)
        if not indices:
            return []
        k = min(self.hard_buffer_max_size, len(indices))
        rng = random.Random(self.hard_buffer_base_seed + iteration)
        selected: List[int] = []
        for _ in range(k):
            chosen = rng.choices(indices, weights=weights, k=1)[0]
            pos = indices.index(chosen)
            indices.pop(pos)
            weights.pop(pos)
            selected.append(chosen)
            if not indices:
                break
        return selected

    # -- public API -------------------------------------------------------

    def compose_batch(self, iteration: int) -> List[int]:
        """Return sample indices for this evaluation call.

        The result is deterministic given ``iteration`` and the persisted
        ``sample_stats``. All candidates in the same generation should call
        this with the same ``iteration`` value.
        """
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        lock_path = self.state_path + ".lock"
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                state = self._read_state()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

        anchor_indices = self._compute_anchors(iteration)
        anchor_set = set(anchor_indices)

        hard_indices = self._compute_hard_buffer(
            state.get("sample_stats", {}), anchor_set, iteration
        )
        hard_set = set(hard_indices)

        used = anchor_set | hard_set
        remaining = [i for i in range(self.total_data_size) if i not in used]
        random_needed = max(
            0, self.batch_size - len(anchor_indices) - len(hard_indices)
        )
        rng = random.Random(iteration)
        random_indices = rng.sample(remaining, k=min(random_needed, len(remaining)))

        return (anchor_indices + hard_indices + random_indices)[: self.batch_size]

    def update_results(
        self,
        batch_indices: List[int],
        correct_mask: List[bool],
        iteration: int,
    ) -> None:
        """Increment per-sample error/eval counters."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        lock_path = self.state_path + ".lock"
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self._read_state()
                stats = state.setdefault("sample_stats", {})
                for idx, correct in zip(batch_indices, correct_mask):
                    key = str(idx)
                    if key not in stats:
                        stats[key] = {"errors": 0, "evals": 0}
                    stats[key]["evals"] += 1
                    if not correct:
                        stats[key]["errors"] += 1
                self._write_state(state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
