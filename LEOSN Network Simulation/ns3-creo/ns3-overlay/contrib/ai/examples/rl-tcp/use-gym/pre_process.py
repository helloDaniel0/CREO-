"""CREO+ DWT denoising and PDPA action-space construction.

This module intentionally contains only the two signal/action-space algorithms
described in the CREO+ paper.  It has no ns-3, PyTorch, DRL, training, or
ablation dependencies.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["daubechies_denoise", "pdpa_action_space"]


# Orthonormal Daubechies-2 (db2) analysis filters.  Periodic indexing is used
# below, matching the periodic DWT implementation used by the CREO+ prototype.
_SQRT_3 = math.sqrt(3.0)
_DB2_LOW = np.asarray(
    [1.0 + _SQRT_3, 3.0 + _SQRT_3, 3.0 - _SQRT_3, 1.0 - _SQRT_3],
    dtype=np.float64,
) / (4.0 * math.sqrt(2.0))
_DB2_HIGH = np.asarray(
    [_DB2_LOW[3], -_DB2_LOW[2], _DB2_LOW[1], -_DB2_LOW[0]],
    dtype=np.float64,
)


def _as_finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    """Return *values* as a finite, one-dimensional float64 array."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the paper's sign-preserving soft-threshold operator."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def daubechies_denoise(values: Iterable[float]) -> np.ndarray:
    """Denoise a capacity sequence using periodic db2 DWT.

    The implementation follows the CREO+ connected-phase processing path:

    1. pad the sequence to the next power of two;
    2. recursively compute db2 approximation and detail coefficients;
    3. estimate each level's noise with ``median(abs(detail)) / 0.6745``;
    4. apply the universal soft threshold ``sigma * sqrt(2 * ln(L))``; and
    5. reconstruct with the thresholded detail coefficients.

    Sequences shorter than eight samples are returned unchanged because they
    cannot support a useful multi-level db2 decomposition in this implementation.
    The returned sequence always has the original length and ``float32`` dtype.
    """
    original = _as_finite_vector(values, "values")
    if original.size < 8:
        return original.astype(np.float32, copy=True)

    target_length = 1 << int(math.ceil(math.log2(original.size)))
    approximation = np.pad(
        original,
        (0, target_length - original.size),
        mode="edge",
    )
    denoised_details: list[np.ndarray] = []

    # Each pass filters the previous approximation and downsamples by two.
    while approximation.size >= 8:
        coefficient_count = approximation.size // 2
        next_approximation = np.empty(coefficient_count, dtype=np.float64)
        detail = np.empty(coefficient_count, dtype=np.float64)

        for index in range(coefficient_count):
            positions = (2 * index + np.arange(4)) % approximation.size
            samples = approximation[positions]
            next_approximation[index] = np.dot(_DB2_LOW, samples)
            detail[index] = np.dot(_DB2_HIGH, samples)

        sigma = float(np.median(np.abs(detail)) / 0.6745)
        threshold = sigma * math.sqrt(2.0 * math.log(original.size))
        denoised_details.append(_soft_threshold(detail, threshold))
        approximation = next_approximation

    # Inverse DWT.  Crucially, reconstruction uses the thresholded detail
    # coefficients rather than the original noisy coefficients.
    for detail in reversed(denoised_details):
        reconstructed = np.zeros(detail.size * 2, dtype=np.float64)
        for index in range(detail.size):
            positions = (2 * index + np.arange(4)) % reconstructed.size
            reconstructed[positions] += (
                _DB2_LOW * approximation[index] + _DB2_HIGH * detail[index]
            )
        approximation = reconstructed

    return approximation[: original.size].astype(np.float32)


@dataclass(frozen=True)
class _PDPASolution:
    """One Pareto-DP state containing paired decrease/increase actions."""

    negative: tuple[float, ...]
    positive: tuple[float, ...]
    coverage: frozenset[int]
    coverage_probability: float
    cost: float


def _pdpa_score(
    negative: tuple[float, ...],
    positive: tuple[float, ...],
    support: np.ndarray,
    probability: np.ndarray,
    lag: int,
    tolerance: float,
) -> _PDPASolution:
    """Evaluate the coverage and minimum combination cost of one action set."""
    actions = negative + positive
    achievable: list[tuple[float, int]] = []

    # In log space, multiplying actions becomes addition.  The paper permits
    # combinations of at most lag - 1 non-neutral actions.
    for step_count in range(1, lag):
        for sequence in itertools.product(actions, repeat=step_count):
            achievable.append((sum(sequence), step_count))

    covered: set[int] = set()
    weighted_cost = 0.0
    for index, variation in enumerate(support):
        matching_costs = [
            step_count
            for total, step_count in achievable
            if abs(total - variation) <= tolerance
        ]
        if matching_costs:
            covered.add(index)
            weighted_cost += probability[index] * min(matching_costs)

    coverage_probability = float(sum(probability[index] for index in covered))
    expected_cost = weighted_cost / max(coverage_probability, 1e-12)
    return _PDPASolution(
        negative=tuple(sorted(negative)),
        positive=tuple(sorted(positive)),
        coverage=frozenset(covered),
        coverage_probability=coverage_probability,
        cost=float(expected_cost),
    )


def _dominates(left: _PDPASolution, right: _PDPASolution) -> bool:
    """Return whether *left* Pareto-dominates *right*."""
    no_worse = left.coverage.issuperset(right.coverage) and left.cost <= right.cost
    strictly_better = left.coverage != right.coverage or left.cost < right.cost
    return no_worse and strictly_better


def _pareto_prune(
    solutions: Iterable[_PDPASolution],
    frontier_limit: int | None,
) -> list[_PDPASolution]:
    """Remove duplicate and dominated PDPA states."""
    unique: dict[
        tuple[tuple[float, ...], tuple[float, ...]],
        _PDPASolution,
    ] = {}
    for solution in solutions:
        key = (solution.negative, solution.positive)
        previous = unique.get(key)
        if previous is None or (
            solution.coverage_probability,
            -solution.cost,
        ) > (
            previous.coverage_probability,
            -previous.cost,
        ):
            unique[key] = solution

    frontier: list[_PDPASolution] = []
    ordered = sorted(
        unique.values(),
        key=lambda item: (-item.coverage_probability, item.cost),
    )
    for candidate in ordered:
        if any(_dominates(existing, candidate) for existing in frontier):
            continue
        frontier = [
            existing
            for existing in frontier
            if not _dominates(candidate, existing)
        ]
        frontier.append(candidate)

    frontier.sort(key=lambda item: (-item.coverage_probability, item.cost))
    if frontier_limit is not None:
        return frontier[:frontier_limit]
    return frontier


def _candidate_bins(
    support: np.ndarray,
    probability: np.ndarray,
    sign: int,
    count: int,
    tolerance: float,
) -> list[float]:
    """Choose frequent and boundary DPMF bins for one action direction."""
    indices = [
        index
        for index, value in enumerate(support)
        if value * sign > tolerance * 0.5
    ]
    ranked = sorted(indices, key=lambda index: probability[index], reverse=True)
    selected = ranked[: max(3, count - 2)]

    if indices:
        selected.append(min(indices, key=lambda index: support[index] * sign))
        selected.append(max(indices, key=lambda index: support[index] * sign))

    candidates = {float(support[index]) for index in selected}
    synthetic_step = tolerance
    while len(candidates) < 3:
        candidates.add(sign * synthetic_step)
        synthetic_step += tolerance

    return sorted(sorted(candidates, key=abs)[:count])


def pdpa_action_space(
    capacities: Iterable[float],
    lag: int = 3,
    tolerance: float = 0.025,
    candidate_count: int = 7,
    frontier_limit: int | None = 256,
) -> tuple[list[float], dict[str, object]]:
    """Construct CREO+'s seven-action space with Pareto dynamic programming.

    ``capacities`` is the recent denoised capacity sequence.  For granularity
    ``lag``, the function constructs the DPMF of ``c[i] / c[i-lag]``, moves it
    to log space, and builds three decrement/increment action pairs.  At each
    stage it retains only non-dominated states according to coverage and
    expected combination cost.

    The returned multiplicative action space is ordered as three decreases,
    the neutral action ``1.0``, and three increases.  Set ``frontier_limit`` to
    ``None`` for an exact, unbounded Pareto frontier; the default preserves the
    bounded behavior of the current CREO+ implementation.
    """
    capacity = _as_finite_vector(capacities, "capacities")
    if np.any(capacity <= 0.0):
        raise ValueError("capacities must contain only positive values")
    if not isinstance(lag, int) or lag < 2:
        raise ValueError("lag must be an integer of at least 2")
    if capacity.size <= lag:
        raise ValueError("capacities must contain more samples than lag")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")
    if not isinstance(candidate_count, int) or candidate_count < 3:
        raise ValueError("candidate_count must be an integer of at least 3")
    if frontier_limit is not None and (
        not isinstance(frontier_limit, int) or frontier_limit < 1
    ):
        raise ValueError("frontier_limit must be None or a positive integer")

    ratios = capacity[lag:] / capacity[:-lag]
    log_ratios = np.log(np.clip(ratios, 0.25, 4.0))
    quantized = np.round(log_ratios / tolerance) * tolerance
    support, counts = np.unique(quantized, return_counts=True)
    probability = counts.astype(np.float64) / counts.sum()

    negative_candidates = _candidate_bins(
        support,
        probability,
        sign=-1,
        count=candidate_count,
        tolerance=tolerance,
    )
    positive_candidates = _candidate_bins(
        support,
        probability,
        sign=1,
        count=candidate_count,
        tolerance=tolerance,
    )

    frontier = [_PDPASolution((), (), frozenset(), 0.0, 0.0)]
    frontier_sizes: list[int] = []
    for _ in range(3):
        expanded: list[_PDPASolution] = []
        for solution in frontier:
            for negative in negative_candidates:
                if negative in solution.negative:
                    continue
                for positive in positive_candidates:
                    if positive in solution.positive:
                        continue
                    expanded.append(
                        _pdpa_score(
                            solution.negative + (negative,),
                            solution.positive + (positive,),
                            support,
                            probability,
                            lag,
                            tolerance,
                        )
                    )

        frontier = _pareto_prune(expanded, frontier_limit)
        frontier_sizes.append(len(frontier))
        if not frontier:
            raise RuntimeError("PDPA produced an empty Pareto frontier")

    selected = min(
        frontier,
        key=lambda item: (-item.coverage_probability, item.cost),
    )
    decreases = sorted(math.exp(value) for value in selected.negative)
    increases = sorted(math.exp(value) for value in selected.positive)
    actions = decreases + [1.0] + increases
    metadata: dict[str, object] = {
        "lag": lag,
        "tolerance": tolerance,
        "ratio_samples": int(log_ratios.size),
        "support_bins": int(support.size),
        "negative_candidates": len(negative_candidates),
        "positive_candidates": len(positive_candidates),
        "frontier_sizes": frontier_sizes,
        "coverage_probability": selected.coverage_probability,
        "expected_combination_cost": selected.cost,
    }
    return actions, metadata
