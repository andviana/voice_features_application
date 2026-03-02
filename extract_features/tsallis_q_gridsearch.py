"""
    Tsallis q grid search (maximize DP vs HC separation via t-stat)

    Research project context
    ------------------------
    Parkinson’s disease (PD) voice analysis feature extraction as described in the DP research plan.

    Design goals
    ------------
    - Deterministic, reproducible feature extraction for ML / Deep Learning pipelines.
    - Stable feature names (dict outputs) to support dataset building and model comparisons.
    - Explicit edge-case behavior (empty signals, constant signals, unvoiced frames).

    Dependencies
    ------------
    - numpy
    - scipy

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import ttest_ind


def tsallis_entropy(p: np.ndarray, q: float) -> float:
    p = np.asarray(p, dtype=float).flatten()
    p = p[np.isfinite(p)]
    p = p[p > 0.0]
    if p.size == 0:
        return float("nan")

    if q == 1.0:
        return float(-np.sum(p * np.log(p)))

    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


@dataclass(frozen=True)
class GridSearchResult:
    q_opt: float
    score_opt: float
    q_grid: np.ndarray
    scores: np.ndarray


def grid_search_q(
    p_group0: np.ndarray,
    p_group1: np.ndarray,
    q_min: float = 0.1,
    q_max: float = 3.0,
    q_step: float = 0.05,
) -> Tuple[GridSearchResult, Dict[str, float]]:
    """Select q maximizing |t| where t compares Sq distributions between two groups.

    Inputs
    ------
    p_group0: shape (n0, B)
    p_group1: shape (n1, B)
    """
    p0 = np.asarray(p_group0, dtype=float)
    p1 = np.asarray(p_group1, dtype=float)

    if p0.ndim != 2 or p1.ndim != 2 or p0.shape[1] != p1.shape[1]:
        raise ValueError("Expected p arrays with shapes (n, B) for both groups, same B.")

    q_grid = np.arange(q_min, q_max + 1e-12, q_step, dtype=float)
    scores = np.full_like(q_grid, np.nan, dtype=float)

    for i, q in enumerate(q_grid):
        s0 = np.array([tsallis_entropy(p, q) for p in p0], dtype=float)
        s1 = np.array([tsallis_entropy(p, q) for p in p1], dtype=float)

        s0 = s0[np.isfinite(s0)]
        s1 = s1[np.isfinite(s1)]
        if s0.size < 2 or s1.size < 2:
            continue

        t_stat, _ = ttest_ind(s0, s1, equal_var=False, nan_policy="omit")
        scores[i] = abs(float(t_stat))

    if not np.any(np.isfinite(scores)):
        res = GridSearchResult(q_opt=float("nan"), score_opt=float("nan"), q_grid=q_grid, scores=scores)
        return res, {"q_grid_opt": res.q_opt, "q_grid_score_opt": res.score_opt}

    idx = int(np.nanargmax(scores))
    res = GridSearchResult(
        q_opt=float(q_grid[idx]),
        score_opt=float(scores[idx]),
        q_grid=q_grid,
        scores=scores,
    )
    return res, {"q_grid_opt": res.q_opt, "q_grid_score_opt": res.score_opt}


def _sanity_test() -> None:
    rng = np.random.default_rng(0)
    B = 50

    # Two synthetic groups of discrete distributions:
    # group0 concentrated, group1 more spread => typically higher entropy.
    p0 = rng.dirichlet(10.0 * np.ones(B), size=20)
    p1 = rng.dirichlet(3.0 * np.ones(B), size=20)

    res, d = grid_search_q(p0, p1)
    assert "q_grid_opt" in d and "q_grid_score_opt" in d


if __name__ == "__main__":
    _sanity_test()
    print("OK - tsallis_q_gridsearch.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
