"""
    Tsallis q via extensivity (scaling linearity)

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

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def tsallis_entropy(p: np.ndarray, q: float) -> float:
    p = np.asarray(p, dtype=float).flatten()
    p = p[np.isfinite(p)]
    p = p[p > 0.0]
    if p.size == 0:
        return float("nan")

    if q == 1.0:
        return float(-np.sum(p * np.log(p)))

    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


def amplitude_hist_p(z: np.ndarray, n_bins: int, clip_sigma: float) -> np.ndarray:
    z = np.asarray(z, dtype=float).flatten()
    z = np.clip(z, -clip_sigma, clip_sigma)
    edges = np.linspace(-clip_sigma, clip_sigma, n_bins + 1)
    hist, _ = np.histogram(z, bins=edges)
    p = hist.astype(float)
    s = float(np.sum(p))
    return p / s if s > 0 else p


@dataclass(frozen=True)
class ExtensivityResult:
    q_opt: float
    r2_opt: float
    q_grid: np.ndarray
    r2: np.ndarray


def estimate_q_extensivity(
    y: np.ndarray,
    q_min: float = 0.1,
    q_max: float = 3.0,
    q_step: float = 0.05,
    n_bins: int = 100,

    clip_sigma: float = 5.0,
    segment_fracs: Tuple[float, ...] = (1.0, 0.5, 0.25),
) -> Tuple[ExtensivityResult, Dict[str, float]]:
    """Pick q that makes Sq scale ~ linearly with segment length.

    Operationalization:
    - z-score normalize y
    - take prefix segments of different lengths
    - compute Sq for each length
    - fit Sq ~ a*L + b and compute R^2
    - pick q that maximizes R^2
    """
    y = np.asarray(y, dtype=float).flatten()
    if y.size < 100:
        res = ExtensivityResult(float("nan"), float("nan"), np.array([]), np.array([]))
        return res, {"q_ext_hat": res.q_opt, "q_ext_r2": res.r2_opt}

    mu = float(np.mean(y))
    sd = float(np.std(y, ddof=1) if y.size > 1 else 0.0)
    if sd <= 0.0:
        res = ExtensivityResult(1.0, 1.0, np.array([1.0]), np.array([1.0]))
        return res, {"q_ext_hat": 1.0, "q_ext_r2": 1.0}

    z = (y - mu) / sd
    N = z.size
    lengths = np.unique([max(10, int(round(fr * N))) for fr in segment_fracs])
    lengths = lengths[lengths <= N].astype(int)

    q_grid = np.arange(q_min, q_max + 1e-12, q_step, dtype=float)
    r2_vals = np.full_like(q_grid, np.nan, dtype=float)

    for i, q in enumerate(q_grid):
        S = []
        for L in lengths:
            p = amplitude_hist_p(z[:L], n_bins=n_bins, clip_sigma=clip_sigma)
            S.append(tsallis_entropy(p, q=q))
        S = np.asarray(S, dtype=float)
        if not np.all(np.isfinite(S)):
            continue

        x = lengths.astype(float)
        x0 = float(np.mean(x))
        y0 = float(np.mean(S))
        denom = float(np.sum((x - x0) ** 2))
        if denom <= 0:
            continue
        a = float(np.sum((x - x0) * (S - y0)) / denom)
        b = y0 - a * x0
        S_hat = a * x + b

        ss_res = float(np.sum((S - S_hat) ** 2))
        ss_tot = float(np.sum((S - y0) ** 2))
        r2_vals[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    if not np.any(np.isfinite(r2_vals)):
        res = ExtensivityResult(float("nan"), float("nan"), q_grid, r2_vals)
        return res, {"q_ext_hat": res.q_opt, "q_ext_r2": res.r2_opt}

    idx = int(np.nanargmax(r2_vals))
    res = ExtensivityResult(
        q_opt=float(q_grid[idx]),
        r2_opt=float(r2_vals[idx]),
        q_grid=q_grid,
        r2=r2_vals,
    )
    return res, {"q_ext_hat": res.q_opt, "q_ext_r2": res.r2_opt}


def _sanity_test() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    res, d = estimate_q_extensivity(y)
    assert "q_ext_hat" in d and "q_ext_r2" in d


if __name__ == "__main__":
    _sanity_test()
    print("OK - tsallis_q_extensivity.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
