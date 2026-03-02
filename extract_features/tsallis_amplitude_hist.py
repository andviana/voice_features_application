"""
Tsallis entropy from amplitude histogram (Method 1)

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


# =========================================================================
# Tsallis entropy on amplitude histogram (Method 1 in the plan)
# =========================================================================
# The plan defines:
# 1) z-score normalize amplitude
# 2) discretize in B bins over [-5σ, +5σ] (default clip_sigma=5)
# 3) compute p_i as relative frequencies
# 4) compute Sq = (1 - sum p_i^q) / (q - 1), with Shannon limit at q->1


def tsallis_entropy(p: np.ndarray, q: float) -> float:
    p = np.asarray(p, dtype=float).flatten()
    p = p[np.isfinite(p)]
    p = p[p > 0.0]
    if p.size == 0:
        return float("nan")

    if q == 1.0:
        return float(-np.sum(p * np.log(p)))

    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


def amplitude_histogram_distribution(
    y: np.ndarray,
    n_bins: int = 100,
    clip_sigma: float = 5.0,
) -> np.ndarray:
    """Return p (shape: (n_bins,)) from the standardized amplitude histogram."""
    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        return np.array([], dtype=float)

    mu = float(np.mean(y))
    sd = float(np.std(y, ddof=1) if y.size > 1 else 0.0)

    if sd <= 0.0:
        # Constant signal => degenerate distribution
        p = np.zeros((n_bins,), dtype=float)
        p[n_bins // 2] = 1.0
        return p

    z = (y - mu) / sd
    z = np.clip(z, -clip_sigma, clip_sigma)

    edges = np.linspace(-clip_sigma, clip_sigma, n_bins + 1)
    hist, _ = np.histogram(z, bins=edges)
    p = hist.astype(float)
    s = float(np.sum(p))
    return p / s if s > 0 else p


@dataclass(frozen=True)
class TsallisAmpStats:
    sq: float
    shannon: float


def extract_tsallis_amplitude_features(
    y: np.ndarray,
    q: float = 1.3,
    n_bins: int = 100,
    clip_sigma: float = 5.0,
) -> Tuple[TsallisAmpStats, Dict[str, float]]:
    p = amplitude_histogram_distribution(y, n_bins=n_bins, clip_sigma=clip_sigma)
    sq = tsallis_entropy(p, q=q)
    s1 = tsallis_entropy(p, q=1.0)

    stats = TsallisAmpStats(sq=float(sq), shannon=float(s1))
    features = {"tsallis_sq_amp": stats.sq, "shannon_s1_amp": stats.shannon}
    return stats, features


def _sanity_test() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(10000)

    stats, d = extract_tsallis_amplitude_features(y, q=1.3)
    assert set(d.keys()) == {"tsallis_sq_amp", "shannon_s1_amp"}
    assert np.isfinite(d["shannon_s1_amp"])


if __name__ == "__main__":
    _sanity_test()
    print("OK - tsallis_amplitude_hist.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
