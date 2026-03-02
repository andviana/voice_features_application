"""
    Tsallis q via q-Gaussian fit (histogram density)

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
from scipy.optimize import curve_fit


def q_gaussian(x: np.ndarray, A: float, beta: float, q: float) -> np.ndarray:
    """q-Gaussian density model used in nonextensive statistics.

    p(x) = A * [1 - (1-q)*beta*x^2]^(1/(1-q)), where bracket>=0.

    Implementation detail:
    - For q close to 1, we switch to a Gaussian-like form A*exp(-beta*x^2) for stability.
    """
    x = np.asarray(x, dtype=float)
    if abs(q - 1.0) < 1e-6:
        return A * np.exp(-beta * x**2)

    inside = 1.0 - (1.0 - q) * beta * x**2
    inside = np.maximum(inside, 0.0)
    return A * inside ** (1.0 / (1.0 - q))


@dataclass(frozen=True)
class QGaussianFit:
    q_hat: float
    beta_hat: float
    A_hat: float
    rmse: float


def estimate_q_from_amplitude_qgaussian(
    y: np.ndarray,
    n_bins: int = 100,
    clip_sigma: float = 5.0,
    q_bounds: Tuple[float, float] = (0.5, 2.5),
) -> Tuple[QGaussianFit, Dict[str, float]]:
    """Estimate q by fitting a q-Gaussian to the standardized amplitude histogram.

    Warning
    -------
    This is sensitive to binning/clipping and should be validated empirically on your dataset.
    """
    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        fit = QGaussianFit(float("nan"), float("nan"), float("nan"), float("nan"))
        return fit, _as_dict(fit)

    mu = float(np.mean(y))
    sd = float(np.std(y, ddof=1) if y.size > 1 else 0.0)
    if sd <= 0.0:
        fit = QGaussianFit(1.0, 1.0, 1.0, 0.0)
        return fit, _as_dict(fit)

    z = (y - mu) / sd
    z = np.clip(z, -clip_sigma, clip_sigma)

    edges = np.linspace(-clip_sigma, clip_sigma, n_bins + 1)
    hist, edges = np.histogram(z, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Initial guesses
    p0 = (float(np.max(hist) + 1e-12), 1.0, 1.2)
    lower = (0.0, 1e-6, q_bounds[0])
    upper = (np.inf, 1e3, q_bounds[1])

    try:
        popt, _ = curve_fit(
            q_gaussian,
            centers,
            hist,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        A_hat, beta_hat, q_hat = map(float, popt)
        pred = q_gaussian(centers, A_hat, beta_hat, q_hat)
        rmse = float(np.sqrt(np.mean((pred - hist) ** 2)))
        fit = QGaussianFit(q_hat=q_hat, beta_hat=beta_hat, A_hat=A_hat, rmse=rmse)
        return fit, _as_dict(fit)

    except Exception:
        fit = QGaussianFit(float("nan"), float("nan"), float("nan"), float("nan"))
        return fit, _as_dict(fit)


def _as_dict(fit: QGaussianFit) -> Dict[str, float]:
    return {
        "q_qgauss_hat": float(fit.q_hat),
        "q_qgauss_beta_hat": float(fit.beta_hat),
        "q_qgauss_A_hat": float(fit.A_hat),
        "q_qgauss_rmse": float(fit.rmse),
    }


def _sanity_test() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    fit, d = estimate_q_from_amplitude_qgaussian(y)
    assert set(d.keys()) == {
        "q_qgauss_hat", "q_qgauss_beta_hat", "q_qgauss_A_hat", "q_qgauss_rmse"
    }


if __name__ == "__main__":
    _sanity_test()
    print("OK - tsallis_q_qgaussian_fit.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
