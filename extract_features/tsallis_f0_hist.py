"""
    Tsallis entropy from instantaneous F0 histogram (Method 2)

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
    - librosa OR praat-parselmouth

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


def _estimate_f0_track(
    y: np.ndarray,
    sr: int,
    fmin_hz: float,
    fmax_hz: float,
    frame_length_ms: float,
    hop_length_ms: float,
) -> np.ndarray:
    """Prefer Praat if available, else librosa.yin."""
    y = np.asarray(y, dtype=float).flatten()

    hop = max(1, int(round(hop_length_ms * 1e-3 * sr)))
    frame_length = max(256, int(round(frame_length_ms * 1e-3 * sr)))

    try:
        import parselmouth  # type: ignore

        snd = parselmouth.Sound(y, sampling_frequency=sr)
        time_step = hop / float(sr)
        pitch = snd.to_pitch(time_step=time_step, pitch_floor=fmin_hz, pitch_ceiling=fmax_hz)
        f0 = pitch.selected_array["frequency"].astype(float)
        f0 = np.where(f0 > 0.0, f0, np.nan)
        return f0

    except Exception:
        import librosa

        f0 = librosa.yin(
            y,
            sr=sr,
            fmin=fmin_hz,
            fmax=fmax_hz,
            frame_length=frame_length,
            hop_length=hop,
        ).astype(float)
        return f0


def f0_histogram_distribution(
    y: np.ndarray,
    sr: int,
    fmin_hz: float = 75.0,
    fmax_hz: float = 300.0,
    n_bins: int = 50,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
) -> np.ndarray:
    f0 = _estimate_f0_track(
        y=y,
        sr=sr,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms,
    )
    f0 = np.asarray(f0, dtype=float)
    f0 = f0[np.isfinite(f0)]

    edges = np.linspace(fmin_hz, fmax_hz, n_bins + 1)
    if f0.size == 0:
        return np.zeros((n_bins,), dtype=float)

    hist, _ = np.histogram(f0, bins=edges)
    p = hist.astype(float)
    s = float(np.sum(p))
    return p / s if s > 0 else p


@dataclass(frozen=True)
class TsallisF0Stats:
    sq: float
    shannon: float


def extract_tsallis_f0_features(
    y: np.ndarray,
    sr: int,
    q: float = 1.3,
    fmin_hz: float = 75.0,
    fmax_hz: float = 300.0,
    n_bins: int = 50,
) -> Tuple[TsallisF0Stats, Dict[str, float]]:
    p = f0_histogram_distribution(
        y=y,
        sr=sr,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        n_bins=n_bins,
    )
    sq = tsallis_entropy(p, q=q)
    s1 = tsallis_entropy(p, q=1.0)

    stats = TsallisF0Stats(sq=float(sq), shannon=float(s1))
    features = {"tsallis_sq_f0": stats.sq, "shannon_s1_f0": stats.shannon}
    return stats, features


def _sanity_test() -> None:
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    y = 0.2 * np.sin(2.0 * np.pi * 150.0 * t)

    stats, d = extract_tsallis_f0_features(y, sr, q=1.3)
    assert set(d.keys()) == {"tsallis_sq_f0", "shannon_s1_f0"}


if __name__ == "__main__":
    _sanity_test()
    print("OK - tsallis_f0_hist.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
