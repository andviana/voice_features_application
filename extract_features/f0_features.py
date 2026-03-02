
"""
    F0 (Fundamental Frequency) Features

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
    - librosa (recommended)
    - praat-parselmouth (optional; preferred if installed)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np



# =========================================================================
# Why F0 matters in PD voice analysis
# =========================================================================
# In sustained vowel phonation, PD-related dysphonia can manifest as altered
# pitch stability and control. Summary statistics of F0 are standard baseline
# features in many voice-PD pipelines and are explicitly listed in the plan.


@dataclass(frozen=True)
class F0Stats:
    """Summary statistics for the F0 track."""

    mean_hz: float
    std_hz: float
    min_hz: float
    max_hz: float
    cv: float  # coefficient of variation = std/mean


def _safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    """mean/std/min/max that tolerates NaNs."""
    x = np.asarray(x, dtype=float).flatten()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1) if x.size > 1 else 0.0)
    return mean, std, float(np.min(x)), float(np.max(x))


def estimate_f0_track(
    y: np.ndarray,
    sr: int,
    fmin_hz: float = 75.0,
    fmax_hz: float = 300.0,
    frame_length: int = 2048,
    hop_length: int = 256,
    prefer_parselmouth: bool = True,
    librosa_method: str = "yin",
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate an F0 track in Hz, with frame-center times.

    Strategy
    --------
    1) If praat-parselmouth is available and prefer_parselmouth=True:
        use Praat pitch tracking (aligned with clinical voice metrics).
    2) Otherwise: use librosa (yin or pyin).

    Returns
    -------
    f0_hz: np.ndarray
        Shape (n_frames,), NaN for unvoiced frames (Praat gives 0 Hz).
    times_s: np.ndarray
        Shape (n_frames,), frame-center times in seconds.
    """

    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # ---- Option 1: Praat/Parselmouth ----
    if prefer_parselmouth:
        try:
            import parselmouth  # type: ignore

            snd = parselmouth.Sound(y, sampling_frequency=sr)
            time_step = hop_length / float(sr)
            pitch = snd.to_pitch(
                time_step=time_step,
                pitch_floor=fmin_hz,
                pitch_ceiling=fmax_hz,
            )
            f0 = pitch.selected_array["frequency"].astype(float)
            f0 = np.where(f0 > 0.0, f0, np.nan)  # Praat: 0 => unvoiced
            times = pitch.xs().astype(float)
            return f0, times
        except Exception:
            # Fall back to librosa
            pass

    # ---- Option 2: librosa ----
    try:
        import librosa

        if librosa_method.lower() == "pyin":
            f0, _, _ = librosa.pyin(
                y,
                fmin=fmin_hz,
                fmax=fmax_hz,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            f0 = np.asarray(f0, dtype=float)
        else:
            f0 = librosa.yin(
                y,
                fmin=fmin_hz,
                fmax=fmax_hz,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            ).astype(float)

        times = librosa.frames_to_time(
            np.arange(f0.size),
            sr=sr,
            hop_length=hop_length,
        ).astype(float)
        return f0, times

    except Exception as exc:
        raise RuntimeError(
            "F0 estimation failed. Install librosa or praat-parselmouth:   pip install librosa praat-parselmouth"
        ) from exc


def extract_f0_features(
    y: np.ndarray,
    sr: int,
    fmin_hz: float = 75.0,
    fmax_hz: float = 300.0,
    **kwargs,
) -> Tuple[F0Stats, Dict[str, float]]:
    """Compute baseline F0 summary features."""

    f0, _ = estimate_f0_track(
        y=y,
        sr=sr,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        **kwargs,
    )
    mu, sd, mn, mx = _safe_stats(f0)
    cv = float(sd / mu) if np.isfinite(mu) and abs(mu) > 0 else float("nan")

    stats = F0Stats(mean_hz=mu, std_hz=sd, min_hz=mn, max_hz=mx, cv=cv)
    features = {
        "f0_mean_hz": stats.mean_hz,
        "f0_std_hz": stats.std_hz,
        "f0_min_hz": stats.min_hz,
        "f0_max_hz": stats.max_hz,
        "f0_cv": stats.cv,
    }
    return stats, features


def _sanity_test() -> None:
    """Basic check on a clean sinusoid."""
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    y = 0.2 * np.sin(2.0 * np.pi * 150.0 * t)

    stats, d = extract_f0_features(y, sr, prefer_parselmouth=False)
    assert set(d.keys()) == {
        "f0_mean_hz", "f0_std_hz", "f0_min_hz", "f0_max_hz", "f0_cv"
    }
    assert np.isfinite(stats.mean_hz)
    assert abs(stats.mean_hz - 150.0) < 10.0


if __name__ == "__main__":
    _sanity_test()
    print("OK - f0_features.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
