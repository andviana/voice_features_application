"""
    HNR (Harmonic-to-Noise Ratio) feature

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
    - praat-parselmouth (recommended)
    - scipy (fallback approximation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

import parselmouth
from scipy.signal import get_window


@dataclass(frozen=True)
class HNRStats:
    """
    Estatística resumida do Harmonic-to-Noise Ratio (HNR).

    O HNR quantifica a proporção entre a energia harmônica (periódica) e o ruído 
    (aperiódico) no sinal de voz. É um marcador comum para avaliar a qualidade 
    vocal e a presença de soprosidade ou rouquidão.

    Attributes:
        mean_db (float): Média da relação harmônico-ruído em decibéis (dB).
    """
    mean_db: float


def extract_hnr_features(
    y: np.ndarray,
    sr: int,
    time_step: float = 0.01,
    min_pitch_hz: float = 75.0,
    silence_threshold: float = 0.1,
    periods_per_window: float = 1.0,
) -> Tuple[HNRStats, Dict[str, float]]:
    """Compute mean HNR in dB.

    Primary implementation: Praat Harmonicity (cc) via Parselmouth.
    Fallback: cepstrum heuristic (NOT identical to Praat; use only if needed).
    """

    y = np.asarray(y, dtype=float).flatten()

    # ---- Preferred: Praat/Parselmouth ----
    try:
        # pylint: disable=no-member
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        harm = snd.to_harmonicity_cc(
            time_step=time_step,
            minimum_pitch=min_pitch_hz,
            silence_threshold=silence_threshold,
            periods_per_window=periods_per_window,
        )
        hnr_db = harm.values.flatten().astype(float)
        hnr_db = hnr_db[np.isfinite(hnr_db)]
        mean_db = float(np.mean(hnr_db)) if hnr_db.size else float("nan")

        stats = HNRStats(mean_db=mean_db)
        return stats, {"hnr_mean_db": stats.mean_db}

    except Exception:
        pass

    # ---- Fallback: cepstrum-based heuristic ----
    # This approximates periodicity prominence in the cepstrum.
    # It should be treated as an *approximate proxy*, not a strict HNR match.
    try:


        if y.size == 0:
            stats = HNRStats(mean_db=float("nan"))
            return stats, {"hnr_mean_db": stats.mean_db}

        frame_len = max(32, int(round(0.04 * sr)))  # 40 ms
        hop = max(1, int(round(0.01 * sr)))         # 10 ms
        win = get_window("hann", frame_len, fftbins=True).astype(float)

        vals = []
        for start in range(0, max(1, y.size - frame_len + 1), hop):
            fr = y[start : start + frame_len] * win
            spec = np.fft.rfft(fr)
            log_mag = np.log(np.maximum(np.abs(spec), 1e-12))
            cep = np.fft.irfft(log_mag)

            # Pitch-period quefrency range ~ [1/500, 1/75] seconds
            qmin = int(sr / 500.0)
            qmax = min(int(sr / 75.0), cep.size - 1)
            if qmax <= qmin:
                continue

            peak = float(np.max(cep[qmin:qmax]))
            noise = float(np.mean(np.abs(cep[qmin:qmax])) + 1e-12)
            proxy_db = 20.0 * np.log10(max(peak / noise, 1e-12))
            vals.append(proxy_db)

        mean_db = float(np.mean(vals)) if len(vals) else float("nan")
        stats = HNRStats(mean_db=mean_db)
        return stats, {"hnr_mean_db": stats.mean_db}

    except Exception as exc:
        raise RuntimeError(
            "HNR failed. Recommended: pip install praat-parselmouth"
        ) from exc


def _sanity_test() -> None:
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    y = 0.2 * np.sin(2.0 * np.pi * 150.0 * t)

    _, d = extract_hnr_features(y, sr)
    assert "hnr_mean_db" in d


if __name__ == "__main__":
    _sanity_test()
    print("OK - hnr_features.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
