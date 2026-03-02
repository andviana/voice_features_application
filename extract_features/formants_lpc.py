"""
    Formants F1–F4 via LPC (order 12)

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


# =========================================================================
# Formants via LPC
# =========================================================================
# The plan specifies LPC order 12, 25 ms frames, 10 ms hop, and reporting
# mean/std for each formant (F1–F4).
#
# Important: Formant estimation is best for voiced speech with rich harmonic
# content; on near-sinusoidal signals it can become ill-posed. We therefore
# return NaNs gracefully if estimation fails.


@dataclass(frozen=True)
class FormantStats:
    """
    Estatísticas resumidas dos formantes F1 a F4 extraídos via LPC.

    Esta dataclass armazena as tendências centrais (médias) e a variabilidade 
    (desvios padrão) das frequências de ressonância do trato vocal, que são 
    indicadores cruciais da configuração articulatória em pacientes com Parkinson.

    Attributes:
        f1_mean_hz (float): Média da primeira frequência formante (F1). 
            Geralmente correlacionada com a abertura da mandíbula.
        f1_std_hz (float): Desvio padrão da primeira frequência formante.
        f2_mean_hz (float): Média da segunda frequência formante (F2). 
            Geralmente correlacionada com o avanço ou recuo da língua.
        f2_std_hz (float): Desvio padrão da segunda frequência formante.
        f3_mean_hz (float): Média da terceira frequência formante (F3). 
            Relacionada a cavidades menores e arredondamento labial.
        f3_std_hz (float): Desvio padrão da terceira frequência formante.
        f4_mean_hz (float): Média da quarta frequência formante (F4). 
            Frequência de ressonância superior, muitas vezes ligada ao timbre.
        f4_std_hz (float): Desvio padrão da quarta frequência formante.
    """
    f1_mean_hz: float
    f1_std_hz: float
    f2_mean_hz: float
    f2_std_hz: float
    f3_mean_hz: float
    f3_std_hz: float
    f4_mean_hz: float
    f4_std_hz: float


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Return frames with shape (n_frames, frame_length)."""
    y = np.asarray(y, dtype=float).flatten()
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    n_frames = 1 + (y.size - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, frame_length),
        strides=(y.strides[0] * hop_length, y.strides[0]),
        writeable=False,
    ).copy()
    return frames


def _lpc_levinson_durbin(x: np.ndarray, order: int) -> np.ndarray:
    """LPC via autocorrelation + Levinson-Durbin recursion.

    Returns a LPC polynomial A(z) with a[0]=1.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    # Biased autocorrelation for stability
    r_full = np.correlate(x, x, mode="full")
    mid = r_full.size // 2
    r = r_full[mid : mid + order + 1].astype(float)

    a = np.zeros(order + 1, dtype=float)
    a[0] = 1.0
    e = float(r[0])

    if e <= 0.0:
        return a

    for i in range(1, order + 1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -(r[i] + acc) / e

        a_prev = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]

        e *= 1.0 - k * k
        if e <= 1e-12:
            break

    return a


def _formants_from_lpc(a: np.ndarray, sr: int) -> np.ndarray:
    """Extract formant frequencies from LPC roots (Hz).

    Heuristics:
    - Keep roots with Im>=0 (avoid duplicates).
    - Convert angle to frequency.
    - Filter to speech-relevant range and reasonable bandwidths.
    """
    a = np.asarray(a, dtype=float).flatten()
    if a.size < 2:
        return np.array([], dtype=float)

    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]

    angles = np.angle(roots)
    freqs = angles * (sr / (2.0 * np.pi))

    mags = np.abs(roots)
    bandwidths = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.clip(mags, 1e-8, 1.0))

    valid = (freqs > 80.0) & (freqs < 5000.0) & (bandwidths < 400.0)
    freqs = np.sort(freqs[valid])
    return freqs.astype(float)


def extract_formant_features(
    y: np.ndarray,
    sr: int,
    lpc_order: int = 12,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    pre_emphasis: float = 0.97,
) -> Tuple[FormantStats, Dict[str, float]]:
    """Compute mean/std of F1–F4 across frames."""
    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        stats = FormantStats(*([float("nan")] * 8))
        return stats, _as_dict(stats)

    # Pre-emphasis: y[t] - a*y[t-1]
    y_pe = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    frame_length = max(32, int(round(frame_length_ms * 1e-3 * sr)))
    hop_length = max(1, int(round(hop_length_ms * 1e-3 * sr)))

    frames = _frame_signal(y_pe, frame_length=frame_length, hop_length=hop_length)
    window = np.hamming(frame_length).astype(float)

    f1, f2, f3, f4 = [], [], [], []
    for fr in frames:
        frw = fr * window
        a = _lpc_levinson_durbin(frw, order=lpc_order)
        formants = _formants_from_lpc(a, sr=sr)

        if formants.size >= 1:
            f1.append(formants[0])
        if formants.size >= 2:
            f2.append(formants[1])
        if formants.size >= 3:
            f3.append(formants[2])
        if formants.size >= 4:
            f4.append(formants[3])

    def _ms(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan"), float("nan")
        mean = float(np.mean(v))
        std = float(np.std(v, ddof=1) if v.size > 1 else 0.0)
        return mean, std

    f1m, f1s = _ms(f1)
    f2m, f2s = _ms(f2)
    f3m, f3s = _ms(f3)
    f4m, f4s = _ms(f4)

    stats = FormantStats(f1m, f1s, f2m, f2s, f3m, f3s, f4m, f4s)
    return stats, _as_dict(stats)


def _as_dict(stats: FormantStats) -> Dict[str, float]:
    return {
        "f1_mean_hz": stats.f1_mean_hz,
        "f1_std_hz": stats.f1_std_hz,
        "f2_mean_hz": stats.f2_mean_hz,
        "f2_std_hz": stats.f2_std_hz,
        "f3_mean_hz": stats.f3_mean_hz,
        "f3_std_hz": stats.f3_std_hz,
        "f4_mean_hz": stats.f4_mean_hz,
        "f4_std_hz": stats.f4_std_hz,
    }


def _sanity_test() -> None:
    sr = 16000
    t = np.arange(int(1.0 * sr)) / sr
    y = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)

    stats, d = extract_formant_features(y, sr)
    assert set(d.keys()) == {
        "f1_mean_hz", "f1_std_hz", "f2_mean_hz", "f2_std_hz",
        "f3_mean_hz", "f3_std_hz", "f4_mean_hz", "f4_std_hz",
    }


if __name__ == "__main__":
    _sanity_test()
    print("OK - formants_lpc.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
