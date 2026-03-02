"""
    MFCC (13) + Delta MFCC stats

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
    - librosa

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MFCCStats:
    """Packed MFCC summary vector (for compact downstream use)."""
    features: np.ndarray  # shape (4*n_mfcc,)


def extract_mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    fmin: float = 80.0,
    fmax: float | None = 8000.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
) -> Tuple[MFCCStats, Dict[str, float]]:
    """Compute MFCC features and return (stats, feature_dict).

    Feature dict keys
    -----------------
    mfcc{i}_mean, mfcc{i}_std, dmfcc{i}_mean, dmfcc{i}_std for i=1..n_mfcc
    """
    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        feats = np.full((4 * n_mfcc,), np.nan, dtype=float)
        stats = MFCCStats(features=feats)
        return stats, _as_dict(stats, n_mfcc=n_mfcc)

    try:
        import librosa

        n_fft = max(256, int(round(frame_length_ms * 1e-3 * sr)))
        hop = max(1, int(round(hop_length_ms * 1e-3 * sr)))

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop,
            fmin=fmin,
            fmax=fmax,
        )  # (n_mfcc, n_frames)

        dmfcc = librosa.feature.delta(mfcc, width=9, order=1)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1, ddof=1) if mfcc.shape[1] > 1 else np.zeros(n_mfcc)
        dmfcc_mean = np.mean(dmfcc, axis=1)
        dmfcc_std = np.std(dmfcc, axis=1, ddof=1) if dmfcc.shape[1] > 1 else np.zeros(n_mfcc)

        feats = np.concatenate([mfcc_mean, mfcc_std, dmfcc_mean, dmfcc_std]).astype(float)
        stats = MFCCStats(features=feats)
        return stats, _as_dict(stats, n_mfcc=n_mfcc)

    except Exception as exc:
        raise RuntimeError("MFCC extraction requires librosa: pip install librosa") from exc


def _as_dict(stats: MFCCStats, n_mfcc: int) -> Dict[str, float]:
    v = np.asarray(stats.features, dtype=float).flatten()
    if v.size != 4 * n_mfcc:
        raise ValueError("Unexpected MFCC vector length.")

    out: Dict[str, float] = {}

    a = v[0:n_mfcc]
    b = v[n_mfcc:2 * n_mfcc]
    c = v[2 * n_mfcc:3 * n_mfcc]
    d = v[3 * n_mfcc:4 * n_mfcc]

    for i in range(n_mfcc):
        out[f"mfcc{i+1}_mean"] = float(a[i])
        out[f"mfcc{i+1}_std"] = float(b[i])
        out[f"dmfcc{i+1}_mean"] = float(c[i])
        out[f"dmfcc{i+1}_std"] = float(d[i])

    return out


def _sanity_test() -> None:
    sr = 16000
    t = np.arange(int(1.0 * sr)) / sr
    y = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)

    stats, d = extract_mfcc_features(y, sr, n_mfcc=13)
    assert len(d) == 13 * 4


if __name__ == "__main__":
    _sanity_test()
    print("OK - mfcc_features.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
