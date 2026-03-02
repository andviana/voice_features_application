"""
    Spectral features (centroid, rolloff, flux, subband energy)

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
from scipy.signal import get_window


@dataclass(frozen=True)
class SpectralStats:
    centroid_mean_hz: float
    centroid_std_hz: float
    rolloff_mean_hz: float
    rolloff_std_hz: float
    flux_mean: float
    flux_std: float
    energy_low_mean: float
    energy_mid_mean: float
    energy_high_mean: float


def _stft_mag(y: np.ndarray, sr: int, n_fft: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (magnitude, freqs). magnitude shape: (n_freq, n_frames)."""
    y = np.asarray(y, dtype=float).flatten()
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    win = get_window("hann", n_fft, fftbins=True).astype(float)
    n_frames = 1 + (y.size - n_fft) // hop

    mags = []
    for i in range(n_frames):
        start = i * hop
        fr = y[start : start + n_fft] * win
        spec = np.fft.rfft(fr)
        mags.append(np.abs(spec))

    mag = np.stack(mags, axis=1).astype(float)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(float)
    return mag, freqs


def extract_spectral_features(
    y: np.ndarray,
    sr: int,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    rolloff_percent: float = 0.85,
    low_band: Tuple[float, float] = (80.0, 500.0),
    mid_band: Tuple[float, float] = (500.0, 2000.0),
    high_band: Tuple[float, float] = (2000.0, 8000.0),
) -> Tuple[SpectralStats, Dict[str, float]]:
    y = np.asarray(y, dtype=float).flatten()
    if y.size == 0:
        stats = SpectralStats(*([float("nan")] * 9))
        return stats, _as_dict(stats)

    n_fft = max(256, int(round(frame_length_ms * 1e-3 * sr)))
    hop = max(1, int(round(hop_length_ms * 1e-3 * sr)))

    mag, freqs = _stft_mag(y, sr, n_fft=n_fft, hop=hop)

    eps = 1e-12
    mag_sum = np.sum(mag, axis=0, keepdims=True) + eps
    p = mag / mag_sum  # per-frame normalized magnitude

    # Spectral centroid
    centroid = np.sum(p * freqs[:, None], axis=0)

    # Spectral rolloff
    csum = np.cumsum(mag, axis=0)
    thresh = rolloff_percent * csum[-1, :]
    idx = np.array([int(np.searchsorted(csum[:, i], thresh[i])) for i in range(mag.shape[1])])
    idx = np.clip(idx, 0, freqs.size - 1)
    rolloff = freqs[idx]

    # Spectral flux (normalized spectrum difference)
    p2 = p / (np.linalg.norm(p, axis=0, keepdims=True) + eps)
    dp = np.diff(p2, axis=1)
    flux = np.sqrt(np.sum(dp * dp, axis=0))  # length n_frames-1

    def _band_energy(band):
        f0, f1 = band
        sel = (freqs >= f0) & (freqs < f1)
        if not np.any(sel):
            return np.full((mag.shape[1],), np.nan, dtype=float)
        return np.mean(mag[sel, :] ** 2, axis=0)

    e_low = _band_energy(low_band)
    e_mid = _band_energy(mid_band)
    e_high = _band_energy(high_band)

    def _ms(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0)

    c_m, c_s = _ms(centroid)
    r_m, r_s = _ms(rolloff)
    f_m, f_s = _ms(flux)

    stats = SpectralStats(
        centroid_mean_hz=c_m,
        centroid_std_hz=c_s,
        rolloff_mean_hz=r_m,
        rolloff_std_hz=r_s,
        flux_mean=f_m,
        flux_std=f_s,
        energy_low_mean=float(np.nanmean(e_low)),
        energy_mid_mean=float(np.nanmean(e_mid)),
        energy_high_mean=float(np.nanmean(e_high)),
    )
    return stats, _as_dict(stats)


def _as_dict(stats: SpectralStats) -> Dict[str, float]:
    return {
        "spec_centroid_mean_hz": stats.centroid_mean_hz,
        "spec_centroid_std_hz": stats.centroid_std_hz,
        "spec_rolloff_mean_hz": stats.rolloff_mean_hz,
        "spec_rolloff_std_hz": stats.rolloff_std_hz,
        "spec_flux_mean": stats.flux_mean,
        "spec_flux_std": stats.flux_std,
        "spec_energy_low_mean": stats.energy_low_mean,
        "spec_energy_mid_mean": stats.energy_mid_mean,
        "spec_energy_high_mean": stats.energy_high_mean,
    }


def _sanity_test() -> None:
    rng = np.random.default_rng(0)
    sr = 16000
    y = 0.05 * rng.standard_normal(int(1.0 * sr))

    stats, d = extract_spectral_features(y, sr)
    assert "spec_centroid_mean_hz" in d


if __name__ == "__main__":
    _sanity_test()
    print("OK - spectral_features.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
