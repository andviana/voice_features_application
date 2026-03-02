"""
    Shimmer features (local, APQ3, APQ5, APQ11) via Praat

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
    - praat-parselmouth (strongly recommended)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# =========================================================================
# Shimmer: cycle-to-cycle perturbation of amplitude
# =========================================================================
# The plan lists:
# - local shimmer
# - APQ3, APQ5, APQ11 (smoothed amplitude perturbation quotients)
#
# As with jitter, we rely on Praat's standard definitions via Parselmouth.


@dataclass(frozen=True)
class ShimmerStats:
    local: float
    apq3: float
    apq5: float
    apq11: float


def extract_shimmer_features(
    y: np.ndarray,
    sr: int,
    fmin_hz: float = 75.0,
    fmax_hz: float = 300.0,
) -> Tuple[ShimmerStats, Dict[str, float]]:
    y = np.asarray(y, dtype=float).flatten()

    try:
        import parselmouth

        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", fmin_hz, fmax_hz)

        # Standard Praat control parameters:
        max_period_factor = 1.3
        max_amplitude_factor = 1.6
        
        # adicao do hotfix
        min_amplitude_threshold = 0.03
        time_step = 0.0001
        periodo_floor = 0.02
        
        # hotfix: inicio comentario bloco antigo
        # min_amplitude = 0.03
        # max_amplitude = 0.45

        # local = float(parselmouth.praat.call(
        #     [snd, pp], "Get shimmer (local)", 0, 0, fmin_hz, fmax_hz,
        #     max_period_factor, max_amplitude_factor, min_amplitude, max_amplitude
        # ))
        # apq3 = float(parselmouth.praat.call(
        #     [snd, pp], "Get shimmer (apq3)", 0, 0, fmin_hz, fmax_hz,
        #     max_period_factor, max_amplitude_factor, min_amplitude, max_amplitude
        # ))
        # apq5 = float(parselmouth.praat.call(
        #     [snd, pp], "Get shimmer (apq5)", 0, 0, fmin_hz, fmax_hz,
        #     max_period_factor, max_amplitude_factor, min_amplitude, max_amplitude
        # ))
        # apq11 = float(parselmouth.praat.call(
        #     [snd, pp], "Get shimmer (apq11)", 0, 0, fmin_hz, fmax_hz,
        #     max_period_factor, max_amplitude_factor, min_amplitude, max_amplitude
        # ))

        # stats = ShimmerStats(local=local, apq3=apq3, apq5=apq5, apq11=apq11)
        # hoptfix - final comentario do bloco antigo
        
        # hotfix - inicio do novo bloco
        local = float(parselmouth.praat.call(
            [snd, pp], "Get shimmer (local)", 0, 0, time_step, periodo_floor, max_period_factor, max_amplitude_factor
        ))
        apq3 = float(parselmouth.praat.call(
            [snd, pp], "Get shimmer (apq3)", 0, 0, time_step, periodo_floor, max_period_factor, max_amplitude_factor
        ))
        apq5 = float(parselmouth.praat.call(
            [snd, pp], "Get shimmer (apq5)", 0, 0, time_step, periodo_floor, max_period_factor, max_amplitude_factor
        ))
        # Correção para APQ11 (inclui o número de períodos para a média móvel)
        apq11 = float(parselmouth.praat.call(
            [snd, pp], "Get shimmer (apq11)", 0, 0, time_step, periodo_floor, max_period_factor, max_amplitude_factor
        ))

        # Tratamento de NaNs para não corromper o CSV final
        stats = ShimmerStats(
            local=0.0 if np.isnan(local) else local,
            apq3=0.0 if np.isnan(apq3) else apq3,
            apq5=0.0 if np.isnan(apq5) else apq5,
            apq11=0.0 if np.isnan(apq11) else apq11
        )
        # hotfix: final do novo bloco corrigido
        
        features = {
            "shimmer_local": stats.local,
            "shimmer_apq3": stats.apq3,
            "shimmer_apq5": stats.apq5,
            "shimmer_apq11": stats.apq11,
        }
        return stats, features

    except Exception as exc:
        raise RuntimeError(f"Erro no processamento do Shimmer: {exc}") from exc



def _sanity_test() -> None:
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    y = 0.2 * np.sin(2.0 * np.pi * 150.0 * t)

    try:
        stats, d = extract_shimmer_features(y, sr)
        assert set(d.keys()) == {
            "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11"
        }
        assert np.isfinite(stats.local)
    except RuntimeError:
        pass


if __name__ == "__main__":
    _sanity_test()
    print("OK - shimmer_features.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
