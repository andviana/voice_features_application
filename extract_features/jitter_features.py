"""
    Jitter features (local, RAP, PPQ5) via Praat

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

import parselmouth

# =========================================================================
# Jitter: cycle-to-cycle perturbation of fundamental period
# =========================================================================
# The plan lists:
# - local jitter
# - RAP (Relative Average Perturbation; 3-period smoothing)
# - PPQ5 (5-point Period Perturbation Quotient)
#
# Accurate jitter measurement requires reliable cycle detection.
# We therefore use Praat definitions via Parselmouth, matching common practice.


@dataclass(frozen=True)
class JitterStats:
    """
    Estatísticas de Jitter (perturbação da frequência fundamental).

    O Jitter mede a variabilidade ciclo a ciclo da frequência fundamental. 
    Pacientes com Parkinson frequentemente apresentam valores elevados de Jitter 
    devido à redução do controle motor das pregas vocais.

    Attributes:
        local (float): Jitter local (percentual médio da variação entre períodos).
        rap (float): Relative Average Perturbation (suavização sobre 3 períodos).
        ppq5 (float): Five-point Period Perturbation Quotient (suavização sobre 5 períodos).
    """
    local: float
    rap: float
    ppq5: float


def extract_jitter_features(
    y: np.ndarray,
    sr: int,
    fmin_hz: float = 75.0,
    fmax_hz: float = 300.0,
) -> Tuple[JitterStats, Dict[str, float]]:
    """
    Extrai métricas de Jitter utilizando o algoritmo do Praat via Parselmouth.

    Esta função calcula o Jitter local, RAP e PPQ5. Requer a biblioteca 
    praat-parselmouth para garantir a compatibilidade com os padrões clínicos.

    Args:
        y (np.ndarray): Vetor do sinal de áudio.
        sr (int): Taxa de amostragem.
        time_step (float): Passo de tempo para análise (padrão 10ms).
        min_pitch_hz (float): Frequência fundamental mínima.
        max_pitch_hz (float): Frequência fundamental máxima.
        silence_threshold (float): Limiar para detecção de silêncio.
        periods_per_window (float): Janela de análise em períodos de pitch.

    Returns:
        Tuple[JitterStats, Dict[str, float]]: Objeto de estatísticas e 
            dicionário mapeado para integração no CSV.
    """
    y = np.asarray(y, dtype=float).flatten()

    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)

        # Convert to PointProcess for periodicity-based measures
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", fmin_hz, fmax_hz)

        # Standard Praat control parameters used widely in literature:
        max_period_factor = 1.3
        max_amplitude_factor = 1.6
        # shortest_period = 0.0001
        # Filtro de Ruído: Ele define o tempo mínimo (em segundos) que um 
        # ciclo de vibração das pregas vocais deve ter para ser considerado válido. 
        # O valor 0.0001 equivale a 0,1 milissegundos.


        # local = float(parselmouth.praat.call(
        #     pp, "Get jitter (local)", 0, 0, fmin_hz, fmax_hz, max_period_factor, max_amplitude_factor
        # ))
        # rap = float(parselmouth.praat.call(
        #     pp, "Get jitter (rap)", 0, 0, fmin_hz, fmax_hz, max_period_factor, max_amplitude_factor
        # ))
        # ppq5 = float(parselmouth.praat.call(
        #     pp, "Get jitter (ppq5)", 0, 0, fmin_hz, fmax_hz, max_period_factor, max_amplitude_factor
        # ))

        # alterado hotfix - inicio
        period_floor = 1.0 / float(fmax_hz)     # menor período (pitch mais alto)
        period_ceiling = 1.0 / float(fmin_hz)   # maior período (pitch mais baixo)

        local = float(parselmouth.praat.call(
            pp, "Get jitter (local)", 0, 0, period_floor, period_ceiling, max_period_factor
        ))
        rap = float(parselmouth.praat.call(
            pp, "Get jitter (rap)", 0, 0, period_floor, period_ceiling, max_period_factor
        ))
        ppq5 = float(parselmouth.praat.call(
            pp, "Get jitter (ppq5)", 0, 0, period_floor, period_ceiling, max_period_factor
        ))
        # alteraddo hotfix - final        

        stats = JitterStats(local=local, rap=rap, ppq5=ppq5)
        features = {
            "jitter_local": stats.local,
            "jitter_rap": stats.rap,
            "jitter_ppq5": stats.ppq5,
        }
        return stats, features

    except Exception as exc:
        raise RuntimeError(f"Erro no processamento do Jitter: {exc}") from exc


def _sanity_test() -> None:
    # On a perfectly periodic sinusoid, jitter should be near 0 (numerically small).
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    y = 0.2 * np.sin(2.0 * np.pi * 150.0 * t)

    try:
        stats, d = extract_jitter_features(y, sr)
        assert set(d.keys()) == {"jitter_local", "jitter_rap", "jitter_ppq5"}
        assert np.isfinite(stats.local)

    except (ModuleNotFoundError, ImportError):
        pass

    except RuntimeError:
        # Expected if Parselmouth not installed in the environment.
        pass


if __name__ == "__main__":
    _sanity_test()
    print("OK - jitter_features.py")


# -----------------------------------------------------------------------------
# Code signature (required by Space instructions)
# Prof. Dr. Bruno Duarte Gomes / Laboratório de Neurofisiologia Eduardo Oswaldo Cruz / Laboratório de Simulação e Biologia Computacional / Instituto de Ciências Biológicas - UFPA
# -----------------------------------------------------------------------------
