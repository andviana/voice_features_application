# import librosa
import numpy as np


# def resample_audio(y, sr, target_sr=44100):
#     if sr != target_sr:
#         return librosa.resample(y, orig_sr=sr, target_sr=target_sr)
#     return y


def ensure_mono(y):
    """
    Converte para mono tirando a média dos canais (item 4.1.1).
    """
    if len(y.shape) > 1:
        return np.mean(y, axis=1)
    return y


def scale_amplitude(y, target_db=-1.0):
    """
    Normaliza para o intervalo [-1, 1] (item 4.1.5.
    Normaliza o pico para evitar clipping (como em Iyer et al.).
    """
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y


def sanity_test():
    """Valida conversão mono e normalização de amplitude."""
    # Teste Mono
    y_stereo = np.array([[1.0, 0.5], [-1.0, -0.5], [0.0, 0.0]])
    y_mono = ensure_mono(y_stereo)
    assert len(y_mono.shape) == 1
    assert y_mono[0] == 0.75 # Média de 1.0 e 0.5
    
    # Teste Escalonamento
    y_unscaled = np.array([0, 5.0, -2.0])
    y_scaled = scale_amplitude(y_unscaled)
    assert np.max(y_scaled) == 1.0
    assert np.min(y_scaled) == -0.4
    
    print("OK - normalize.py: Mono e Escalonamento validados.")

# todo: adicionar ensure_mono no sanity_test
if __name__ == "__main__":
    sanity_test()