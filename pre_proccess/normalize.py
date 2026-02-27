import librosa
import numpy as np

def resample_audio(y, sr, target_sr=44100):
    if sr != target_sr:
        return librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y

def scale_amplitude(y, target_db=-1.0):
    """Normaliza o pico para evitar clipping (como em Iyer et al.)."""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

def sanity_test():
    y = np.array([0, 2.0, -2.0])
    y_scaled = scale_amplitude(y)
    assert np.max(y_scaled) == 1.0
    print("OK - normalize.py")

if __name__ == "__main__":
    sanity_test()