import librosa
import numpy as np

def trim_and_segment(y, sr, duration=2.0):
    # Remove cliques de acionamento do microfone
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    
    # Seleção do segmento estável central
    n_samples = int(duration * sr)
    if len(y_trim) > n_samples:
        start = (len(y_trim) - n_samples) // 2
        y_trim = y_trim[start : start + n_samples]
    
    # Aplicação da Janela de Hamming
    return y_trim * np.hamming(len(y_trim))

def sanity_test():
    y = np.ones(88200)
    y_win = trim_and_segment(y, 44100)
    assert len(y_win) <= 88200
    print("OK - windowing.py")

if __name__ == "__main__":
    sanity_test()