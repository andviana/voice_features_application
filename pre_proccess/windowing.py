# import librosa
import numpy as np


# def trim_and_segment(y, sr, duration=2.0):
#     # Remove cliques de acionamento do microfone
#     y_trim, _ = librosa.effects.trim(y, top_db=25)
    
#     # Seleção do segmento estável central
#     n_samples = int(duration * sr)
#     if len(y_trim) > n_samples:
#         start = (len(y_trim) - n_samples) // 2
#         y_trim = y_trim[start : start + n_samples]
    
#     # Aplicação da Janela de Hamming
#     return y_trim * np.hamming(len(y_trim))


def remove_silence_adaptive(y, sr, frame_ms=25, hop_ms=10):
    """
    Implementa o item 4.1.4: Remoção de silêncio via limiar adaptativo.
    """
    frame_length = int(frame_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    
    # Cálculo da energia de curto prazo
    energy = np.array([
        np.sum(y[i:i+frame_length]**2)
        for i in range(0, len(y) - frame_length, hop_length)
    ])
    
    mu_e = np.mean(energy)
    sigma_e = np.std(energy)
    eth = mu_e - 2 * sigma_e
    
    # Máscara de frames acima do limiar
    keep_indices = np.where(energy > eth)[0]
    
    if len(keep_indices) == 0:
        return y # Fallback
    
    start_sample = keep_indices[0] * hop_length
    end_sample = min(keep_indices[-1] * hop_length + frame_length, len(y))
    
    return y[start_sample:end_sample]


def get_stable_segment(y, sr, duration=2.5):
    """
    Extrai a porção central (item 4.1.6).
    """
    n_samples = int(duration * sr)
    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        return y[start : start + n_samples]
    return y

def sanity_test():
    """Valida remoção de silêncio e segmentação."""
    sr = 1000
    # Sinal com silêncio (zeros) nas extremidades
    y = np.concatenate([np.zeros(200), np.ones(600), np.zeros(200)])
    
    # Teste Silêncio
    y_trimmed = remove_silence_adaptive(y, sr)
    assert len(y_trimmed) < len(y)
    
    # Teste Segmento Estável
    y_long = np.ones(5000) # 5 segundos a 1kHz
    y_stable = get_stable_segment(y_long, sr, duration=2.0)
    assert len(y_stable) == 2000
    
    print("OK - windowing.py: Silêncio adaptativo e Segmentação validados.")

if __name__ == "__main__":
    sanity_test()