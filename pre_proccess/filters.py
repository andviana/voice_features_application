import numpy as np
from scipy import signal


def remove_dc_offset(y):
    """
    Subtrai a média temporal do sinal conforme item 4.1.2.
    Vital para Tsallis: centraliza o histograma do sinal no zero.
    """
    return y - np.mean(y)


def apply_bandpass(y, sr, low=80, high=8000):
    """
    Filtro Butterworth de 4ª ordem (80-8000 Hz).
    Remove sub-harmônicos e ruído ultrassônico (item 4.1.3).
    isolar a banda vocal.
    """
    nyq = 0.5 * sr
    low_norm = low / nyq
    high_norm = high / nyq

    # Garantir que a frequência de corte não exceda Nyquist
    if high_norm >= 1:
        high_norm = 0.99

    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
    return signal.lfilter(b, a, y)


# def apply_shen_filter(y, sr, sex):
#     """
#     Implementa os limites de Shen (2025) / Iyer (2023):
#     - Masculino (M): 75 Hz (floor) a 300 Hz (ceiling)
#     - Feminino (F): 100 Hz (floor) a 600 Hz (ceiling)
#     """
#     if sex.upper() == 'M':
#         low, high = 75, 300
#     elif sex.upper() == 'F':
#         low, high = 100, 600
#     else:
#         # Fallback para casos sem metadados (banda vocal padrão)
#         low, high = 80, 10000 
        
#     return apply_bandpass(y, sr, low, high)

def sanity_test():
    """Valida as funções de filtragem."""
    sr = 1000
    t = np.linspace(0, 1, sr)
    # Sinal com DC offset e frequências fora da banda
    y = 0.5 + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 200 * t)
    
    # Teste DC Offset
    y_no_dc = remove_dc_offset(y)
    assert np.isclose(np.mean(y_no_dc), 0, atol=1e-7)
    
    # Teste Bandpass (verificando atenuação de frequências baixas)
    y_filtered = apply_bandpass(y_no_dc, sr, low=50, high=450)
    # A componente de 10Hz deve ser drasticamente reduzida
    assert np.std(y_filtered) < np.std(y_no_dc)
    
    print("OK - filters.py: DC Offset e Bandpass validados.")

if __name__ == "__main__":
    sanity_test()