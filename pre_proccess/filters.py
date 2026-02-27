import numpy as np
from scipy import signal

def remove_dc_offset(y):
    """Vital para Tsallis: centraliza o histograma do sinal no zero."""
    return y - np.mean(y)

def apply_bandpass(y, sr, low, high):
    """Filtro de 4ª ordem para isolar a banda vocal."""
    nyq = 0.5 * sr
    # Normalizamos as frequências de corte pela frequência de Nyquist    
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
    return signal.lfilter(b, a, y)

def apply_shen_filter(y, sr, sex):
    """
    Implementa os limites de Shen (2025) / Iyer (2023):
    - Masculino (M): 75 Hz (floor) a 300 Hz (ceiling)
    - Feminino (F): 100 Hz (floor) a 600 Hz (ceiling)
    """
    if sex.upper() == 'M':
        low, high = 75, 300
    elif sex.upper() == 'F':
        low, high = 100, 600
    else:
        # Fallback para casos sem metadados (banda vocal padrão)
        low, high = 80, 10000 
        
    return apply_bandpass(y, sr, low, high)

def sanity_test():
    y = np.ones(100) + 0.5
    y_clean = remove_dc_offset(y)
    assert np.isclose(np.mean(y_clean), 0)
    print("OK - filters.py")

if __name__ == "__main__":
    sanity_test()