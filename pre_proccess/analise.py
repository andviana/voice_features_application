import numpy as np

def analisar_amostra(y, sr):
    """Extrai metadados do sinal para validação estatística."""
    return {
        "Duração (s)": round(len(y)/sr, 2),
        "Sample Rate": sr,
        "Amp Máx": round(float(np.max(y)), 4),
        "Amp Mín": round(float(np.min(y)), 4),
        "DC Offset": round(float(np.mean(y)), 6),
        "RMS": round(float(np.sqrt(np.mean(y**2))), 4)
    }

def sanity_test():
    """Valida a extração de metadados."""
    sr = 44100
    y_test = np.random.uniform(-0.5, 0.5, sr)
    res = analisar_amostra(y_test, sr)
    
    # Verificação de chaves e tipos
    assert all(k in res for k in ["Duração (s)", "Sample Rate", "Amp Máx", "DC Offset", "RMS"])
    assert res["Sample Rate"] == sr
    assert isinstance(res["RMS"], float)
    print("OK - analise.py: Metadados validados.")

if __name__ == "__main__":
    sanity_test()