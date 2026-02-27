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
    y_test = np.random.uniform(-0.5, 0.5, 44100)
    res = analisar_amostra(y_test, 44100)
    assert res["Sample Rate"] == 44100
    print("OK - analise.py")

if __name__ == "__main__":
    sanity_test()