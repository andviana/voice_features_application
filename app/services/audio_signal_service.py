import librosa
import numpy as np

from pathlib import Path
from scipy.signal import butter, freqz, welch, correlate


class AudioSignalsService:

    @staticmethod
    def load_audio(path: str | Path):
        """Centraliza o carregamento de áudio com librosa."""
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(float), int(sr)

    @staticmethod
    def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 3000):
        """Reduz a densidade de pontos para otimizar a renderização no front-end."""
        if len(x) <= max_points:
            return x, y
        idx = np.linspace(0, len(x) - 1, num=max_points).astype(int)
        return x[idx], y[idx]


    @staticmethod
    def waveform(y: np.ndarray, sr: int, max_points: int = 3000):
        """Gera os dados da forma de onda."""
        t = np.arange(len(y)) / float(sr)
        t_ds, y_ds = AudioSignalsService.downsample_xy(t, y, max_points)
        return {"x": t_ds.tolist(), "y": y_ds.tolist()}


    @staticmethod
    def spectrum(y: np.ndarray, sr: int, max_points: int = 3000):
        """Calcula o espectro de magnitude via FFT."""
        n = int(2 ** np.ceil(np.log2(len(y)))) if len(y) > 1 else 1
        Y = np.fft.rfft(y, n=n)
        mag = np.abs(Y)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
        freqs_ds, mag_ds = AudioSignalsService.downsample_xy(freqs, mag, max_points)
        return {"x": freqs_ds.tolist(), "y": mag_ds.tolist()}


    @staticmethod
    def spectrogram(y: np.ndarray, sr: int, max_time_bins: int = 220, max_freq_bins: int = 180):
        """Gera o espetrograma em dB."""
        n_fft = 1024
        hop_length = 256

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
        S_db = librosa.power_to_db(S, ref=np.max)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)

        # Redimensionamento para performance
        if S_db.shape[1] > max_time_bins:
            tidx = np.linspace(0, S_db.shape[1] - 1, num=max_time_bins).astype(int)
            S_db = S_db[:, tidx]
            times = times[tidx]

        if S_db.shape[0] > max_freq_bins:
            fidx = np.linspace(0, S_db.shape[0] - 1, num=max_freq_bins).astype(int)
            S_db = S_db[fidx, :]
            freqs = freqs[fidx]

        return {
            "x": times.tolist(),
            "y": freqs.tolist(),
            "z": S_db.tolist(),
        }
    

    # Métodos adicionais de análise avançada
    @staticmethod
    def psd(y: np.ndarray, sr: int):
        f, pxx = welch(y, fs=sr, nperseg=min(2048, len(y)))
        return {"x": f.tolist(), "y": pxx.tolist()}


    @staticmethod
    def psd_zoom_f0(y: np.ndarray, sr: int, max_hz: float = 500.0):
        f, pxx = welch(y, fs=sr, nperseg=min(4096, len(y)))
        mask = f <= max_hz
        return {"x": f[mask].tolist(), "y": pxx[mask].tolist()}


    @staticmethod    
    def autocorr(y: np.ndarray, sr: int, max_lag_sec: float = 0.05):
        corr = correlate(y, y, mode="full")
        corr = corr[len(corr) // 2 :]
        max_lag = min(len(corr), int(max_lag_sec * sr))
        corr = corr[:max_lag]
        lags = np.arange(len(corr)) / float(sr)
        return {"x": lags.tolist(), "y": corr.tolist()}


    @staticmethod    
    def amplitude_hist(y: np.ndarray, bins: int = 60):
        hist, edges = np.histogram(y, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {"x": centers.tolist(), "y": hist.tolist()}

    
    @staticmethod
    def butter_response(sr: int, cutoff: float = 80.0, order: int = 4):
        nyq = sr / 2.0
        norm = cutoff / nyq
        b, a = butter(order, norm, btype="highpass")
        w, h = freqz(b, a, worN=4096)
        freqs = (w * sr) / (2.0 * np.pi)
        gain_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        phase = np.unwrap(np.angle(h))
        return {
            "x": freqs.tolist(),
            "gain_db": gain_db.tolist(),
            "phase": phase.tolist(),
            "cutoff_hz": cutoff,
        }
