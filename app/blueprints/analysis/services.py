import librosa
import numpy as np
from scipy.signal import butter, freqz, correlate
from scipy.signal import welch
from pathlib import Path


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    t = np.arange(len(y)) / sr
    return y, sr, t


def waveform(y, sr):
    t = np.arange(len(y)) / sr
    return t.tolist(), y.tolist()


def spectrum(y, sr):
    Y = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    return freqs.tolist(), Y.tolist()


def spectrogram(y, sr):
    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S)

    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)

    return freqs.tolist(), times.tolist(), S_db.tolist()


def butterworth_response(sr, cutoff=80, order=4):

    nyq = sr / 2
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype="highpass")

    w, h = freqz(b, a, worN=8000)

    freqs = (w * sr) / (2 * np.pi)

    # evitar divisao por zero
    # gain = 20 * np.log10(abs(h))
    gain = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    return freqs.tolist(), gain.tolist()


def psd(y, sr):

    f, pxx = welch(y, sr)

    return f.tolist(), pxx.tolist()


def autocorrelation(y):

    corr = correlate(y, y, mode="full")
    corr = corr[len(corr) // 2 :]

    return corr.tolist()


def amplitude_histogram(y):

    hist, bins = np.histogram(y, bins=50)

    return bins.tolist(), hist.tolist()