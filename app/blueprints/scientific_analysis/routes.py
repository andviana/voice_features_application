from __future__ import annotations

import os
import math
import soundfile as sf
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, freqz, welch, correlate

from flask import current_app, render_template, abort, jsonify, send_file

from . import bp

ALLOWED_GROUPS = {"HC_AH", "PD_AH"}


@dataclass(frozen=True)
class AudioProps:
    group: str
    filename: str
    size_bytes: int
    modified_at: str
    n_channels: int
    sample_rate_hz: int
    n_frames: int
    sample_width_bytes: int
    duration_sec: float
    comptype: str
    compname: str


def _data_root() -> Path:
    return Path(current_app.config["DATA_DIR"]).resolve()


def _raw_root() -> Path:
    return (_data_root() / "audio_raw").resolve()


def _processed_root() -> Path:
    return (_data_root() / "audio_processed").resolve()


def _safe_group_dir(base: Path, group: str) -> Path:
    if group not in ALLOWED_GROUPS:
        abort(404)
    d = (base / group).resolve()
    if base not in d.parents and d != base:
        abort(400)
    return d


def _safe_wav_path(base: Path, group: str, filename: str) -> Path:
    gdir = _safe_group_dir(base, group)
    safe_name = os.path.basename(filename)
    fpath = (gdir / safe_name).resolve()
    if not fpath.exists() or fpath.suffix.lower() != ".wav":
        abort(404)
    return fpath


def _read_wav_props(path: Path, group: str) -> AudioProps:
    st = path.stat()
    modified = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    info = sf.info(str(path))

    n_channels = int(info.channels)
    sr = int(info.samplerate)
    n_frames = int(info.frames)

    duration = float(info.duration) if info.duration is not None else (
        n_frames / float(sr) if sr else 0.0
    )

    compname = str(info.subtype) if info.subtype else ""
    comptype = str(info.format) if info.format else ""

    subtype_bits_map = {
        "PCM_U8": 8,
        "PCM_S8": 8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
    }
    bits = subtype_bits_map.get(compname, 0)
    sample_width_bytes = bits // 8 if bits else 0

    return AudioProps(
        group=group,
        filename=path.name,
        size_bytes=st.st_size,
        modified_at=modified,
        n_channels=n_channels,
        sample_rate_hz=sr,
        n_frames=n_frames,
        sample_width_bytes=sample_width_bytes,
        duration_sec=duration,
        comptype=comptype,
        compname=compname,
    )


def _sanitize(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 3000):
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points).astype(int)
    return x[idx], y[idx]


def _load_audio(path: Path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y.astype(float), int(sr)


def _waveform(y: np.ndarray, sr: int):
    t = np.arange(len(y)) / float(sr)
    t_ds, y_ds = _downsample_xy(t, y, max_points=3000)
    return {"x": t_ds.tolist(), "y": y_ds.tolist()}


def _spectrum(y: np.ndarray, sr: int):
    n = int(2 ** np.ceil(np.log2(len(y)))) if len(y) > 1 else 1
    Y = np.fft.rfft(y, n=n)
    mag = np.abs(Y)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    freqs_ds, mag_ds = _downsample_xy(freqs, mag, max_points=3000)
    return {"x": freqs_ds.tolist(), "y": mag_ds.tolist()}


def _spectrogram(y: np.ndarray, sr: int, max_time_bins: int = 220, max_freq_bins: int = 180):
    n_fft = 1024
    hop_length = 256

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)

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


def _psd(y: np.ndarray, sr: int):
    f, pxx = welch(y, fs=sr, nperseg=min(2048, len(y)))
    return {"x": f.tolist(), "y": pxx.tolist()}


def _psd_zoom_f0(y: np.ndarray, sr: int, max_hz: float = 500.0):
    f, pxx = welch(y, fs=sr, nperseg=min(4096, len(y)))
    mask = f <= max_hz
    return {"x": f[mask].tolist(), "y": pxx[mask].tolist()}


def _autocorr(y: np.ndarray, sr: int, max_lag_sec: float = 0.05):
    corr = correlate(y, y, mode="full")
    corr = corr[len(corr) // 2 :]
    max_lag = min(len(corr), int(max_lag_sec * sr))
    corr = corr[:max_lag]
    lags = np.arange(len(corr)) / float(sr)
    return {"x": lags.tolist(), "y": corr.tolist()}


def _amplitude_hist(y: np.ndarray, bins: int = 60):
    hist, edges = np.histogram(y, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"x": centers.tolist(), "y": hist.tolist()}


def _butter_response(sr: int, cutoff: float = 80.0, order: int = 4):
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


def _load_manifest_row(filename: str):
    manifest = (_data_root() / "metadata" / "manifest.csv").resolve()
    if not manifest.exists():
        return {"highlight": {}, "all": {}}

    df = pd.read_csv(manifest)

    file_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id"):
            file_col = c
            break

    if file_col is None:
        return {"highlight": {}, "all": {}}

    key1 = filename
    key2 = filename.replace(".wav", "")
    s = df[file_col].astype(str).str.strip()
    row = df.loc[(s == key1) | (s == key2)]
    if row.empty:
        return {"highlight": {}, "all": {}}

    d = {k: _sanitize(v) for k, v in row.iloc[0].to_dict().items()}

    highlight = {}
    for out_key, candidates in {
        "label": ["label", "grupo", "group", "class"],
        "age": ["age", "idade"],
        "sex": ["sex", "sexo", "gender"],
    }.items():
        for c in candidates:
            if c in d:
                highlight[out_key] = d[c]
                break

    return {"highlight": highlight, "all": d}


def _features_csv_for_group(group: str) -> Path:
    p1 = (_data_root() / "features" / group / "dataset_voz_completo.csv").resolve()
    if p1.exists():
        return p1
    p2 = (_data_root() / "features" / "dataset_voz_completo.csv").resolve()
    return p2


def _pair_features(row: pd.Series):

    data = {k: _sanitize(v) for k, v in row.to_dict().items()}

    used = set()
    paired = []
    single = []

    IGNORE_COLUMNS = {"file_name", "group"}

    for k, v in data.items():

        if k in IGNORE_COLUMNS:
            continue

        lk = k.lower()

        if "_mean" in lk:

            std_key = k.replace("_mean_hz", "_std_hz").replace("_mean", "_std")
            base = lk.replace("_mean_hz", "").replace("_mean", "")

            if std_key in data:

                paired.append({
                    "feature": base,
                    "mean": data.get(k),
                    "std": data.get(std_key),
                    "mean_key": k,
                    "std_key": std_key,
                })

                used.add(k)
                used.add(std_key)

            else:
                # mean sem std → vai para single
                single.append({
                    "feature": base,
                    "value": data.get(k)
                })

                used.add(k)

    for k, v in data.items():

        if k in IGNORE_COLUMNS:
            continue

        if k in used:
            continue

        single.append({
            "feature": k,
            "value": v
        })

    paired.sort(key=lambda x: x["feature"])
    single.sort(key=lambda x: x["feature"])

    return paired, single


def _safe_max(values, default=1.0):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return max(vals) if vals else default


def _safe_min(values, default=0.0):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return min(vals) if vals else default


def _flatten_2d(z):
    out = []
    for row in z:
        for v in row:
            if v is not None and np.isfinite(v):
                out.append(float(v))
    return out


@bp.get("/")
def index():
    raw_base = _raw_root()
    audios = {g: [] for g in sorted(ALLOWED_GROUPS)}

    for group in sorted(ALLOWED_GROUPS):
        gdir = _safe_group_dir(raw_base, group)
        audios[group] = sorted([p.name for p in gdir.glob("*.wav")], key=lambda s: s.lower())

    return render_template("scientific_analysis/list.html", audios=audios)


@bp.get("/patient/<group>/<path:filename>")
def patient_page(group: str, filename: str):
    return render_template("scientific_analysis/view.html", group=group, filename=filename)


@bp.get("/data/<group>/<path:filename>")
def patient_data(group: str, filename: str):
    raw_path = _safe_wav_path(_raw_root(), group, filename)
    processed_path = _safe_wav_path(_processed_root(), group, filename)

    y_raw, sr_raw = _load_audio(raw_path)
    y_proc, sr_proc = _load_audio(processed_path)

    raw_props = asdict(_read_wav_props(raw_path, group))
    proc_props = asdict(_read_wav_props(processed_path, group))

    demographics = _load_manifest_row(filename)

    paired_features = []
    single_features = []
    features_csv = _features_csv_for_group(group)
    if features_csv.exists():
        df = pd.read_csv(features_csv)
        if "file_name" in df.columns:
            hit = df.loc[df["file_name"].astype(str).str.strip() == filename]
            if not hit.empty:
                paired_features, single_features = _pair_features(hit.iloc[0])

    wave_raw = _waveform(y_raw, sr_raw)
    wave_proc = _waveform(y_proc, sr_proc)

    spec_raw = _spectrum(y_raw, sr_raw)
    spec_proc = _spectrum(y_proc, sr_proc)

    specgram_raw = _spectrogram(y_raw, sr_raw)
    specgram_proc = _spectrogram(y_proc, sr_proc)

    psd_raw = _psd(y_raw, sr_raw)
    psd_proc = _psd(y_proc, sr_proc)

    psd_f0_raw = _psd_zoom_f0(y_raw, sr_raw, max_hz=500.0)
    psd_f0_proc = _psd_zoom_f0(y_proc, sr_proc, max_hz=500.0)

    auto_raw = _autocorr(y_raw, sr_raw)
    auto_proc = _autocorr(y_proc, sr_proc)

    hist_raw = _amplitude_hist(y_raw)
    hist_proc = _amplitude_hist(y_proc)

    payload = {
        "group": group,
        "filename": filename,
        "demographics": demographics,
        "raw": {
            "play_url": f"/scientific-analysis/play/raw/{group}/{filename}",
            "props": raw_props,
            "waveform": wave_raw,
            "spectrum": spec_raw,
            "spectrogram": specgram_raw,
            "psd": psd_raw,
            "psd_f0": psd_f0_raw,
            "autocorr": auto_raw,
            "hist": hist_raw,
        },
        "processed": {
            "play_url": f"/scientific-analysis/play/processed/{group}/{filename}",
            "props": proc_props,
            "waveform": wave_proc,
            "spectrum": spec_proc,
            "spectrogram": specgram_proc,
            "psd": psd_proc,
            "psd_f0": psd_f0_proc,
            "autocorr": auto_proc,
            "hist": hist_proc,
        },
        "filter_response": _butter_response(sr_raw, cutoff=80.0, order=4),
        "features": {
            "paired": paired_features,
            "single": single_features,
        },
        "ranges": {
            "waveform": {
                "x": [0, _safe_max([
                    wave_raw["x"][-1] if wave_raw["x"] else 0,
                    wave_proc["x"][-1] if wave_proc["x"] else 0,
                ], 1.0)],
                "y": [
                    _safe_min([
                        min(wave_raw["y"]) if wave_raw["y"] else 0,
                        min(wave_proc["y"]) if wave_proc["y"] else 0,
                    ], -1.0),
                    _safe_max([
                        max(wave_raw["y"]) if wave_raw["y"] else 1,
                        max(wave_proc["y"]) if wave_proc["y"] else 1,
                    ], 1.0),
                ],
            },
            "spectrum": {
                "x": [0, _safe_max([
                    spec_raw["x"][-1] if spec_raw["x"] else 0,
                    spec_proc["x"][-1] if spec_proc["x"] else 0,
                ], 1000.0)],
                "y": [
                    0,
                    _safe_max([
                        max(spec_raw["y"]) if spec_raw["y"] else 1,
                        max(spec_proc["y"]) if spec_proc["y"] else 1,
                    ], 1.0),
                ],
            },
            "spectrogram": {
                "x": [0, _safe_max([
                    specgram_raw["x"][-1] if specgram_raw["x"] else 0,
                    specgram_proc["x"][-1] if specgram_proc["x"] else 0,
                ], 1.0)],
                "y": [0, _safe_max([
                    specgram_raw["y"][-1] if specgram_raw["y"] else 0,
                    specgram_proc["y"][-1] if specgram_proc["y"] else 0,
                ], 4000.0)],
                "z": [
                    _safe_min(_flatten_2d(specgram_raw["z"]) + _flatten_2d(specgram_proc["z"]), -100.0),
                    _safe_max(_flatten_2d(specgram_raw["z"]) + _flatten_2d(specgram_proc["z"]), 0.0),
                ],
            },
            "psd_f0": {
                "x": [0, 500],
                "y": [
                    _safe_min(
                        [v for v in psd_f0_raw["y"] if v > 0] + [v for v in psd_f0_proc["y"] if v > 0],
                        1e-12
                    ),
                    _safe_max(psd_f0_raw["y"] + psd_f0_proc["y"], 1.0),
                ],
            },
            "autocorr": {
                "x": [0, _safe_max([
                    auto_raw["x"][-1] if auto_raw["x"] else 0,
                    auto_proc["x"][-1] if auto_proc["x"] else 0,
                ], 0.05)],
                "y": [
                    _safe_min(auto_raw["y"] + auto_proc["y"], -1.0),
                    _safe_max(auto_raw["y"] + auto_proc["y"], 1.0),
                ],
            },
            "hist": {
                "x": [
                    _safe_min(hist_raw["x"] + hist_proc["x"], -1.0),
                    _safe_max(hist_raw["x"] + hist_proc["x"], 1.0),
                ],
                "y": [
                    0,
                    _safe_max(hist_raw["y"] + hist_proc["y"], 1.0),
                ],
            },
        },
    }

    return jsonify(payload)


@bp.get("/play/raw/<group>/<path:filename>")
def play_raw(group: str, filename: str):
    fpath = _safe_wav_path(_raw_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


@bp.get("/play/processed/<group>/<path:filename>")
def play_processed(group: str, filename: str):
    fpath = _safe_wav_path(_processed_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)