from __future__ import annotations

import os
import wave
import math
import soundfile as sf
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

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


def _sanitize_json_value(v):
    # converte numpy types para python float/int
    try:
        # numpy.float64, etc.
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
    except Exception:
        pass

    # NaN/Inf -> None (vira null no JSON)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None

    return v


def _sanitize_features(features: list[dict]) -> list[dict]:
    out = []
    for r in features:
        out.append({
            **r,
            "mean": _sanitize_json_value(r.get("mean")),
            "std": _sanitize_json_value(r.get("std")),
        })
    return out

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

    # 1) tenta wave (rápido e bom p/ PCM simples)
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            comptype = wf.getcomptype()
            compname = wf.getcompname()
            duration = (n_frames / float(sr)) if sr else 0.0

        return AudioProps(
            group=group,
            filename=path.name,
            size_bytes=st.st_size,
            modified_at=modified,
            n_channels=n_channels,
            sample_rate_hz=sr,
            n_frames=n_frames,
            sample_width_bytes=sampwidth,
            duration_sec=duration,
            comptype=comptype,
            compname=compname,
        )

    except wave.Error:
        # 2) fallback: WAVE_FORMAT_EXTENSIBLE (65534) e outros casos
        with sf.SoundFile(str(path)) as snd:
            sr = int(snd.samplerate)
            n_channels = int(snd.channels)
            n_frames = int(len(snd))  # frames
            # subtype/format são strings do libsndfile
            comptype = str(snd.subtype)  # ex: 'PCM_16', 'FLOAT'
            compname = str(snd.format)   # ex: 'WAV', 'WAVEX'
            duration = (n_frames / float(sr)) if sr else 0.0

            # sample width em bytes (aproximação correta para PCM/float)
            # snd.subtype_info às vezes traz bits, mas não é padronizado.
            # Vamos inferir do subtype quando possível:
            sample_width_bytes = 0
            subtype = snd.subtype.upper()
            if "PCM_16" in subtype:
                sample_width_bytes = 2
            elif "PCM_24" in subtype:
                sample_width_bytes = 3
            elif "PCM_32" in subtype:
                sample_width_bytes = 4
            elif "FLOAT" in subtype:
                sample_width_bytes = 4
            elif "DOUBLE" in subtype:
                sample_width_bytes = 8

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


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 2000):
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points).astype(int)
    return x[idx], y[idx]


def _plots_waveform_spectrum(wav_path: Path, max_points: int = 2000) -> dict:
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0 or sr is None or sr == 0:
        return {"sr": 0, "waveform": None, "spectrum": None}

    t = np.arange(len(y)) / float(sr)
    t_ds, y_ds = _downsample_xy(t, y, max_points=max_points)

    n = int(2 ** np.ceil(np.log2(len(y)))) if len(y) > 1 else 1
    Y = np.fft.rfft(y, n=n)
    mag = np.abs(Y)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))

    freqs_ds, mag_ds = _downsample_xy(freqs, mag, max_points=max_points)

    return {
        "sr": int(sr),
        "waveform": {"t": np.round(t_ds, 2).tolist(), "y": y_ds.tolist()},
        "spectrum": {"f": np.round(freqs_ds, 2).tolist(), "mag": mag_ds.tolist()},
    }


def _compute_spectrogram(wav_path: Path, max_time_bins: int = 180, max_freq_bins: int = 128) -> dict:
    """
    Gera dados leves para plotar espectrograma STFT:
    - x: tempos
    - y: frequências
    - z: matriz em dB (freq x tempo)
    """
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0 or sr is None or sr == 0:
        return {"times": [], "freqs": [], "z": []}

    n_fft = 1024
    hop_length = 256

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)

    # Downsample temporal
    if S_db.shape[1] > max_time_bins:
        tidx = np.linspace(0, S_db.shape[1] - 1, num=max_time_bins).astype(int)
        S_db = S_db[:, tidx]
        times = times[tidx]

    # Downsample frequencial
    if S_db.shape[0] > max_freq_bins:
        fidx = np.linspace(0, S_db.shape[0] - 1, num=max_freq_bins).astype(int)
        S_db = S_db[fidx, :]
        freqs = freqs[fidx]

    return {
        "times": times.tolist(),
        "freqs": freqs.tolist(),
        "z": S_db.tolist(),  # matriz [freq][tempo]
    }


def _load_manifest_row(filename: str) -> dict:
    manifest = (_data_root() / "metadata" / "manifest.csv").resolve()
    if not manifest.exists():
        return {}

    df = pd.read_csv(manifest)

    # encontra coluna do nome do arquivo
    file_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id"):
            file_col = c
            break
    if file_col is None:
        return {}

    # aceita match por nome completo ou sem .wav
    key = filename
    key2 = filename.replace(".wav", "")
    s = df[file_col].astype(str).str.strip()
    row = df.loc[(s == key) | (s == key2)]
    if row.empty:
        return {}

    d = row.iloc[0].to_dict()

    # tenta normalizar só os campos que você quer destacar (sem assumir que existam)
    out = {}
    for k in ("label", "age", "sex"):
        # procura variações comuns
        candidates = [k, k.upper(), k.capitalize()]
        found = None
        for c in candidates:
            if c in d:
                found = c
                break
        # variações
        if found is None and k == "sex":
            for c in ("sexo", "gender"):
                if c in d:
                    found = c
                    break
        if found is None and k == "age":
            for c in ("idade",):
                if c in d:
                    found = c
                    break
        if found is None and k == "label":
            for c in ("grupo", "group", "class"):
                if c in d:
                    found = c
                    break
        if found is not None:
            out[k] = d.get(found)

    # e também manda a linha inteira para referência
    return {"highlight": out, "all": d}


def _features_csv_for_group(group: str) -> Path:
    """
    Ajuste aqui conforme seu output:
    - se você salva por grupo: data/features/<group>/dataset_voz_completo.csv
    - ou se for global: data/features/dataset_voz_completo.csv
    """
    p1 = (_data_root() / "features" / group / "dataset_voz_completo.csv").resolve()
    if p1.exists():
        return p1
    p2 = (_data_root() / "features" / "dataset_voz_completo.csv").resolve()
    return p2


def _pair_features_mean_std(row: pd.Series) -> list[dict]:
    """
    Constrói lista de linhas {feature, mean, std}.
    Para qualquer coluna que termine com _mean_* ou _std_* (ou _mean/_std),
    tenta parear pelo prefixo base.
    """
    data = row.to_dict()

    means = {}
    stds = {}

    for k, v in data.items():
        if k == "file_name" or k == "group":
            continue
        kl = str(k).lower()

        # lista de chaves que devem ser verificadas
        keys_to_check = {
            "f0_min_hz", "f0_max_hz", "f0_cv",
            "hnr_mean_db",
            "jitter_local", "jitter_rap", "jitter_ppq5",
            "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
            "tsallis_sq_amp", "shannon_s1_amp",
            "tsallis_sq_f0", "shannon_s1_f0"
        }

        # exemplo de uso
        if k in keys_to_check:
            means[k] = (k, v)

        # casos típicos: f1_mean_hz, f1_std_hz, mfcc_01_mean, mfcc_01_std, etc.
        if "_mean" in kl:
            base = kl.replace("_mean_hz", "").replace("_mean", "")
            means[base] = (k, v)
        if "_std" in kl:
            base = kl.replace("_std_hz", "").replace("_std", "")
            stds[base] = (k, v)

    # união por base
    bases = sorted(set(means.keys()) | set(stds.keys()))
    out = []
    for b in bases:
        mean_k, mean_v = means.get(b, (None, None))
        std_k, std_v = stds.get(b, (None, None))
        # usa o "b" como nome amigável
        out.append({
            "feature": b,
            "mean": mean_v,
            "std": std_v,
            "mean_key": mean_k,
            "std_key": std_k,
        })
    return out


@bp.get("/compare")
def compare_view():
    # lista de pacientes = nomes de raw (e se não existir processed, ainda aparece)
    raw_base = _raw_root()
    patients = {g: [] for g in sorted(ALLOWED_GROUPS)}

    for group in sorted(ALLOWED_GROUPS):
        gdir = _safe_group_dir(raw_base, group)
        patients[group] = sorted([p.name for p in gdir.glob("*.wav")], key=lambda s: s.lower())

    return render_template("compare/view.html", patients=patients)


@bp.get("/compare/info/<group>/<path:filename>")
def compare_info(group: str, filename: str):

    print("\n=============================")
    print("COMPARE INFO REQUEST")
    print("group:", group)
    print("filename:", filename)

    try:
        print("1) Resolving raw path...")
        raw_path = _safe_wav_path(_raw_root(), group, filename)
        print("   raw_path:", raw_path)

        print("2) Resolving processed path...")
        processed_path = _safe_wav_path(_processed_root(), group, filename)
        print("   processed_path:", processed_path)

        print("3) Reading RAW properties...")
        raw_props = _read_wav_props(raw_path, group)

        print("4) Reading PROCESSED properties...")
        proc_props = _read_wav_props(processed_path, group)

        print("5.1) Computing RAW plots (waveform + spectrum)...")
        raw_plots = _plots_waveform_spectrum(raw_path)

        print("5.2) Computing PROCESSED plots (waveform + spectrum)...")
        proc_plots = _plots_waveform_spectrum(processed_path)

        print("6.1) Computing RAW spectrogram...")
        raw_specgram = _compute_spectrogram(raw_path)

        print("6.2) Computing PROCESSED spectrogram...")
        proc_specgram = _compute_spectrogram(processed_path)

        print("7) Loading demographics...")
        demo = _load_manifest_row(raw_path.name)

        print("8) Loading features CSV...")
        features_csv = _features_csv_for_group(group)
        print("   features_csv:", features_csv)

        features = []

        if features_csv.exists():
            print("9) Reading CSV...")
            df = pd.read_csv(features_csv)

            if "file_name" in df.columns:
                print("10) Searching row for:", raw_path.name)

                hit = df.loc[df["file_name"].astype(str).str.strip() == raw_path.name]

                if not hit.empty:
                    print("11) Features found")
                    features = _pair_features_mean_std(hit.iloc[0])
                else:
                    print("11) WARNING: audio not found in CSV")

        else:
            print("WARNING: features CSV not found")

        print("12) Building response JSON")

        features = _sanitize_features(features)
        resp = {
            "group": group,
            "filename": raw_path.name,
            "demographics": demo,
            "raw": {
                "props": asdict(raw_props),
                "plots": raw_plots,
                "spectrogram": raw_specgram,
                "play_url": f"/compare/play/raw/{group}/{raw_path.name}",
            },
            "processed": {
                "props": asdict(proc_props),
                "plots": proc_plots,
                "spectrogram": proc_specgram,
                "play_url": f"/compare/play/processed/{group}/{processed_path.name}",
            },
            "features": features,
        }

        print("13) Response ready")
        print("=============================\n")


        return jsonify(resp)

    except Exception as e:

        print("\n!!!!!! ERROR IN compare_info !!!!!!")
        print(e)
        import traceback
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        raise


@bp.get("/compare/play/raw/<group>/<path:filename>")
def play_raw(group: str, filename: str):
    fpath = _safe_wav_path(_raw_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


@bp.get("/compare/play/processed/<group>/<path:filename>")
def play_processed(group: str, filename: str):
    fpath = _safe_wav_path(_processed_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)