from __future__ import annotations

import os
import soundfile as sf
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from flask import render_template, abort, jsonify, send_file
from app. utils.path_utils import PathUtils
from app.utils.audio_props import AudioProps

from app.services.metadata_service import MetadataService
from app.services.audio_signal_service import AudioSignalsService
from app.services.analysis_service import AnalysisService
from . import bp

ALLOWED_GROUPS = {"HC_AH", "PD_AH"}

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
        rel_id=path.name,
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


def _load_audio(path: Path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y.astype(float), int(sr)


def _features_csv_for_group(group: str) -> Path:
    p1 = (PathUtils.data_root() / "features" / group / "dataset_voz_completo.csv").resolve()
    if p1.exists():
        return p1
    p2 = (PathUtils.data_root() / "features" / "dataset_voz_completo.csv").resolve()
    return p2


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
    raw_base = PathUtils.raw_root()
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
    raw_path = _safe_wav_path(PathUtils.raw_root(), group, filename)
    processed_path = _safe_wav_path(PathUtils.processed_root(), group, filename)

    y_raw, sr_raw = _load_audio(raw_path)
    y_proc, sr_proc = _load_audio(processed_path)

    raw_props = asdict(_read_wav_props(raw_path, group))
    proc_props = asdict(_read_wav_props(processed_path, group))

    demographics = MetadataService.load_manifest_row(filename)

    paired_features = []
    single_features = []
    features_csv = _features_csv_for_group(group)
    if features_csv.exists():
        df = pd.read_csv(features_csv)
        if "file_name" in df.columns:
            hit = df.loc[df["file_name"].astype(str).str.strip() == filename]
            if not hit.empty:
                paired_features, single_features = AnalysisService.pair_features(hit.iloc[0])

    wave_raw = AudioSignalsService.waveform(y_raw, sr_raw)
    wave_proc = AudioSignalsService.waveform(y_proc, sr_proc)

    spec_raw = AudioSignalsService.spectrum(y_raw, sr_raw)
    spec_proc = AudioSignalsService.spectrum(y_proc, sr_proc)

    specgram_raw = AudioSignalsService.spectrogram(y_raw, sr_raw)
    specgram_proc = AudioSignalsService.spectrogram(y_proc, sr_proc)

    psd_raw = AudioSignalsService.psd(y_raw, sr_raw)
    psd_proc = AudioSignalsService.psd(y_proc, sr_proc)

    psd_f0_raw = AudioSignalsService.psd_zoom_f0(y_raw, sr_raw, max_hz=500.0)
    psd_f0_proc = AudioSignalsService.psd_zoom_f0(y_proc, sr_proc, max_hz=500.0)

    auto_raw = AudioSignalsService.autocorr(y_raw, sr_raw)
    auto_proc = AudioSignalsService.autocorr(y_proc, sr_proc)

    hist_raw = AudioSignalsService.amplitude_hist(y_raw)
    hist_proc = AudioSignalsService.amplitude_hist(y_proc)

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
        "filter_response": AudioSignalsService.butter_response(sr_raw, cutoff=80.0, order=4),
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
    fpath = _safe_wav_path(PathUtils.raw_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


@bp.get("/play/processed/<group>/<path:filename>")
def play_processed(group: str, filename: str):
    fpath = _safe_wav_path(PathUtils.processed_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)