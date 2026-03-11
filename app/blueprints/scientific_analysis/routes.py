from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np

from flask import render_template, jsonify, send_file
from app. utils.path_utils import PathUtils
from app.utils.audio_props import read_wav_props
from app.services.metadata_service import MetadataService
from app.services.audio_signal_service import AudioSignalsService
from app.services.analysis_service import AnalysisService
from . import bp


# Helper para evitar valores infinitos ou NaN no JSON de resposta
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
    """Lista todos os áudios disponíveis para análise científica."""
    raw_base = PathUtils.raw_root()
    audios = {}

    for group in sorted(PathUtils.ALLOWED_GROUPS):
        try:
            gdir = PathUtils.safe_group_dir(raw_base, group)
            audios[group] = sorted([p.name for p in gdir.glob("*.wav")], key=lambda s: s.lower())
        except Exception:
            audios[group] = []

    return render_template("scientific_analysis/list.html", audios=audios)


@bp.get("/patient/<group>/<path:filename>")
def patient_page(group: str, filename: str):
    return render_template("scientific_analysis/view.html", group=group, filename=filename)


@bp.get("/data/<group>/<path:filename>")
def patient_data(group: str, filename: str):
    """
    Agrega todos os dados para a análise comparativa (Raw vs Processed).
    Usa serviços centralizados para garantir consistência.
    """
    # 1. Resolução segura de caminhos via PathUtils
    raw_path = PathUtils.safe_wav_path(PathUtils.raw_root(), group, filename)
    proc_path = PathUtils.safe_wav_path(PathUtils.processed_root(), group, filename)

    # 2. Carregamento de sinal via AudioSignalsService
    y_raw, sr_raw = AudioSignalsService.load_audio(raw_path)
    y_proc, sr_proc = AudioSignalsService.load_audio(proc_path)

    # 3. Metadados e Propriedades via serviços globais
    raw_props = asdict(read_wav_props(raw_path, group))
    proc_props = asdict(read_wav_props(proc_path, group))
    demographics = MetadataService.load_manifest_row(filename)

    # 4. Extração de Features (Lógica do AnalysisService)
    paired_features, single_features = [], []
    # Tenta localizar o CSV de features do grupo
    features_csv = PathUtils.features_csv_for_group(group)
    if features_csv.exists():
        df = pd.read_csv(features_csv)
        hit = df.loc[df["file_name"].astype(str).str.strip() == filename]
        if not hit.empty:
            paired_features, single_features = AnalysisService.pair_features(hit.iloc[0])
    
    # 5. Processamento de Sinais para Gráficos
    # Dados Brutos
    wave_raw = AudioSignalsService.waveform(y_raw, sr_raw)
    spec_raw = AudioSignalsService.spectrum(y_raw, sr_raw)
    specgram_raw = AudioSignalsService.spectrogram(y_raw, sr_raw)
    psd_raw = AudioSignalsService.psd(y_raw, sr_raw)
    psd_f0_raw = AudioSignalsService.psd_zoom_f0(y_raw, sr_raw, max_hz=500.0)
    auto_raw = AudioSignalsService.autocorr(y_raw, sr_raw)
    hist_raw = AudioSignalsService.amplitude_hist(y_raw)

    # Dados processados
    wave_proc = AudioSignalsService.waveform(y_proc, sr_proc)
    spec_proc = AudioSignalsService.spectrum(y_proc, sr_proc)
    specgram_proc = AudioSignalsService.spectrogram(y_proc, sr_proc)
    psd_proc = AudioSignalsService.psd(y_proc, sr_proc)
    psd_f0_proc = AudioSignalsService.psd_zoom_f0(y_proc, sr_proc, max_hz=500.0)
    auto_proc = AudioSignalsService.autocorr(y_proc, sr_proc)
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
        # Mantém a resposta do filtro Butterworth para o gráfico de resposta em frequência
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
    fpath = PathUtils.safe_wav_path(PathUtils.raw_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


@bp.get("/play/processed/<group>/<path:filename>")
def play_processed(group: str, filename: str):
    fpath = PathUtils.safe_wav_path(PathUtils.processed_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)