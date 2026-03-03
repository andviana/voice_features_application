from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from flask import current_app, render_template, abort, send_file, jsonify, Response, stream_with_context

from app.utils.audio_props import AudioProps, AudioProps, read_wav_props, SOUNDFILE_ERRORS

import numpy as np
import pandas as pd
import librosa
from . import bp

ALLOWED_GROUPS = {"HC_AH", "PD_AH"}


def _data_root() -> Path:
    return Path(current_app.config["DATA_DIR"]).resolve()


def _audio_processed_root() -> Path:
    return (_data_root() / "audio_processed").resolve()


def _safe_group_dir(group: str) -> Path:
    if group not in ALLOWED_GROUPS:
        abort(404)

    base = _audio_processed_root()
    d = (base / group).resolve()

    if base not in d.parents and d != base:
        abort(400)

    return d


def _safe_wav_path(group: str, filename: str) -> Path:
    gdir = _safe_group_dir(group)
    safe_name = os.path.basename(filename)
    fpath = (gdir / safe_name).resolve()

    if not fpath.exists() or fpath.suffix.lower() != ".wav":
        abort(404)

    return fpath


def _load_manifest_row(filename: str) -> dict:
    """
    Retorna a linha inteira do manifest (todas as colunas) para o arquivo.
    Não assume nomes fixos de colunas.
    """
    manifest = (_data_root() / "metadata" / "manifest.csv").resolve()
    if not manifest.exists():
        return {}

    df = pd.read_csv(manifest)

    # tenta identificar coluna de filename
    file_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio"):
            file_col = c
            break

    if file_col is None:
        return {}

    # match exato por nome
    row = df.loc[df[file_col].astype(str).str.strip() == filename]
    if row.empty:
        return {}

    # pega a primeira ocorrência
    return row.iloc[0].to_dict()


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 2000):
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points).astype(int)
    return x[idx], y[idx]


def _compute_waveform_and_spectrum(wav_path: Path, max_points: int = 2000):
    """
    Gera dados leves pra plotar:
    - waveform: tempo(s) vs amplitude
    - spectrum: freq(Hz) vs magnitude (FFT)
    """
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y.size == 0 or sr is None or sr == 0:
        return {"waveform": None, "spectrum": None}

    # Waveform
    t = np.arange(len(y)) / float(sr)
    t_ds, y_ds = _downsample_xy(t, y, max_points=max_points)

    # Spectrum (FFT)
    n = int(2 ** np.ceil(np.log2(len(y)))) if len(y) > 1 else 1
    Y = np.fft.rfft(y, n=n)
    mag = np.abs(Y)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))

    # reduz pontos do espectro também
    freqs_ds, mag_ds = _downsample_xy(freqs, mag, max_points=max_points)

    return {
        "sr": int(sr),
        "waveform": {
            "t": t_ds.tolist(),
            "y": y_ds.tolist(),
        },
        "spectrum": {
            "f": freqs_ds.tolist(),
            "mag": mag_ds.tolist(),
        },
    }


def _iter_wavs(group_dir: Path) -> Iterable[Path]:
    # ordena por nome
    return sorted(group_dir.glob("*.wav"), key=lambda p: p.name.lower())


@bp.get("/audio-processed")
def list_audios_processed():
    base = _audio_processed_root()
    items: list[AudioProps] = []
    counts = {}

    for group in sorted(ALLOWED_GROUPS):
        gdir = _safe_group_dir(group)
        wavs = list(_iter_wavs(gdir))
        counts[group] = len(wavs)

        for w in wavs:
            try:
                items.append(read_wav_props(w, group))
            
            except SOUNDFILE_ERRORS:
                st = w.stat()
                items.append(
                    AudioProps(
                        group=group,
                        filename=w.name,
                        rel_id=f"{group}/{w.name}",
                        size_bytes=st.st_size,
                        modified_at="",
                        n_channels=0,
                        sample_rate_hz=0,
                        n_frames=0,
                        sample_width_bytes=0,
                        duration_sec=0.0,
                        comptype="invalid",
                        compname="invalid",
                    )
                )

    return render_template(
        "audio_processed/list.html",
        items=items,
        counts=counts,
        base=str(base),
    )


@bp.get("/audio-processed/play/<group>/<path:filename>")
def play_audio_processed(group: str, filename: str):
    fpath = _safe_wav_path(group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


@bp.get("/audio-processed/info/<group>/<path:filename>")
def audio_processed_info(group: str, filename: str):
    """
    Retorna JSON para o modal:
    - props do wav (wave)
    - demographics (linha do manifest)
    - waveform e spectrum para plot
    """
    fpath = _safe_wav_path(group, filename)

    props = read_wav_props(fpath, group)
    demographics = _load_manifest_row(fpath.name)
    plots = _compute_waveform_and_spectrum(fpath, max_points=2000)

    return jsonify({
        "props": asdict(props),
        "demographics": demographics,
        "plots": plots,
    })
