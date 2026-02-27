from __future__ import annotations

import os
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from flask import current_app, render_template, abort, send_file, jsonify
from pathlib import Path
from flask import jsonify

from app.services.preproccess_service import start_preprocess_run

from . import bp

ALLOWED_GROUPS = {"HC_AH", "PD_AH"}


@dataclass(frozen=True)
class AudioProps:
    group: str
    filename: str
    rel_id: str  # "HC_AH/arquivo.wav"
    size_bytes: int
    modified_at: str
    n_channels: int
    sample_rate_hz: int
    n_frames: int
    sample_width_bytes: int
    duration_sec: float
    comptype: str
    compname: str


def _audio_raw_root() -> Path:
    return Path(current_app.config["AUDIO_RAW_DIR"]).resolve()


def _safe_group_dir(group: str) -> Path:
    if group not in ALLOWED_GROUPS:
        abort(404)
    base = _audio_raw_root()
    d = (base / group).resolve()
    # garante que está dentro do root
    if base not in d.parents and d != base:
        abort(400)
    return d

def _safe_wav_path(group: str, filename: str) -> Path:
    gdir = _safe_group_dir(group)
    safe_name = os.path.basename(filename)  # evita traversal
    fpath = (gdir / safe_name).resolve()
    if not fpath.exists() or fpath.suffix.lower() != ".wav":
        abort(404)
    return fpath

def _read_wav_props(path: Path, group: str) -> AudioProps:
    st = path.stat()
    modified = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # wave suporta PCM WAV padrão
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        comptype = wf.getcomptype()
        compname = wf.getcompname()
        duration = (n_frames / float(sr)) if sr else 0.0

    rel_id = f"{group}/{path.name}"

    return AudioProps(
        group=group,
        filename=path.name,
        rel_id=rel_id,
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


def _iter_wavs(group_dir: Path) -> Iterable[Path]:
    # ordena por nome
    return sorted(group_dir.glob("*.wav"), key=lambda p: p.name.lower())


@bp.get("/audio-raw")
def list_audio_raw():
    base = _audio_raw_root()
    items: list[AudioProps] = []
    counts = {}

    for group in sorted(ALLOWED_GROUPS):
        gdir = _safe_group_dir(group)
        wavs = list(_iter_wavs(gdir))
        counts[group] = len(wavs)

        for w in wavs:
            try:
                items.append(_read_wav_props(w, group))
            except wave.Error:
                # WAV inválido/corrompido: mostra com props mínimas
                st = w.stat()
                items.append(
                    AudioProps(
                        group=group,
                        filename=w.name,
                        rel_id=f"{group}/{w.name}",
                        size_bytes=st.st_size,
                        modified_at=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        n_channels=0,
                        sample_rate_hz=0,
                        n_frames=0,
                        sample_width_bytes=0,
                        duration_sec=0.0,
                        comptype="",
                        compname="",
                    )
                )

    return render_template("audio_raw/list.html", items=items, counts=counts, base=str(base))

@bp.get("/audio-raw/play/<group>/<path:filename>")
def play_audio(group: str, filename: str):
    fpath = _safe_wav_path(group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)

@bp.get("/audio-raw/info/<group>/<path:filename>")
def audio_info_json(group: str, filename: str):
    fpath = _safe_wav_path(group, filename)
    props = _read_wav_props(fpath, group)
    return jsonify(asdict(props))

@bp.get("/audio-raw/info/<group>/<path:filename>")
def audio_info(group: str, filename: str):
    gdir = _safe_group_dir(group)
    safe_name = os.path.basename(filename)
    fpath = (gdir / safe_name).resolve()
    if not fpath.exists() or fpath.suffix.lower() != ".wav":
        abort(404)

    props = _read_wav_props(fpath, group)
    return render_template("audio_raw/info.html", props=props, props_dict=asdict(props))


@bp.post("/audio-raw/preprocess/<group>/<path:filename>")
def preprocess_audio(group: str, filename: str):
    # input vindo do raw
    input_path = _safe_wav_path(group, filename)

    # output em data/audio_processed/<grupo>/<arquivo>.wav
    base_data = Path(current_app.config["DATA_DIR"]).resolve()
    output_path = (base_data / "audio_processed" / group / input_path.name).resolve()

    # manifest em data/metadata/manifest.csv
    manifest_path = (base_data / "metadata" / "manifest.csv").resolve()

    run_id = start_preprocess_run(
        input_wav_path=input_path,
        output_wav_path=output_path,
        manifest_path=manifest_path,
    )
    return jsonify({"run_id": run_id}), 202



@bp.get("/audio-raw/preprocess/stream/<run_id>")
def preprocess_stream(run_id: str):
    # Reusa o mesmo streaming do pipeline_manager (SSE)
    # Se você já tem /pipeline/stream/<run_id>, pode só redirecionar.
    from app.services.pipeline_service import pipeline_manager
    run = pipeline_manager.get_run(run_id)
    if run is None:
        return jsonify({"error": "run_not_found"}), 404

    def event_stream():
        yield "retry: 1000\n\n"
        while True:
            try:
                line = run.q.get(timeout=15)
            except Exception:
                yield ": keep-alive\n\n"
                continue

            if line == "__PIPELINE_DONE__":
                yield "event: done\ndata: preprocess_finished\n\n"
                break

            safe = str(line).replace("\r", "")
            yield f"data: {safe}\n\n"

    from flask import Response, stream_with_context
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")