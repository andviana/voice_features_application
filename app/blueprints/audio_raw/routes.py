from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from flask import current_app, render_template, abort, send_file, jsonify, Response, stream_with_context

from app.utils.audio_props import AudioProps, AudioProps, read_wav_props, SOUNDFILE_ERRORS
from app.services.preproccess_service import start_preprocess_run
from . import bp

ALLOWED_GROUPS = {"HC_AH", "PD_AH"}


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
                items.append(read_wav_props(w, group))
            
            except SOUNDFILE_ERRORS:
                # arquivo inválido/corrompido ou formato não suportado pelo backend
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
                        comptype="",
                        compname="",
                    )
                )

    return render_template(
        "audio_raw/list.html", 
        items=items, 
        counts=counts, 
        base=str(base),
    )


@bp.get("/audio-raw/play/<group>/<path:filename>")
def play_audio(group: str, filename: str):
    fpath = _safe_wav_path(group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


# JSON para o modal (o front faz fetch e espera JSON)
@bp.get("/audio-raw/info-json/<group>/<path:filename>")
def audio_info_json(group: str, filename: str):
    fpath = _safe_wav_path(group, filename)
    props = read_wav_props(fpath, group)
    return jsonify(asdict(props))


# (Opcional) página HTML de info — se ainda quiser manter
@bp.get("/audio-raw/info/<group>/<path:filename>")
def audio_info(group: str, filename: str):
    fpath = _safe_wav_path(group, filename)
    props = read_wav_props(fpath, group)
    return render_template("audio_raw/info.html", props=props, props_dict=asdict(props))


@bp.post("/audio-raw/preprocess/<group>/<path:filename>")
def preprocess_audio(group: str, filename: str):
    input_path = _safe_wav_path(group, filename)

    base_data = Path(current_app.config["DATA_DIR"]).resolve()
    output_path = (base_data / "audio_processed" / group / input_path.name).resolve()
    manifest_path = (base_data / "metadata" / "manifest.csv").resolve()

    run_id = start_preprocess_run(
        input_wav_path=input_path,
        output_wav_path=output_path,
        manifest_path=manifest_path,
    )
    return jsonify({"run_id": run_id}), 202



@bp.get("/audio-raw/preprocess/stream/<run_id>")
def preprocess_stream(run_id: str):
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

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@bp.post("/audio-raw/preprocess-batch")
def preprocess_batch():
    base_data = Path(current_app.config["DATA_DIR"]).resolve()
    manifest_path = (base_data / "metadata" / "manifest.csv").resolve()

    tasks: list[tuple[Path, Path]] = []

    # percorre HC_AH e PD_AH (mesma lógica usada na listagem)
    for group in sorted(ALLOWED_GROUPS):
        gdir = _safe_group_dir(group)
        for w in _iter_wavs(gdir):
            input_path = w
            output_path = (base_data / "audio_processed" / group / w.name).resolve()
            tasks.append((input_path, output_path))

    from app.services.preproccess_service import start_preprocess_batch_run
    run_id = start_preprocess_batch_run(tasks=tasks, manifest_path=manifest_path)
    return jsonify({"run_id": run_id, "total": len(tasks)}), 202