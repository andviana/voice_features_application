from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from flask import current_app, render_template, abort, send_file, jsonify, Response, stream_with_context
from app.utils.audio_props import read_wav_props, SOUNDFILE_ERRORS
from app.services.preproccess_service import start_preprocess_run, start_preprocess_batch_run
from . import bp

ALLOWED_GROUPS = {"HC_AH", "PD_AH"}

def _get_paths_config():
    """Centraliza as configurações de diretórios do app."""
    return {
        "raw_root": Path(current_app.config["AUDIO_RAW_DIR"]).resolve(),
        "data_root": Path(current_app.config["DATA_DIR"]).resolve(),
    }


def _resolve_output_path(group: str, filename: str) -> Path:
    """Define onde o arquivo processado será salvo (item 4.1)."""
    cfg = _get_paths_config()
    return (cfg["data_root"] / "audio_processed" / group / filename).resolve()


def _safe_group_dir(group: str) -> Path:
    if group not in ALLOWED_GROUPS:
        abort(404)
    base = _get_paths_config()["raw_root"]
    d = (base / group).resolve()
    if base not in d.parents and d != base:
        abort(400)
    return d


def _safe_wav_path(group: str, filename: str) -> Path:
    gdir = _safe_group_dir(group)
    fpath = (gdir / os.path.basename(filename)).resolve()
    if not fpath.exists() or fpath.suffix.lower() != ".wav":
        abort(404)
    return fpath


def _iter_wavs(group_dir: Path) -> Iterable[Path]:
    # ordena por nome
    return sorted(_safe_group_dir(group_dir).glob("*.wav"), key=lambda p: p.name.lower())


@bp.get("/audio-raw")
def list_audio_raw():
    cfg = _get_paths_config()
    items, counts = [], {}
    for group in sorted(ALLOWED_GROUPS):
        wavs = _iter_wavs(group)
        counts[group] = len(wavs)
        for w in wavs:
            try:
                items.append(read_wav_props(w, group))
            except SOUNDFILE_ERRORS:
                items.append(read_wav_props(w, group, error_fallback=True)) # Assumindo helper para fallback
    return render_template("audio_raw/list.html", items=items, counts=counts, base=str(cfg["raw_root"]))




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


# página HTML de info
@bp.get("/audio-raw/info/<group>/<path:filename>")
def audio_info(group: str, filename: str):
    fpath = _safe_wav_path(group, filename)
    props = read_wav_props(fpath, group)
    return render_template("audio_raw/info.html", props=props, props_dict=asdict(props))


@bp.post("/audio-raw/preprocess/<group>/<path:filename>")
def preprocess_audio(group: str, filename: str):
    """Execução única utilizando o service refatorado."""
    input_path = _safe_wav_path(group, filename)
    output_path = _resolve_output_path(group, input_path.name)
    cfg = _get_paths_config()

    run_id = start_preprocess_run(
        input_wav_path=input_path,
        output_wav_path=output_path,
    )
    return jsonify({"run_id": run_id}), 202


@bp.post("/audio-raw/preprocess-batch")
def preprocess_batch():
    """Execução em lote unificada."""
    cfg = _get_paths_config()
    tasks: list[tuple[Path, Path]] = []

    for group in sorted(ALLOWED_GROUPS):
        gdir = _safe_group_dir(group)
        for w in gdir.glob("*.wav"):
            tasks.append((w, _resolve_output_path(group, w.name)))

    run_id = start_preprocess_batch_run(tasks=tasks)
    return jsonify({"run_id": run_id, "total": len(tasks)}), 202



@bp.get("/audio-raw/preprocess/stream/<run_id>")
def preprocess_stream(run_id: str):
    """Mantém o streaming SSE para feedback em tempo real."""
    from app.services.pipeline_service import pipeline_manager
    run = pipeline_manager.get_run(run_id)
    if not run: return jsonify({"error": "run_not_found"}), 404

    def event_stream():
        yield "retry: 1000\n\n"
        while True:
            line = run.q.get() # O PipelineManager lida com o timeout internamente
            if line == "__PIPELINE_DONE__":
                yield "event: done\ndata: preprocess_finished\n\n"
                break
            yield f"data: {str(line).replace(os.linesep, '')}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


