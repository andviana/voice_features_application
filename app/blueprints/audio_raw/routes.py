from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from flask import render_template, jsonify, send_file, Response, stream_with_context

from app.utils.audio_props import read_wav_props, SOUNDFILE_ERRORS
from app.utils.path_utils import PathUtils
from app.services.preproccess_service import start_preprocess_run, start_preprocess_batch_run
from . import bp


@bp.get("/audio-raw")
def list_audio_raw():
    """Lista ficheiros brutos utilizando PathUtils e AudioProps centralizados."""
    raw_root = PathUtils.raw_root()    
    items, counts = [], {}

    # Grupos permitidos centralizados no PathUtils
    for group in sorted(PathUtils.ALLOWED_GROUPS):
        try:
            gdir = PathUtils.safe_group_dir(raw_root, group)
            # Ordena por nome de ficheiro
            wavs = sorted(gdir.glob("*.wav"), key=lambda p: p.name.lower())
            counts[group] = len(wavs)

            for w in wavs:
                # Usa a lógica robusta com fallback para erros de leitura
                items.append(read_wav_props(w, group, error_fallback=True))

        except Exception:
            counts[group] = 0
    
    return render_template("audio_raw/list.html", items=items, counts=counts, base=str(raw_root))


@bp.get("/audio-raw/play/<group>/<path:filename>")
def play_audio(group: str, filename: str):
    """Reproduz áudio usando validação segura de caminhos."""
    fpath = PathUtils.safe_wav_path(PathUtils.raw_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


@bp.get("/audio-raw/info-json/<group>/<path:filename>")
def audio_info_json(group: str, filename: str):
    """Retorna JSON com propriedades técnicas para o modal."""
    fpath = PathUtils.safe_wav_path(PathUtils.raw_root(), group, filename)
    props = read_wav_props(fpath, group)
    return jsonify(asdict(props))


@bp.get("/audio-raw/info/<group>/<path:filename>")
def audio_info(group: str, filename: str):
    """Página HTML de detalhes do áudio bruto."""
    fpath = PathUtils.safe_wav_path(PathUtils.raw_root(), group, filename)
    props = read_wav_props(fpath, group)
    return render_template(
        "audio_raw/info.html", 
        props=props, 
        props_dict=asdict(props),
        group=group,        # Adicione esta linha
        filename=filename   # Adicione esta linha
    )


@bp.post("/audio-raw/preprocess/<group>/<path:filename>")
def preprocess_audio(group: str, filename: str):
    """Inicia o pré-processamento de um ficheiro individual."""
    input_path = PathUtils.safe_wav_path(PathUtils.raw_root(), group, filename)
    output_path = PathUtils.processed_root() / group / input_path.name

    run_id = start_preprocess_run(
        input_wav_path=input_path,
        output_wav_path=output_path,
    )
    return jsonify({"run_id": run_id}), 202


@bp.post("/audio-raw/preprocess-batch")
def preprocess_batch():
    """Execução em lote"""
    tasks = []
    for group in sorted(PathUtils.ALLOWED_GROUPS):
        gdir = PathUtils.safe_group_dir(PathUtils.raw_root(), group)
        for w in gdir.glob("*.wav"):
            output = PathUtils.processed_root() / group / w.name
            tasks.append((w, output))

    run_id = start_preprocess_batch_run(tasks=tasks)
    return jsonify({"run_id": run_id, "total": len(tasks)}), 202


@bp.get("/audio-raw/preprocess/stream/<run_id>")
def preprocess_stream(run_id: str):
    """Mantém o streaming SSE para feedback em tempo real."""
    from app.services.pipeline_service import pipeline_manager
    run = pipeline_manager.get_run(run_id)
    if not run: 
        return jsonify({"error": "run_not_found"}), 404

    def event_stream():
        yield "retry: 1000\n\n"
        while True:
            line = run.q.get() # O PipelineManager lida com o timeout internamente
            if line == "__PIPELINE_DONE__":
                yield "event: done\ndata: preprocess_finished\n\n"
                break
            
            yield f"data: {str(line).replace(os.linesep, '')}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@bp.post("/audio-raw/clear-processing")
def clear_processing():
    """Exclui as pastas de processamento: audio_processed, features, optimization_results."""
    paths_to_clear = [
        PathUtils.processed_root(),
        PathUtils.features_root(),
        PathUtils.data_root() / "optimization_results"
    ]
    
    cleared = []
    errors = []
    
    for p in paths_to_clear:
        if p.exists():
            try:
                shutil.rmtree(p)
                cleared.append(p.name)
            except Exception as e:
                errors.append(f"Erro ao excluir {p.name}: {str(e)}")
    
    if errors:
        return jsonify({"status": "error", "message": "Executado com erros.", "errors": errors}), 500
        
    return jsonify({
        "status": "success", 
        "message": f"Processamento excluído com sucesso: {', '.join(cleared) if cleared else 'Nenhuma pasta encontrada.'}"
    })


