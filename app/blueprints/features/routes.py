from __future__ import annotations

import os
from flask import render_template, jsonify, current_app, send_file, abort
from pathlib import Path

from typing import Optional
from . import bp
from app.services.features_extract_service import start_extract_features_run
from app.services.pipeline_service import pipeline_manager


@bp.post("/features/extract-all")
@bp.post("/features/extract/<group>")
def extract_features_group(group: Optional[str] = None):
    """
    Se group for None, start_extract_features_run processará todos.
    """
    run_id = start_extract_features_run(group=group, filename=None)
    return jsonify({"run_id": run_id}), 202


@bp.post("/features/extract/<group>/<path:filename>")
def extract_features_file(group: str, filename: str):
    run_id = start_extract_features_run(group=group, filename=filename)
    return jsonify({"run_id": run_id}), 202


@bp.get("/features/stream/<run_id>")
def features_stream(run_id: str):
    run = pipeline_manager.get_run(run_id)
    if run is None:
        return jsonify({"error": "run_not_found"}), 404

    def event_stream():
        yield "retry: 1000\n\n"
        while True:
            try:
                line = run.q.get(timeout=120)
            except Exception:
                yield ": keep-alive\n\n"
                continue

            if line == "__PIPELINE_DONE__":
                yield "event: done\ndata: features_finished\n\n"
                break

            safe = str(line).replace("\r", "")
            yield f"data: {safe}\n\n"

    from flask import Response, stream_with_context
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@bp.route('/status-files', methods=['GET'])
def get_features_status():
    """
    Verifica se os arquivos de features extraídos já existem para ambos os grupos.
    Localização: data/features/HC_AH e data/features/PD_AH
    """
    # Define a base do projeto (ajuste se a estrutura for diferente)
    base_dir = Path(current_app.root_path).parent
    features_dir = base_dir / "data" / "features"
    
    groups = ["HC_AH", "PD_AH"]
    status = {}
    
    # O nome do arquivo que seu service gera (ajuste se for outro nome)
    target_filename = "dataset_voz_completo.csv"

    for group in groups:
        group_path = features_dir / group
        file_path = group_path / target_filename
        
        status[group] = {
            "exists": file_path.exists(),
            "path": str(file_path),
            "size_kb": round(file_path.stat().st_size / 1024, 2) if file_path.exists() else 0,
            "last_modified": os.path.getmtime(file_path) if file_path.exists() else None
        }

    return jsonify({
        "ready": all(group["exists"] for group in status.values()),
        "groups": status
    })


@bp.get("/features/list")
def list_features():
    """Lista os arquivos CSV gerados pela extração de características."""
    base_data = Path(current_app.config["DATA_DIR"]).resolve()
    features_root = base_data / "features"
    
    files_info = []
    allowed_groups = ["HC_AH", "PD_AH"]

    for group in allowed_groups:
        group_dir = features_root / group
        if group_dir.exists():
            for csv_file in group_dir.glob("*.csv"):
                stats = csv_file.stat()
                files_info.append({
                    "group": group,
                    "name": csv_file.name,
                    "path": f"{group}/{csv_file.name}",
                    "size": round(stats.st_size / 1024, 2),
                    "modified": stats.st_mtime
                })

    return render_template("features/list_files.html", files=files_info)

@bp.get("/features/download/<group>/<filename>")
def download_features(group: str, filename: str):
    """Permite baixar o CSV para análise no Excel/Pandas."""
    base_data = Path(current_app.config["DATA_DIR"]).resolve()
    fpath = (base_data / "features" / group / filename).resolve()
    
    if not fpath.exists():
        abort(404)
        
    return send_file(fpath, as_attachment=True)