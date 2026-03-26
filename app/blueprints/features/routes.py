from __future__ import annotations

import os
from flask import render_template, jsonify, current_app, send_file, abort
from pathlib import Path
import pandas as pd
from datetime import datetime

from typing import Optional
from . import bp
from app.services.features_extract_service import start_extract_features_run
from app.services.pipeline_service import pipeline_manager
from app.utils.path_utils import PathUtils


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
    features_dir = PathUtils.features_root()
    
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
    features_root = PathUtils.features_root()
    
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
                    "modified": datetime.fromtimestamp(stats.st_mtime).strftime('%d/%m/%Y %H:%M')
                })

    return render_template("features/list_files.html", files=files_info)


@bp.get("/features/download/<group>/<filename>")
def download_features(group: str, filename: str):
    """Permite baixar o CSV para análise no Excel/Pandas."""
    fpath = (PathUtils.features_root() / group / filename).resolve()
    
    if not fpath.exists():
        abort(404)
        
    return send_file(fpath, as_attachment=True)


@bp.get("/features/view-details/<group>/<filename>")
def view_dataset_details(group: str, filename: str):
    """
    Carrega o dataset e gera estatísticas descritivas completas.
    """
    fpath = (PathUtils.features_root() / group / filename).resolve()
    
    if not fpath.exists():
        abort(404)

    try:
        df = pd.read_csv(fpath)
        
        # Resumo básico
        summary = {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "filename": filename,
            "group": group
        }

        # Estatísticas Descritivas (Apenas colunas numéricas)
        numeric_df = df.select_dtypes(include=['number'])
        
        stats = numeric_df.agg([
            'mean', 'median', 'std', 'var', 'min', 'max'
        ]).transpose()

        # Adicionando Quartis específicos
        stats['Q1'] = numeric_df.quantile(0.25)
        stats['Q2'] = numeric_df.quantile(0.5)
        stats['Q3'] = numeric_df.quantile(0.75)
        stats['IQR'] = stats['Q3'] - stats['Q1']

        # Cálculo da Moda (Pandas retorna série, pegamos o primeiro valor)
        stats['mode'] = numeric_df.mode().iloc[0]

        # Renomear colunas para o template
        stats.columns = [
            'Média', 'Mediana', 'Desvio Padrão', 'Variância', 
            'Mínimo', 'Máximo', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'IQR', 'Moda'
        ]
        
        # Converter para dicionário para facilitar a iteração no Jinja2
        stats_dict = stats.to_dict('index')

        return render_template(
            "features/view_details.html",
            summary=summary,
            stats=stats_dict,
            columns=df.columns.tolist(),
            data=df.head(100).values.tolist()
        )
    except Exception as e:
        return f"Erro ao processar o dataset: {str(e)}", 500