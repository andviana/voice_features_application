from flask import render_template, jsonify, abort, send_from_directory
from app.services.tsallis_optimization_service import TsallisOptimizationService
from app.utils.audio_loader import load_base_publica_8khz
import os
import pandas as pd
from werkzeug.utils import secure_filename
from . import bp


@bp.route('/tsallis-optimization-protocol', methods=['GET'])
def tsallis_optimization_page():
    """Renderiza a página dedicada ao protocolo de otimização."""
    service = TsallisOptimizationService()
    results = service.get_latest_results()
    return render_template('optimize_tsallis/tsallis_protocol.html', results=results)

@bp.route('/run-tsallis-optimization', methods=['POST'])
def run_tsallis_optimization():
    """
    Executa o protocolo de 3 métodos e salva os logs/resultados.
    """
    service = TsallisOptimizationService()
    try:
        # 1. Carregamento modular dos áudios (utils/audio_loader.py)
        # Base pública de 81 áudios em 8kHz
        (hc_signals, hc_names), (pd_signals, pd_names) = load_base_publica_8khz()

        print(f"HC carregados: {len(hc_signals)}")
        print(f"PD carregados: {len(pd_signals)}")

        if not hc_signals or not pd_signals:
            return jsonify({
                "status": "error",
                "message": "Nenhum sinal de voz encontrado nas pastas HC_AH ou PD_AH."
            }), 400        

        # 2. Execução do Protocolo Científico:
        # - Grid Search (Máxima separação t-stat)
        # - Extensividade (Linearidade R²)
        # - q-Gaussian Fit (RMSE do ajuste de densidade)
        summary_path = service.run_full_optimization(hc_signals, pd_signals)
        
        # 3. Retorno dos caminhos para acesso aos artefatos CSV e Logs
        return jsonify({
            "status": "success",
            "message": "Protocolo concluído com sucesso.",
            "data": {
                "csv_summary": os.path.basename(summary_path),
                "log_file": os.path.basename(service.log_file),
                "timestamp": service.logger.name
            }
        })

    except Exception as e:
        # O log detalhado já é capturado dentro do service
        return jsonify({
            "status": "error", 
            "message": f"Falha no processamento: {str(e)}"
        }), 500
    

@bp.route('/download-optimization-file/<filename>')
def download_opt_file(filename):
    service = TsallisOptimizationService()
    return send_from_directory(service.output_dir, filename)


@bp.route('/view-csv-table/<filename>')
def view_csv_table(filename):
    service = TsallisOptimizationService()
    safe_filename = secure_filename(filename)
    path = os.path.join(service.output_dir, safe_filename)

    if not os.path.exists(path):
        abort(404)

    df = pd.read_csv(path).fillna('')

    records = df.to_dict(orient='records')
    columns = df.columns.tolist()

    return render_template(
        'optimize_tsallis/csv_table_view.html',
        filename=filename,
        columns=columns,
        records=records,
        row_count=len(records),

    )

@bp.route('/group-comparison', methods=['GET'])
def group_comparison_page():
    service = TsallisOptimizationService()
    results = service.get_latest_group_comparison_results()
    return render_template(
        'optimize_tsallis/group_comparison.html',
        results=results
    )


@bp.route('/run-group-comparison', methods=['POST'])
def run_group_comparison():
    service = TsallisOptimizationService()

    try:
        hc_data, pd_data = load_base_publica_8khz()

        hc_signals, _ = hc_data
        pd_signals, _ = pd_data

        if not hc_signals or not pd_signals:
            return jsonify({
                "status": "error",
                "message": "Nenhum sinal de voz encontrado nas pastas HC_AH ou PD_AH."
            }), 400

        result = service.run_group_comparison(hc_data, pd_data)

        return jsonify({
            "status": "success",
            "message": "Comparação por grupos concluída com sucesso.",
            "data": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Falha no processamento: {str(e)}"
        }), 500