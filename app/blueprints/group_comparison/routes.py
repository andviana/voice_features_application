from flask import render_template, jsonify
from app.services.tsallis_optimization_service import TsallisOptimizationService
from app.utils.audio_loader import load_base_publica_8khz
import os
from . import bp

@bp.route('/group-comparison', methods=['GET'])
def group_comparison_page():
    service = TsallisOptimizationService()
    results = service.get_latest_group_comparison_results()
    return render_template(
        'group_comparison/group_comparison.html',
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
