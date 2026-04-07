from flask import render_template, jsonify, send_file
from app.services.tsallis_optimization_service import TsallisOptimizationService
from app.services.data_consolidation_service import DataConsolidationService
from app.utils.audio_loader import load_base_publica_8khz
import os
from datetime import datetime
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

@bp.route('/download-consolidated-xlsx', methods=['GET'])
def download_consolidated_xlsx():
    service = DataConsolidationService()
    try:
        df = service.consolidate()
        xlsx_file = service.generate_xlsx(df)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consolidado_voz_{timestamp}.xlsx"
        
        return send_file(
            xlsx_file,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return f"Erro ao gerar planilha: {str(e)}", 500

@bp.route('/view-consolidated-data', methods=['GET'])
def view_consolidated_data():
    service = DataConsolidationService()
    try:
        df = service.consolidate()
        columns = df.columns.tolist()
        records = df.to_dict(orient='records')
        
        return render_template(
            'group_comparison/consolidated_table.html',
            columns=columns,
            records=records
        )
    except Exception as e:
        return f"Erro ao carregar dados: {str(e)}", 500
