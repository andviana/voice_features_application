from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from ...repositories.params_repository import ParamsRepository
from ...services.params_service import ParamsService
from ...globals import params_singleton
from . import bp


service = ParamsService(ParamsRepository())

def _format_params_for_table(obj):
    # Formatação técnica (unidades)
    return [
        {"key": "SR", "value": f"{obj.SR:,.0f}", "unit": "Hz", "desc": "Taxa de amostragem"},
        {"key": "duration", "value": f"{obj.duration:.3f}", "unit": "s", "desc": "Duração alvo"},
        {"key": "f_low_woman", "value": f"{obj.f_low_woman:.2f}", "unit": "Hz", "desc": "F0 mínimo (mulher)"},
        {"key": "f_high_woman", "value": f"{obj.f_high_woman:.2f}", "unit": "Hz", "desc": "F0 máximo (mulher)"},
        {"key": "f_low_man", "value": f"{obj.f_low_man:.2f}", "unit": "Hz", "desc": "F0 mínimo (homem)"},
        {"key": "f_high_man", "value": f"{obj.f_high_man:.2f}", "unit": "Hz", "desc": "F0 máximo (homem)"},
        {"key": "target_db", "value": f"{obj.target_db:.1f}", "unit": "dB", "desc": "Normalização de loudness alvo"},
        {"key": "path_audio", "value": obj.path_audio, "unit": "", "desc": "Caminho base para áudio"},
        {"key": "path_demographics", "value": obj.path_demographics, "unit": "", "desc": "Caminho base para metadados"},
    ]

@bp.get("/params")
def edit_params():
    obj = service.get()
    return render_template("params/edit.html", p=obj)


@bp.post("/params")
def update_params():
    data = {
        "SR": float(request.form["SR"]),
        "duration": float(request.form["duration"]),
        "f_low_woman": float(request.form["f_low_woman"]),
        "f_high_woman": float(request.form["f_high_woman"]),
        "f_low_man": float(request.form["f_low_man"]),
        "f_high_man": float(request.form["f_high_man"]),
        "target_db": float(request.form["target_db"]),
        "path_audio": request.form["path_audio"],
        "path_demographics": request.form["path_demographics"],
    }
    service.update(data)
    flash("Parâmetros atualizados e singleton sincronizado.", "success")
    return redirect(url_for("params.edit_params"))


@bp.get("/params/view")
def view_params_table():
    obj = service.get()
    rows = _format_params_for_table(obj)
    return render_template("params/view_table.html", rows=rows)


# Endpoint útil para o pipeline (snapshot atual do singleton)
@bp.get("/params/snapshot")
def params_snapshot():
    snap = params_singleton.get()
    return jsonify({
        "SR": snap.SR,
        "duration": snap.duration,
        "f_low_woman": snap.f_low_woman,
        "f_high_woman": snap.f_high_woman,
        "f_low_man": snap.f_low_man,
        "f_high_man": snap.f_high_man,
        "target_db": snap.target_db,
        "path_audio": snap.path_audio,
        "path_demographics": snap.path_demographics,
    })