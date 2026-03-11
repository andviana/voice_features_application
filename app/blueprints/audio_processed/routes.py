from __future__ import annotations

import os
from dataclasses import asdict
from flask import current_app, render_template, jsonify, send_file

from app.utils.audio_props import read_wav_props
from app.utils.path_utils import PathUtils
from app.services.metadata_service import MetadataService
from app.services.audio_signal_service import AudioSignalsService
from . import bp


@bp.get("/audio-processed")
def list_audios_processed():
    """Lista áudios processados utilizando utilitários centralizados."""
    base = PathUtils.processed_root()    
    items = []
    counts = {}

    for group in sorted(PathUtils.ALLOWED_GROUPS):
        try:
            gdir = PathUtils.safe_group_dir(base, group)
            wavs = sorted(gdir.glob("*.wav"), key=lambda p: p.name.lower())
            counts[group] = len(wavs)

            for w in wavs:
                # Usa fallback para props caso o ficheiro esteja inacessível
                items.append(read_wav_props(w, group, error_fallback=True))
            
        except Exception:
                counts[group] = 0
    
    # Verificação de integridade dos datasets de features
    features_dir = PathUtils.data_root() / PathUtils.FEATURES_ROOT
    hc_ready = (features_dir / "HC_AH" / PathUtils.FEATURES_FILE).exists()
    pd_ready = (features_dir / "PD_AH" / PathUtils.FEATURES_FILE).exists()

    return render_template(
        "audio_processed/list.html",
        items=items,
        counts=counts,
        base=str(base),
        features_ready=(hc_ready and pd_ready)
    )


@bp.get("/audio-processed/play/<group>/<path:filename>")
def play_audio_processed(group: str, filename: str):
    """Reprodução segura de áudio processado."""
    fpath = PathUtils.safe_wav_path(PathUtils.processed_root(), group, filename)
    return send_file(fpath, mimetype="audio/wav", as_attachment=False)


@bp.get("/audio-processed/info/<group>/<path:filename>")
def audio_processed_info(group: str, filename: str):
    """
    Retorna JSON consolidado para o modal de info:
    - Propriedades técnicas (AudioProps)
    - Dados demográficos (MetadataService)
    - Gráficos (AudioSignalsService)
    """
    fpath = PathUtils.safe_wav_path(PathUtils.processed_root(), group, filename)    
    
    try:
        # 1. Carregamento do sinal
        y, sr = AudioSignalsService.load_audio(fpath)

        # 2. Resposta
        props = asdict(read_wav_props(fpath, group))
        demographics = MetadataService.load_manifest_row(fpath.name)
        waveform = AudioSignalsService.waveform(y, sr)
        spectrum = AudioSignalsService.spectrum(y, sr)

        return jsonify({
            "props": props,
            "demographics": demographics,
            "plots": {
                "sr": sr,
                "waveform":waveform,
                "spectrum":spectrum
            }
        })
    except Exception as e:
        current_app.logger.error(f"Erro ao processar info de áudio: {str(e)}")
        return jsonify({"error": "Falha ao processar dados do áudio"}), 500