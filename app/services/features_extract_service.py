from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import librosa

from app.globals import params_singleton
from app.services.pipeline_service import pipeline_manager


ALLOWED_GROUPS = {"HC_AH", "PD_AH"}

def _manifest_path() -> Path:
    p = params_singleton.get()
    return (Path(p.path_demographics).resolve() / "manifest.csv").resolve()


def _load_manifest_map(manifest_csv: Path) -> Dict[str, str]:
    """
    Retorna {file_name: sex} para ajuste de faixas de F0.
    Se não achar, retorna dict vazio e seguimos com faixa default.
    """
    if not manifest_csv.exists():
        return {}
    
    try:
        df = pd.read_csv(manifest_csv)
        file_col = next((c for c in df.columns if c.strip().lower() in ("file_name", "filename", "wav")), None)
        sex_col = next((c for c in df.columns if c.strip().lower() in ("sex", "sexo", "gender")), None)
        
        if not file_col or not sex_col: return {}

        return {str(row[file_col]).strip(): str(row[sex_col]).strip().upper() for _, row in df.iterrows()}
    
    except:
        return {}

def start_extract_features_run(
    group: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Executa extração sobre os áudios processados. 
    Se group for None, processa todos os grupos em ALLOWED_GROUPS sequencialmente.
    """

    def job():
        # Imports locais para evitar dependências circulares e garantir captura de logs
        from extract_features.formants_lpc import extract_formant_features
        from extract_features.f0_features import extract_f0_features
        from extract_features.hnr_features import extract_hnr_features
        from extract_features.jitter_features import extract_jitter_features
        from extract_features.shimmer_features import extract_shimmer_features
        from extract_features.mfcc_features import extract_mfcc_features
        from extract_features.spectral_features import extract_spectral_features
        from extract_features.tsallis_amplitude_hist import extract_tsallis_amplitude_features
        from extract_features.tsallis_f0_hist import extract_tsallis_f0_features

        p = params_singleton.get()
        base_data = Path("data").resolve()
        manifest_csv = _manifest_path()
        sex_map = _load_manifest_map(manifest_csv)

        # Define quais grupos serão processados
        groups_to_process = [group] if group else sorted(list(ALLOWED_GROUPS))
        
        print(f"== Iniciando Extração de Características ==")
        print(f"Grupos alvo: {', '.join(groups_to_process)}")

        for current_group in groups_to_process:
            in_dir = (base_data / "audio_processed" / current_group).resolve()
            out_dir = (base_data / "features" / current_group).resolve()
            
            if not in_dir.exists():
                print(f"[AVISO] Pasta não encontrada, pulando: {in_dir}")
                continue

            # Seleção de arquivos
            if filename and group: # filename só faz sentido se um grupo específico foi passado
                wavs = [(in_dir / Path(filename).name).resolve()]
            else:
                wavs = sorted(in_dir.glob("*.wav"))

            if not wavs:
                print(f"[AVISO] Nenhum arquivo .wav encontrado em {current_group}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / (f"features_{Path(filename).stem}.csv" if filename else "dataset_voz_completo.csv")

            print(f"\n>>> Processando Grupo: {current_group} ({len(wavs)} arquivos) <<<")
            resultados = []

            for wav_path in wavs:
                if not wav_path.exists(): continue
                try:
                    print(f"Extraindo: {wav_path.name}")
                    y, sr = librosa.load(wav_path, sr=None)
                    
                    # Definição de limites de Pitch conforme sexo do manifest
                    sx = sex_map.get(wav_path.name, "").upper()
                    if sx == "M":
                        fmin, fmax = float(p.f_low_man), float(p.f_high_man)
                    else:
                        fmin, fmax = float(p.f_low_woman), float(p.f_high_woman)

                    registro = {"file_name": wav_path.name, "group": current_group}

                    # --- Pipeline de Extração de Features ---
                    _, d = extract_f0_features(y, sr, fmin_hz=fmin, fmax_hz=fmax)
                    registro.update(d)
                    _, d = extract_formant_features(y, sr)
                    registro.update(d)
                    _, d = extract_hnr_features(y, sr, min_pitch_hz=fmin)
                    registro.update(d)
                    _, d = extract_jitter_features(y, sr, fmin_hz=fmin, fmax_hz=fmax)
                    registro.update(d)
                    _, d = extract_shimmer_features(y, sr, fmin_hz=fmin, fmax_hz=fmax)
                    registro.update(d)
                    _, d = extract_mfcc_features(y, sr)
                    registro.update(d)
                    _, d = extract_spectral_features(y, sr)
                    registro.update(d)
                    
                    # --- Inovação: Entropia de Tsallis ---
                    _, d = extract_tsallis_amplitude_features(y, q=p.tsallis_q)
                    registro.update(d)
                    _, d = extract_tsallis_f0_features(y, sr, q=p.tsallis_q, fmin_hz=fmin, fmax_hz=fmax)
                    registro.update(d)

                    resultados.append(registro)
                except Exception as e:
                    print(f"  [ERRO] Falha em {wav_path.name}: {e}")

            # Salvamento do CSV do grupo
            if resultados:
                df = pd.DataFrame(resultados)
                # Reorganiza colunas para file_name e group virem primeiro
                cols = ["file_name", "group"] + [c for c in df.columns if c not in ("file_name", "group")]
                df[cols].to_csv(out_csv, index=False, encoding="utf-8")
                print(f"[OK] Grupo {current_group} finalizado. CSV salvo em: {out_csv.name}")

        print("\n== Extração Global Finalizada ==")

    return pipeline_manager.start(job)