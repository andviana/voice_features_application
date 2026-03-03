from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from app.globals import params_singleton
from app.services.pipeline_service import pipeline_manager



def _manifest_path() -> Path:
    p = params_singleton.get()
    return (Path(p.path_demographics).resolve() / "manifest.csv").resolve()


def _load_manifest_map(manifest_csv: Path) -> Dict[str, str]:
    """
    Retorna {file_name: sex}, tentando achar colunas típicas.
    Se não achar, retorna dict vazio e seguimos com faixa default.
    """
    if not manifest_csv.exists():
        return {}

    df = pd.read_csv(manifest_csv)

    # busca coluna com nome do arquivo
    file_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio"):
            file_col = c
            break

    sex_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("sex", "sexo", "gender"):
            sex_col = c
            break

    if not file_col or not sex_col:
        return {}

    m = {}
    for _, row in df.iterrows():
        fn = str(row[file_col]).strip()
        sx = str(row[sex_col]).strip().upper()
        if fn:
            m[fn] = sx
    return m

def start_extract_features_run(
    group: str,
    filename: Optional[str] = None,
) -> str:
    """
    Executa extração sobre data/audio_processed/<group> (ou apenas 1 arquivo) e salva CSV.
    Toda saída via print() aparece via SSE (PipelineManager).
    """

    def job():
        from extract_features.formants_lpc import extract_formant_features
        from extract_features.f0_features import extract_f0_features
        from extract_features.hnr_features import extract_hnr_features
        from extract_features.jitter_features import extract_jitter_features
        from extract_features.shimmer_features import extract_shimmer_features
        from extract_features.mfcc_features import extract_mfcc_features
        from extract_features.spectral_features import extract_spectral_features
        from extract_features.tsallis_amplitude_hist import extract_tsallis_amplitude_features
        from extract_features.tsallis_f0_hist import extract_tsallis_f0_features

        import librosa

        p = params_singleton.get()

        base_data = Path("data").resolve()
        in_dir = (base_data / "audio_processed" / group).resolve()
        if not in_dir.exists():
            print(f"[ERRO] Pasta não existe: {in_dir}")
            return

        wavs = []
        if filename:
            f = (in_dir / Path(filename).name).resolve()
            if not f.exists() or f.suffix.lower() != ".wav":
                print(f"[ERRO] Arquivo não encontrado: {f}")
                return
            wavs = [f]
        else:
            wavs = sorted(in_dir.glob("*.wav"))

        if not wavs:
            print(f"[AVISO] Nenhum .wav em: {in_dir}")
            return

        manifest_csv = _manifest_path()
        sex_map = _load_manifest_map(manifest_csv)

        out_dir = (base_data / "features" / group).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSV: se for arquivo único -> nome específico; se for pasta -> dataset completo
        out_csv = out_dir / (f"features_{Path(filename).stem}.csv" if filename else "dataset_voz_completo.csv")

        # extratores conforme seu main :contentReference[oaicite:7]{index=7}
        resultados = []
        print(f"== Extração iniciada ==")
        print(f"Entrada: {in_dir}")
        print(f"Manifest: {manifest_csv}")
        print(f"Saída CSV: {out_csv}")
        print(f"Arquivos: {len(wavs)}")

        for wav_path in wavs:
            try:
                print(f"\n--- Processando: {wav_path.name} ---")
                y, sr = librosa.load(wav_path, sr=None)

                # define faixa f0 por sexo (se existir no manifest), senão usa faixas do Params (mulher como default)
                sx = sex_map.get(wav_path.name, "").upper()
                if sx == "M":
                    fmin, fmax = float(p.f_low_man), float(p.f_high_man)
                elif sx == "F":
                    fmin, fmax = float(p.f_low_woman), float(p.f_high_woman)
                else:
                    fmin, fmax = float(p.f_low_woman), float(p.f_high_woman)

                registro = {"file_name": wav_path.name}

                # Chamadas SEM alterar os módulos, apenas passando parâmetros
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

                # Tsallis_Amp no seu main usa lambda y,sr: extract_tsallis_amplitude_features(y):contentReference[oaicite:8]{index=8}
                _, d = extract_tsallis_amplitude_features(y)
                registro.update(d)

                _, d = extract_tsallis_f0_features(y, sr, fmin_hz=fmin, fmax_hz=fmax)
                registro.update(d)

                resultados.append(registro)
                print(f"[OK] Features extraídas para {wav_path.name} ({len(registro)} campos)")

            except Exception as e:
                print(f"[ERRO] Falha em {wav_path.name}: {e}")

        if not resultados:
            print("\n[ERRO] Nenhum dado extraído.")
            return

        df = pd.DataFrame(resultados)
        cols = ["file_name"] + [c for c in df.columns if c != "file_name"]
        df = df[cols]
        df.to_csv(out_csv, index=False, encoding="utf-8")

        print(f"\n== Extração finalizada com sucesso ==")
        print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
        print(f"CSV: {out_csv}")

    return pipeline_manager.start(job)