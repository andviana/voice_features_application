import sys
from pathlib import Path
from typing import Dict, Optional

import librosa
import pandas as pd
from tqdm import tqdm

# Adiciona o diretório raiz ao sys.path para permitir importações do pacote 'app'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from .f0_features import extract_f0_features
from .formants_lpc import extract_formant_features
from .hnr_features import extract_hnr_features
from .jitter_features import extract_jitter_features
from .shimmer_features import extract_shimmer_features
from .mfcc_features import extract_mfcc_features
from .spectral_features import extract_spectral_features
from .tsallis_amplitude_hist import extract_tsallis_amplitude_features
from .tsallis_f0_hist import extract_tsallis_f0_features

def _load_manifest_map(manifest_csv: Path) -> Dict[str, str]:
    """Retorna {file_name: sex} para ajuste de faixas de F0."""
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

def processar_pasta_audios(caminho_pasta: str, arquivo_saida: str):
    """
    Varre a pasta e aplica múltiplos extratores em cada áudio, 
    consolidando tudo em um único CSV, usando parâmetros do sistema.
    """
    from app import create_app
    from app.services.params_service import ParamsService
    from app.repositories.params_repository import ParamsRepository

    # Inicializa contexto do app para acessar o banco de dados
    app = create_app()
    with app.app_context():
        service = ParamsService(ParamsRepository())
        p = service.get()
        print(f"[INFO] Parâmetros carregados: tsallis_q={p.tsallis_q}, SR={p.SR}")

        manifest_path = (Path(p.path_demographics).resolve() / "manifest.csv").resolve()
        sex_map = _load_manifest_map(manifest_path)
        if sex_map:
            print(f"[INFO] Manifest carregado com {len(sex_map)} entradas.")
        else:
            print(f"[AVISO] Manifest não encontrado em {manifest_path}. Usando faixas default.")

    pasta = Path(caminho_pasta).resolve()
    if not pasta.exists():
        print(f"[ERRO] A pasta especificada não existe: {pasta}")
        return

    arquivos_wav = list(pasta.glob("*.wav"))
    if not arquivos_wav:
        print(f"[AVISO] Nenhum arquivo .wav encontrado em: {pasta}")
        return

    resultados = []
    print(f"Processando {len(arquivos_wav)} arquivos...")

    for caminho_arquivo in tqdm(arquivos_wav):
        try:
            y, sr = librosa.load(caminho_arquivo, sr=None)
            registro_audio = {"file_name": caminho_arquivo.name}

            # Determinação dinâmica de fmin/fmax por gênero
            sex = sex_map.get(caminho_arquivo.name, "").upper()
            if sex == "M":
                fmin, fmax = float(p.f_low_man), float(p.f_high_man)
            else:
                # Default para Mulher ou se não encontrado
                fmin, fmax = float(p.f_low_woman), float(p.f_high_woman)

            # --- Execução dos Extratores ---
            # F0
            _, d = extract_f0_features(y, sr, fmin_hz=fmin, fmax_hz=fmax)
            registro_audio.update(d)
            # Formantes
            _, d = extract_formant_features(y, sr)
            registro_audio.update(d)
            # HNR
            _, d = extract_hnr_features(y, sr, min_pitch_hz=fmin)
            registro_audio.update(d)
            # Jitter
            _, d = extract_jitter_features(y, sr, fmin_hz=fmin, fmax_hz=fmax)
            registro_audio.update(d)
            # Shimmer
            _, d = extract_shimmer_features(y, sr, fmin_hz=fmin, fmax_hz=fmax)
            registro_audio.update(d)
            # MFCC
            _, d = extract_mfcc_features(y, sr)
            registro_audio.update(d)
            # Spectral
            _, d = extract_spectral_features(y, sr)
            registro_audio.update(d)
            # Tsallis Amp
            _, d = extract_tsallis_amplitude_features(y, q=p.tsallis_q)
            registro_audio.update(d)
            # Tsallis F0
            _, d = extract_tsallis_f0_features(y, sr, q=p.tsallis_q, fmin_hz=fmin, fmax_hz=fmax)
            registro_audio.update(d)

            resultados.append(registro_audio)

        except Exception as e:
            print(f" [ERRO] Falha em {caminho_arquivo.name}: {e}")

    # Consolidação e Salvamento
    if resultados:
        df = pd.DataFrame(resultados)
        cols = ["file_name"] + [c for c in df.columns if c != "file_name"]
        df = df[cols]
        saida = Path(arquivo_saida).resolve()
        saida.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(saida, index=False, encoding='utf-8')
        print(f"\nSucesso! Dataset gerado em: {saida}")
    else:
        print("\n[ERRO] Nenhum dado extraído.")

if __name__ == "__main__":
    # Exemplo para a pasta HC_AH
    BASE_DIR = Path(__file__).resolve().parent.parent
    PASTA_AUDIOS = BASE_DIR / "data" / "audio_processed" / "HC_AH"
    ARQUIVO_CSV = BASE_DIR / "data" / "features" / "HC_AH" / "dataset_voz_completo.csv"

    processar_pasta_audios(str(PASTA_AUDIOS), str(ARQUIVO_CSV))
