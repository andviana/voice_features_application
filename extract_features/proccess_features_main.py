"""
Módulo de automação para extração de características de F0 em lote.

Este script varre um diretório em busca de arquivos .wav, processa cada áudio
utilizando as funções de extração definidas no pacote dp_voice_features_3p5
e exporta os resultados consolidados em um arquivo CSV.
"""

from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm

from modules.extract_features.f0_features import extract_f0_features
from modules.extract_features.formants_lpc import extract_formant_features
from modules.extract_features.hnr_features import extract_hnr_features
from modules.extract_features.jitter_features import extract_jitter_features
from modules.extract_features.shimmer_features import extract_shimmer_features
from modules.extract_features.mfcc_features import extract_mfcc_features
from modules.extract_features.spectral_features import extract_spectral_features
from modules.extract_features.tsallis_amplitude_hist import extract_tsallis_amplitude_features
from modules.extract_features.tsallis_f0_hist import extract_tsallis_f0_features

def processar_pasta_audios(caminho_pasta: str, arquivo_saida: str):
    """
    Varre a pasta e aplica múltiplos extratores em cada áudio, 
    consolidando tudo em um único CSV.

    Args:
        caminho_pasta (str): Caminho para o diretório contendo os arquivos de áudio.
        arquivo_saida (str): Caminho completo (incluindo nome) para o CSV de saída.

    Returns:
        None
    """
    pasta = Path(caminho_pasta).resolve() # Resolve para caminho absoluto

    if not pasta.exists():
        print(f"[ERRO] A pasta especificada não existe: {pasta}")
        return

    arquivos_wav = list(pasta.glob("*.wav"))

    if not arquivos_wav:
        print(f"[AVISO] Nenhum arquivo .wav encontrado em: {pasta}")
        return

    # LISTA DE EXTRATORES: Adicione aqui as novas funções conforme criarmos os arquivos
    # A estrutura é: "nome_amigavel": função_de_extracao
    extratores = {
        "F0": extract_f0_features,
        "Formantes": extract_formant_features,
        "HNR": extract_hnr_features,
        "Jitter": extract_jitter_features,
        "Shimmer": extract_shimmer_features,
        "MFCC": extract_mfcc_features,
        "Spectral": extract_spectral_features,
        # Usamos lambda para que o loop possa passar (y, sr) sem erro
        "Tsallis_Amp": lambda y, sr: extract_tsallis_amplitude_features(y),
        "Tsallis_F0": extract_tsallis_f0_features,
        
    }

    resultados = []
    print(f"Processando {len(arquivos_wav)} arquivos com {len(extratores)} extratores...")

    for caminho_arquivo in tqdm(arquivos_wav):
        try:
            # 1. Carregar o áudio
            # sr=None mantém a taxa de amostragem original do arquivo
            y, sr = librosa.load(caminho_arquivo, sr=None)

            # Registro base com o nome do arquivo
            registro_audio = {"file_name": caminho_arquivo.name}

            # 2. Executar todos os extratores mapeados
            for nome, funcao_extracao in extratores.items():
                try:
                    # Todas as suas funções seguem o padrão (y, sr) -> (stats, dict)
                    _, features = funcao_extracao(y, sr)
                    registro_audio.update(features) # Une os dicionários
                except Exception as e:
                    print(f" Falha no extrator {nome} para {caminho_arquivo.name}: {e}")

            resultados.append(registro_audio)


        except (RuntimeError, ValueError) as e:
            # Captura erros específicos do Librosa ou do seu extrator (ex: áudio curto demais)
            print(f"Erro de processamento em {caminho_arquivo.name}: {e}")

        except FileNotFoundError:
            print(f"Arquivo não encontrado: {caminho_arquivo.name}")

        except OSError as e:
            # Captura problemas de leitura de disco ou codecs corrompidos
            print(f"Erro de sistema/leitura em {caminho_arquivo.name}: {e}")

        except Exception as e:
            print(f" Erro crítico ao carregar {caminho_arquivo.name}: {e}")

    # 3. Consolidação e Salvamento
    if resultados:
        df = pd.DataFrame(resultados)

        # Garante que 'file_name' seja a primeira coluna
        cols = ["file_name"] + [c for c in df.columns if c != "file_name"]
        df = df[cols]

        saida = Path(arquivo_saida).resolve()
        saida.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(saida, index=False, encoding='utf-8')
        print(f"\nSucesso! Dataset gerado com {df.shape[1]} colunas em: {saida}")
    else:
        print("\n[ERRO] Nenhum dado extraído.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    # # Exemplo para a pasta PD_AH
    # PASTA_AUDIOS = BASE_DIR / "data" / "raw_audios" / "PD_AH"
    # ARQUIVO_CSV = BASE_DIR / "data" / "features" / "PD_AH" / "dataset_voz_completo.csv"

    # Exemplo para a pasta HC_AH
    PASTA_AUDIOS = BASE_DIR / "data" / "raw_audios" / "HC_AH"
    ARQUIVO_CSV = BASE_DIR / "data" / "features" / "HC_AH" / "dataset_voz_completo.csv"

    processar_pasta_audios(str(PASTA_AUDIOS), str(ARQUIVO_CSV))
