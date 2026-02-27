import soundfile as sf
import librosa
import pandas as pd
import os

from pre_proccess import analise, filters, normalize, windowing


# Caminho para o arquivo de metadados
MANIFEST_PATH = 'manifest.csv'

def get_sex_from_manifest(recording_id):
    """Busca o sexo do paciente no arquivo manifest.csv."""
    try:
        df = pd.read_csv(MANIFEST_PATH)
        row = df[df['recording_id'] == recording_id]
        if not row.empty:
            return row.iloc[0]['sex']
    except Exception as e:
        print(f"Erro ao ler manifest: {e}")
    return 'Unknown'

def executar_pipeline(input_path, output_path, target_sr=44100, duration=2.0):
    # 1. Carregamento original
    y, sr = librosa.load(input_path, sr=None)
    
    # Identificação do áudio para busca no metadado
    recording_id = os.path.basename(input_path).replace('.wav', '')
    sexo = get_sex_from_manifest(recording_id)

    # 2. Sequência Metodológica Shen 2025 + Tsallis
    y = normalize.resample_audio(y, sr, target_sr)
    y = filters.remove_dc_offset(y)

    # y = filters.apply_bandpass(y, target_sr)
    # Filtra conforme floor e ceiling por sexo 
    y = filters.apply_shen_filter(y, target_sr, sexo)

    y = normalize.scale_amplitude(y) # Normalização [-1, 1]
    y_final = windowing.trim_and_segment(y, target_sr, duration)
    
    # 3. Salva o áudio processado
    sf.write(output_path, y_final, target_sr)
    
    # 4. Log de consistência
    info = analise.analisar_amostra(y_final, target_sr)
    info['sexo_aplicado'] = sexo
    return info

if __name__ == "__main__":
    # Exemplo de teste com um arquivo
    # info = executar_pipeline("data/raw/PD/exemplo.wav", "data/processed/PD/exemplo.wav")
    # print(info)
    print("Pipeline pronto para processamento em lote.")