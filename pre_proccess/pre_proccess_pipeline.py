import librosa
import soundfile as sf
import os

from .analise import analisar_amostra
from .filters import remove_dc_offset, apply_bandpass
from .normalize import ensure_mono, scale_amplitude
from .windowing import remove_silence_adaptive, get_stable_segment


def executar_pipeline(input_path, output_path, target_sr=48000, duration=2.5):
    """
    Executa a sequência 4.1 do Plano de Pesquisa PPGEE/UFPA.
    Pipeline estrito conforme Plano de Pesquisa 4.1.
    Nota: Frequência padrão de 48kHz para o microfone Samson Q2U.
    """
    # 1. Carregamento e Conversão para Mono (Item 4.1.1)
    y, sr = librosa.load(input_path, sr=target_sr, mono=False)
    y = ensure_mono(y)
    
    # 2. Remoção de DC Offset (Item 4.1.2)
    y = remove_dc_offset(y)
    
    # 3. Filtragem Passa-Banda 80-8000 Hz (Item 4.1.3)
    y = apply_bandpass(y, target_sr)
    
    # 4. Remoção de Silêncio Adaptativa (Item 4.1.4)
    y = remove_silence_adaptive(y, target_sr)
    
    # 5. Normalização de Amplitude [-1, 1] (Item 4.1.5)
    y = scale_amplitude(y)
    
    # 6. Seleção de Segmento Estável Central (Item 4.1.6)
    # Definido 2.5s como meio-termo do intervalo 2-3s solicitado
    y_final = get_stable_segment(y, target_sr, duration)
       
    # Salva o áudio processado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_final, target_sr)
    
    # 4. Log de consistência
    info = analisar_amostra(y_final, target_sr)
    return info

if __name__ == "__main__":
    # info = executar_pipeline("data/raw/PD/exemplo.wav", "data/processed/PD/exemplo.wav")
    # print(info)
    print("Pipeline pronto para processamento em lote.")