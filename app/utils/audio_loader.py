import os
import librosa
import logging
from app.utils.path_utils import PathUtils


logger = logging.getLogger('TsallisOptimization')

def load_base_publica_8khz():
    """
    Carrega os 81 áudios da base pública divididos entre HC e PD.
    Localização esperada: data/audio_processed/
    """
    base_path = PathUtils.processed_root
    hc_dir = os.path.join(base_path, 'HC_AH')
    pd_dir = os.path.join(base_path, 'PD_AH')
    
    hc_signals = []
    pd_signals = []

    def _load_from_dir(directory):
        signals = []
        if not os.path.exists(directory):
            logger.warning(f"Diretório não encontrado: {directory}")
            return signals
            
        for file in os.listdir(directory):
            if file.endswith('.wav'):
                path = os.path.join(directory, file)
                try:
                    # Forçando sr=8000 conforme sua base pública atual
                    y, _ = librosa.load(path, sr=8000)
                    if y.size > 0:
                        signals.append(y)
                    else:
                        logger.warning(f"Arquivo vazio ou corrompido: {file}")
                except Exception as e:
                    logger.error(f"Erro ao carregar {file}: {str(e)}")
        return signals

    logger.info("Carregando sinais HC...")
    hc_signals = _load_from_dir(hc_dir)
    
    logger.info("Carregando sinais PD...")
    pd_signals = _load_from_dir(pd_dir)

    return hc_signals, pd_signals