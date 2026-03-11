from __future__ import annotations
import os
import soundfile as sf
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path



# Lista de erros comuns para facilitar o try/except nos blueprints
SOUNDFILE_ERRORS = (RuntimeError, ValueError, TypeError)



@dataclass(frozen=True)
class AudioProps:
    group: str
    filename: str
    rel_id: str  # "HC_AH/arquivo.wav"
    size_bytes: int
    modified_at: str
    n_channels: int
    sample_rate_hz: int
    n_frames: int
    sample_width_bytes: int
    duration_sec: float
    comptype: str
    compname: str


def read_wav_props(path: Path, group: str, error_fallback: bool = False) -> AudioProps:
    """
    Lê propriedades técnicas de um arquivo WAV de forma robusta.
    Implementa a lógica de mapeamento de bits baseada no subtipo do SoundFile.
    """
    path = Path(path)
    st = path.stat()
    modified = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    try:
        info = sf.info(str(path))
        
        n_channels = int(info.channels)
        sr = int(info.samplerate)
        n_frames = int(info.frames)
        duration = float(info.duration) if info.duration is not None else (n_frames / float(sr) if sr else 0.0)
        
        compname = str(info.subtype) if info.subtype else ""
        comptype = str(info.format) if info.format else ""

        # Mapeamento robusto de bits por subtipo (extraído de scientific_analysis)
        subtype_bits_map = {
            "PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, 
            "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64,
        }
        bits = subtype_bits_map.get(compname, 0)
        sample_width_bytes = bits // 8 if bits else 0

    except Exception as e:
        if not error_fallback:
            raise e
        # Fallback para arquivos corrompidos ou ilegíveis
        return AudioProps(
            group=group, filename=path.name, rel_id=path.name,
            size_bytes=st.st_size, modified_at=modified,
            n_channels=0, sample_rate_hz=0, n_frames=0,
            sample_width_bytes=0, duration_sec=0.0,
            comptype="error", compname=str(e)
        )

    return AudioProps(
        group=group,
        filename=path.name,
        rel_id=f"{group}/{path.name}",
        size_bytes=st.st_size,
        modified_at=modified,
        n_channels=n_channels,
        sample_rate_hz=sr,
        n_frames=n_frames,
        sample_width_bytes=sample_width_bytes,
        duration_sec=duration,
        comptype=comptype,
        compname=compname,
    )