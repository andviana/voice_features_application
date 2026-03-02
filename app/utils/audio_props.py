from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import soundfile as sf


# soundfile pode lançar erros diferentes dependendo da versão / backend
# - sf.LibsndfileError (quando disponível)
# - RuntimeError / OSError (comum em alguns builds)
SOUNDFILE_ERRORS = tuple(
    e for e in (
        getattr(sf, "LibsndfileError", None),
        getattr(sf, "SoundFileError", None),
        RuntimeError,
        OSError,
    )
    if e is not None
)


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


def read_wav_props(path: Path, group: str) -> AudioProps:
    st = path.stat()
    modified = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Usa libsndfile via soundfile (suporta WAV extensible / float)
    info = sf.info(str(path))

    sr = int(info.samplerate or 0)
    frames = int(info.frames or 0)
    duration = (frames / float(sr)) if sr else 0.0

    subtype = (info.subtype or "").upper()
    sample_width = 0
    if "PCM_16" in subtype:
        sample_width = 2
    elif "PCM_24" in subtype:
        sample_width = 3
    elif "PCM_32" in subtype:
        sample_width = 4
    elif "FLOAT" in subtype:
        sample_width = 4
    elif "DOUBLE" in subtype:
        sample_width = 8

    return AudioProps(
        group=group,
        filename=path.name,
        rel_id=f"{group}/{path.name}",
        size_bytes=st.st_size,
        modified_at=modified,
        n_channels=int(info.channels or 0),
        sample_rate_hz=sr,
        n_frames=frames,
        sample_width_bytes=sample_width,
        duration_sec=duration,
        comptype=str(info.format or ""),
        compname=f"{info.format} / {info.subtype}".strip(" /"),
    )