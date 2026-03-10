from pathlib import Path
from flask import current_app

class PathUtils:
    
    @staticmethod
    def data_root() -> Path:
        return Path(current_app.config["DATA_DIR"]).resolve()

    @staticmethod
    def raw_root() -> Path:
        return (PathUtils.data_root() / "audio_raw").resolve()

    @staticmethod
    def processed_root() -> Path:
        return (PathUtils.data_root() / "audio_processed").resolve()

