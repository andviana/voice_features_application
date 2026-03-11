from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
from flask import current_app, abort



class PathUtils:
    MANIFEST_FILE = "manifest.csv"
    FEATURES_FILE = "dataset_voz_completo.csv"

    FEATURES_ROOT = "features"
    MANIFEST_ROOT = "metadata"

    ALLOWED_GROUPS = ["HC_AH", "PD_AH"]


    @staticmethod
    def data_root() -> Path:
        """Retorna o caminho absoluto para o diretório de dados."""
        return Path(current_app.config["DATA_DIR"]).resolve()

    @staticmethod
    def raw_root() -> Path:
        """Retorna o caminho absoluto para o diretório de áudios brutos."""
        return (PathUtils.data_root() / "audio_raw").resolve()

    @staticmethod
    def processed_root() -> Path:
        """Retorna o caminho absoluto para o diretório de áudios processados."""
        return (PathUtils.data_root() / "audio_processed").resolve()
    
    @staticmethod
    def safe_group_dir(base_path: Path, group: str) -> Path:
        """
        Valida se o grupo é permitido e se o diretório está dentro da base.
        Extraído da lógica de segurança presente nos blueprints.
        """
        allowed_groups = {"HC_AH", "PD_AH"}
        if group not in allowed_groups:
            abort(404)
        
        target_dir = (base_path / group).resolve()
        
        # Garante que não há tentativa de 'Path Traversal'
        if base_path not in target_dir.parents and target_dir != base_path:
            abort(400)
            
        return target_dir

    @staticmethod
    def safe_wav_path(base_path: Path, group: str, filename: str) -> Path:
        """Resolve e valida a existência de um arquivo WAV específico."""
        group_dir = PathUtils.safe_group_dir(base_path, group)
        # os.path.basename evita injeção de subdiretórios no nome do arquivo
        fpath = (group_dir / os.path.basename(filename)).resolve()
        
        if not fpath.exists() or fpath.suffix.lower() != ".wav":
            abort(404)
            
        return fpath

    @staticmethod
    def manifest_filepath() -> Path:
        return (PathUtils.data_root() / PathUtils.MANIFEST_ROOT / PathUtils.MANIFEST_FILE)
    
    # @staticmethod
    # def features_filepath() -> Path:
    #     return (PathUtils.data_root() / PathUtils.FEATURES_ROOT / PathUtils.FEATURES_FILE)
    
    # @staticmethod
    # def find_filename_column(df: pd.DataFrame) -> str | None:
    #     for c in df.columns:
    #         cl = str(c).strip().lower()
    #         if cl in ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id"):
    #             return c
    #     return None
    
    @staticmethod
    def features_csv_for_group(group: str) -> Path:
        p1 = (PathUtils.data_root() / PathUtils.FEATURES_ROOT / group / PathUtils.FEATURES_FILE).resolve()
        if p1.exists():
            return p1
        p2 = (PathUtils.data_root() / PathUtils.FEATURES_ROOT / PathUtils.FEATURES_FILE).resolve()
        return p2