from __future__ import annotations
import os
from pathlib import Path
from flask import current_app, abort

class PathUtils:
    # -------------------------------------------------------------
    # Constantes Centrais (Facilitam a alteração de diretórios)
    # -------------------------------------------------------------
    PROJECT_NAME = "antigravity"  # Fallback caso a aplicação não esteja iniciada corretamente
    DATA_DIR_NAME = "data"
    AUDIO_RAW_DIR_NAME = "audio_raw"
    AUDIO_PROCESSED_DIR_NAME = "audio_processed"
    AUDIO_REJECTED_DIR_NAME = "audio_rejected"
    FEATURES_DIR_NAME = "features"
    METADATA_DIR_NAME = "metadata"
    
    MANIFEST_FILE = "manifest.csv"
    FEATURES_FILE = "dataset_voz_completo.csv"

    ALLOWED_GROUPS_LIST = ["HC_AH", "PD_AH"]
    ALLOWED_GROUPS = set(ALLOWED_GROUPS_LIST)

    # -------------------------------------------------------------
    # Métodos de Resolução de Caminhos (Diretórios)
    # -------------------------------------------------------------
    @staticmethod
    def project_root() -> Path:
        """Retorna o caminho raiz do projeto."""
        return Path(current_app.root_path).parent.resolve()

    @staticmethod
    def data_root() -> Path:
        """Retorna o caminho absoluto para o diretório raiz de dados."""
        # Se configurado no app, usa; senão faz fallback seguro para a raiz do projeto
        if "DATA_DIR" in current_app.config:
            return Path(current_app.config["DATA_DIR"]).resolve()
        return (PathUtils.project_root() / PathUtils.DATA_DIR_NAME).resolve()

    @staticmethod
    def raw_root() -> Path:
        """Retorna o caminho absoluto para o diretório de áudios brutos."""
        return (PathUtils.data_root() / PathUtils.AUDIO_RAW_DIR_NAME).resolve()

    @staticmethod
    def processed_root() -> Path:
        """Retorna o caminho absoluto para o diretório de áudios processados."""
        return (PathUtils.data_root() / PathUtils.AUDIO_PROCESSED_DIR_NAME).resolve()

    @staticmethod
    def rejected_root() -> Path:
        """Retorna o caminho absoluto para o diretório de áudios rejeitados/lixeira."""
        return (PathUtils.data_root() / PathUtils.AUDIO_REJECTED_DIR_NAME).resolve()

    @staticmethod
    def features_root() -> Path:
        """Retorna o caminho absoluto para o diretório base de features."""
        return (PathUtils.data_root() / PathUtils.FEATURES_DIR_NAME).resolve()

    @staticmethod
    def metadata_root() -> Path:
        """Retorna o caminho absoluto para o diretório base de metadados."""
        return (PathUtils.data_root() / PathUtils.METADATA_DIR_NAME).resolve()

    # -------------------------------------------------------------
    # Métodos Seguros de Acesso a Grupos / Arquivos
    # -------------------------------------------------------------
    @staticmethod
    def safe_group_dir(base_path: Path, group: str) -> Path:
        """
        Valida se o grupo é permitido e se o diretório está dentro da base para prevenir Path Traversal.
        """
        if group not in PathUtils.ALLOWED_GROUPS:
            abort(404)
        
        target_dir = (base_path / group).resolve()
        
        if base_path not in target_dir.parents and target_dir != base_path:
            abort(400)
            
        return target_dir

    @staticmethod
    def safe_wav_path(base_path: Path, group: str, filename: str) -> Path:
        """Resolve e valida a existência de um arquivo WAV específico dentro de um grupo e base informados."""
        group_dir = PathUtils.safe_group_dir(base_path, group)
        fpath = (group_dir / os.path.basename(filename)).resolve()
        
        if not fpath.exists() or fpath.suffix.lower() != ".wav":
            abort(404)
            
        return fpath

    # -------------------------------------------------------------
    # Atalhos para Arquivos Específicos
    # -------------------------------------------------------------
    @staticmethod
    def manifest_filepath() -> Path:
        """Caminho absoluto para o arquivo CSV de manifesto com metadados/demográficos."""
        return (PathUtils.metadata_root() / PathUtils.MANIFEST_FILE).resolve()
    
    @staticmethod
    def features_csv_for_group(group: str) -> Path:
        """Busca o CSV de features, priorizando a pasta do grupo; se não existir lá, tenta a raiz de features."""
        p1 = (PathUtils.features_root() / group / PathUtils.FEATURES_FILE).resolve()
        if p1.exists():
            return p1
        p2 = (PathUtils.features_root() / PathUtils.FEATURES_FILE).resolve()
        return p2