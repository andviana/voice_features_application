import pandas as pd
from pathlib import Path
from flask import current_app

from app.utils.path_utils import PathUtils
from app.utils.manifest_utils import ManifestUtils


class MetadataService:
    
    @staticmethod
    def load_manifest_row(filename: str)-> dict:
        """
        Localiza e higieniza os dados de um paciente no manifest.csv.
        Centraliza a lógica de busca por nome de ficheiro ou ID de gravação.
        """
        manifest_path = PathUtils.manifest_filepath().resolve()        

        if not manifest_path.exists():
            return {"highlight": {}, "all": {}}

        try:
            df = pd.read_csv(manifest_path)
        
        except Exception:
            return {"highlight": {}, "all": {}}

        # Identifica a coluna que contém o nome do ficheiro ou ID
        file_col = None
        possible_cols = ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id")

        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in possible_cols:
                file_col = c
                break

        if file_col is None:
            return {"highlight": {}, "all": {}}

        
        # Busca flexível: aceita com ou sem a extensão .wav
        search_key = str(filename).strip()
        search_key_no_ext = search_key.replace(".wav", "")
        
        series = df[file_col].astype(str).str.strip()
        row = df.loc[(series == search_key) | (series == search_key_no_ext)]

        if row.empty:
            return {"highlight": {}, "all": {}}

        # Higieniza todos os valores da linha usando o ManifestUtils
        full_data = {k: ManifestUtils.sanitize(v) for k, v in row.iloc[0].to_dict().items()}

        # Extrai destaques (campos comuns em análises clínicas)
        highlight = {}
        mapping = {
            "label": ["label", "grupo", "group", "class", "diagnostico"],
            "age": ["age", "idade"],
            "sex": ["sex", "sexo", "gender", "genero"],
        }
        
        for key, candidates in mapping.items():
            for c in candidates:
                if c in full_data:
                    highlight[key] = full_data[c]
                    break

        return {"highlight": highlight, "all": full_data}
    