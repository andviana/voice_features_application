import pandas as pd

from pathlib import Path
from flask import current_app
from app.utils.manifest_utils import ManifestUtils

class MetadataService:
    
    @staticmethod
    def data_root() -> Path:
        return Path(current_app.config["DATA_DIR"]).resolve()

    @staticmethod
    def load_manifest_row(filename: str):
        manifest = (MetadataService.data_root() / "metadata" / "manifest.csv").resolve()
        if not manifest.exists():
            return {"highlight": {}, "all": {}}

        df = pd.read_csv(manifest)

        file_col = None
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id"):
                file_col = c
                break

        if file_col is None:
            return {"highlight": {}, "all": {}}

        key1 = filename
        key2 = filename.replace(".wav", "")
        s = df[file_col].astype(str).str.strip()
        row = df.loc[(s == key1) | (s == key2)]
        if row.empty:
            return {"highlight": {}, "all": {}}

        d = {k: ManifestUtils.sanitize(v) for k, v in row.iloc[0].to_dict().items()}

        highlight = {}
        for out_key, candidates in {
            "label": ["label", "grupo", "group", "class"],
            "age": ["age", "idade"],
            "sex": ["sex", "sexo", "gender"],
        }.items():
            for c in candidates:
                if c in d:
                    highlight[out_key] = d[c]
                    break

        return {"highlight": highlight, "all": d}    