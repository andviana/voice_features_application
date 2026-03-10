import pandas as pd
import numpy as np
import math 

class ManifestUtils:
    
    # Caminho para o arquivo de metadados
    MANIFEST_PATH = 'manifest.csv'

    @staticmethod
    def get_sex_from_manifest(recording_id):
        """Busca o sexo do paciente no arquivo manifest.csv."""
        try:
            df = pd.read_csv(ManifestUtils.MANIFEST_PATH)
            row = df[df['recording_id'] == recording_id]
            if not row.empty:
                return row.iloc[0]['sex']
        except Exception as e:
            print(f"Erro ao ler manifest: {e}")
        return 'Unknown'


    @staticmethod
    def sanitize(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v