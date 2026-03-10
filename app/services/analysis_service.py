import pandas as pd
import numpy as np
import math

class AnalysisService:
    
    @staticmethod
    def sanitize_value(v):
        """Lógica de sanitização extraída de scientific_analysis."""
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    @staticmethod
    def pair_features(row: pd.Series):
        """
        Versão avançada para extração de 'mean' e 'std'.
        Identifica pares (mean/std) e trata valores isolados (single).
        """
        data = {k: AnalysisService.sanitize_value(v) for k, v in row.to_dict().items()}
        used = set()
        paired, single = [], []
        ignore_columns = {"file_name", "group"}

        for k, v in data.items():
            if k in ignore_columns: continue
            lk = k.lower()

            if "_mean" in lk:
                # Lógica robusta para diferentes sufixos (hz ou comum)
                std_key = k.replace("_mean_hz", "_std_hz").replace("_mean", "_std")
                base = lk.replace("_mean_hz", "").replace("_mean", "")

                if std_key in data:
                    paired.append({
                        "feature": base,
                        "mean": data.get(k),
                        "std": data.get(std_key)
                    })
                    used.update([k, std_key])
                else:
                    single.append({"feature": base, "value": data.get(k)})
                    used.add(k)

        # Adiciona colunas que não seguem o padrão mean/std
        for k, v in data.items():
            if k not in ignore_columns and k not in used:
                single.append({"feature": k, "value": v})
        
        return sorted(paired, key=lambda x: x["feature"]), sorted(single, key=lambda x: x["feature"])