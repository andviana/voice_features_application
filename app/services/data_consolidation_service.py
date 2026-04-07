import pandas as pd
import os
import io
from datetime import datetime
from app.utils.path_utils import PathUtils

class DataConsolidationService:
    def __init__(self):
        self.data_root = PathUtils.data_root()
        self.metadata_root = PathUtils.metadata_root()
        self.features_root = PathUtils.features_root()
        # Optimization results are usually in data/optimization_results
        self.optimization_root = self.data_root / "optimization_results"

    def consolidate(self) -> pd.DataFrame:
        """
        Consolida dados de múltiplas fontes: Demográficos, Manifesto, Características Acústicas e Tsallis.
        """
        # 1. Carregar arquivos com tratamento de erros básicos
        demographics_path = self.metadata_root / "Demographics_age_sex.xlsx"
        manifest_path = PathUtils.manifest_filepath()
        hc_path = PathUtils.features_csv_for_group("HC_AH")
        pd_path = PathUtils.features_csv_for_group("PD_AH")
        tsallis_path = self.optimization_root / "group_comparison_detailed.csv"

        if not demographics_path.exists():
            raise FileNotFoundError(f"Arquivo demográfico não encontrado: {demographics_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Arquivo manifesto não encontrado: {manifest_path}")

        df_demographics = pd.read_excel(demographics_path, sheet_name="Parselmouth")
        df_manifest = pd.read_csv(manifest_path)
        
        # Acoustic features (HC + PD)
        dfs_acoustic = []
        if hc_path.exists():
            dfs_acoustic.append(pd.read_csv(hc_path))
        if pd_path.exists():
            dfs_acoustic.append(pd.read_csv(pd_path))
        
        df_acoustic = pd.concat(dfs_acoustic, ignore_index=True) if dfs_acoustic else pd.DataFrame()
        
        # Tsallis results (opcional, pode não ter sido gerado ainda)
        df_tsallis = pd.DataFrame()
        if tsallis_path.exists():
            df_tsallis = pd.read_csv(tsallis_path)

        # 2. Função de limpeza de IDs
        def clean_id(val):
            if pd.isna(val): return val
            s = str(val).strip()
            if s.lower().endswith(".wav"):
                s = s[:-4]
            return s

        # Aplicar limpeza nos IDs de todas as fontes
        df_demographics['Sample ID'] = df_demographics['Sample ID'].apply(clean_id)
        df_manifest['recording_id'] = df_manifest['recording_id'].apply(clean_id)
        
        if not df_acoustic.empty:
            df_acoustic['file_name'] = df_acoustic['file_name'].apply(clean_id)
            
        if not df_tsallis.empty:
            df_tsallis['Arquivo'] = df_tsallis['Arquivo'].apply(clean_id)

        # 3. Merge (Base: Demographics)
        base = df_demographics.copy()
        
        # Merge com Manifesto
        # Mantemos todas as colunas exceto as que já existem ou IDs redundantes
        manifest_cols = [c for c in df_manifest.columns if c not in ['recording_id', 'age', 'sex', 'group']]
        base = base.merge(df_manifest[['recording_id'] + manifest_cols], left_on='Sample ID', right_on='recording_id', how='left')
        
        # Merge com Características Acústicas
        if not df_acoustic.empty:
            # Evitar colunas duplicadas que não sejam o ID
            acoustic_cols = [c for c in df_acoustic.columns if c not in ['file_name', 'group'] and c not in base.columns]
            base = base.merge(df_acoustic[['file_name'] + acoustic_cols], left_on='Sample ID', right_on='file_name', how='left')
        
        # Merge com Tsallis
        if not df_tsallis.empty:
            tsallis_cols = [c for c in df_tsallis.columns if c not in ['Arquivo', 'Grupo'] and c not in base.columns]
            base = base.merge(df_tsallis[['Arquivo'] + tsallis_cols], left_on='Sample ID', right_on='Arquivo', how='left')

        # 4. Limpeza final de colunas de junção
        drop_cols = ['recording_id', 'file_name', 'Arquivo']
        base.drop(columns=[c for c in drop_cols if c in base.columns], inplace=True)

        return base

    def generate_xlsx(self, df: pd.DataFrame) -> io.BytesIO:
        """Gera um arquivo XLSX em memória a partir do DataFrame."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Consolidado')
        output.seek(0)
        return output
