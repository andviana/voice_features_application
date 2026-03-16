import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from app.config import Config

# Importação dos módulos do orientador
from extract_features.tsallis_amplitude_hist import amplitude_histogram_distribution
from extract_features.tsallis_q_gridsearch import grid_search_q
from extract_features.tsallis_q_extensivity import estimate_q_extensivity
from extract_features.tsallis_q_qgaussian_fit import estimate_q_from_amplitude_qgaussian

from app.utils.path_utils import PathUtils

class TsallisOptimizationService:
    def __init__(self):        
        self.output_dir = (PathUtils.data_root() / "optimization_results").resolve
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuração de Log Específico para Otimização
        self.log_file = os.path.join(self.output_dir, f'opt_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        self.logger = logging.getLogger('TsallisOptimization')
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run_full_optimization(self, hc_signals: List[np.ndarray], pd_signals: List[np.ndarray]) -> str:
        """
        Executa o protocolo completo e salva os artefatos em CSV.
        """
        try:
            self.logger.info("Iniciando Protocolo de Otimização de Tsallis (Base 8kHz).")
            
            # 1. Gerar Distribuições de Probabilidade (Artefato 1)
            self.logger.info("Passo 1: Gerando histogramas de amplitude (Z-score).")
            prob_hc = [amplitude_histogram_distribution(s) for s in hc_signals]
            prob_pd = [amplitude_histogram_distribution(s) for s in pd_signals]
            
            # Salvar cache das distribuições para auditoria
            pd.DataFrame(prob_hc).to_csv(os.path.join(self.output_dir, 'dist_cache_hc.csv'))
            pd.DataFrame(prob_pd).to_csv(os.path.join(self.output_dir, 'dist_cache_pd.csv'))

            # 2. Estimação via Grid Search
            self.logger.info("Passo 2: Executando Grid Search (DP vs HC).")
            res_grid, dict_grid = grid_search_q(np.array(prob_hc), np.array(prob_pd))
            
            # 3. Estimação via Extensividade (Média do Dataset)
            self.logger.info("Passo 3: Executando Critério de Extensividade.")
            ext_results = []
            for s in (hc_signals + pd_signals):
                _, d = estimate_q_extensivity(s)
                ext_results.append(d)
            df_ext = pd.DataFrame(ext_results)
            q_ext_mean = df_ext['q_ext_hat'].mean()

            # 4. Estimação via q-Gaussian Fit (Média do Dataset)
            self.logger.info("Passo 4: Executando Ajuste q-Gaussiano.")
            fit_results = []
            for s in (hc_signals + pd_signals):
                _, d = estimate_q_from_amplitude_qgaussian(s)
                fit_results.append(d)
            df_fit = pd.DataFrame(fit_results)
            q_fit_mean = df_fit['q_qgauss_hat'].mean()

            # 5. Consolidação da Tabela de Decisão (Artefato 2)
            results_summary = [
                {"Metodo": "Grid Search", "Q_Estimado": dict_grid['q_grid_opt'], "Metrica": "t-stat", "Score": dict_grid['q_grid_score_opt']},
                {"Metodo": "Extensividade", "Q_Estimado": q_ext_mean, "Metrica": "R2_medio", "Score": df_ext['q_ext_r2'].mean()},
                {"Metodo": "q-Gaussian Fit", "Q_Estimado": q_fit_mean, "Metrica": "RMSE_medio", "Score": df_fit['q_qgauss_rmse'].mean()}
            ]
            
            summary_path = os.path.join(self.output_dir, 'summary_q_optimization.csv')
            pd.DataFrame(results_summary).to_csv(summary_path, index=False)
            
            self.logger.info(f"Otimização concluída com sucesso. Resultados em: {summary_path}")
            return summary_path

        except Exception as e:
            self.logger.error(f"Erro crítico na otimização: {str(e)}", exc_info=True)
            raise e