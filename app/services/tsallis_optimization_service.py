import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import  List
import matplotlib.pyplot as plt
import seaborn as sns

from app.utils.path_utils import PathUtils

# Importação dos módulos do orientador
from extract_features.tsallis_amplitude_hist import amplitude_histogram_distribution
from extract_features.tsallis_q_gridsearch import grid_search_q
from extract_features.tsallis_q_extensivity import estimate_q_extensivity
from extract_features.tsallis_q_qgaussian_fit import estimate_q_from_amplitude_qgaussian

class TsallisOptimizationService:
    def __init__(self):        
       # 1. Obtemos o Path e garantimos a execução do .resolve()
        path_base = PathUtils.data_root() / "optimization_results"
        path_absoluto = path_base.resolve()
        
        # 2. Convertemos OBRIGATORIAMENTE para string antes de qualquer uso
        self.output_dir = str(path_absoluto)
        
        # 3. Verificamos se é string antes de criar a pasta
        if not isinstance(self.output_dir, str):
            raise TypeError(f"Erro interno: self.output_dir deveria ser str, mas é {type(self.output_dir)}")
            
        os.makedirs(self.output_dir, exist_ok=True)        
        # Configuração de Log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f'opt_log_{timestamp}.log')
        
        self.logger = logging.getLogger('TsallisOptimization')
        if not self.logger.handlers:
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
            if not isinstance(hc_signals, list) or not isinstance(pd_signals, list):
                raise TypeError("hc_signals e pd_signals devem ser listas de numpy arrays.")

            if hc_signals and not isinstance(hc_signals[0], np.ndarray):
                raise TypeError("hc_signals deve conter numpy arrays individuais.")

            if pd_signals and not isinstance(pd_signals[0], np.ndarray):
                raise TypeError("pd_signals deve conter numpy arrays individuais.")

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
        
        
    def get_latest_results(self) -> dict:
        """Retorna metadados sobre o último processamento realizado."""
        import glob
        
        # Procura o log mais recente
        log_files = glob.glob(os.path.join(self.output_dir, "opt_log_*.log"))
        latest_log = max(log_files, key=os.path.getctime) if log_files else None
        
        log_content = ""
        if latest_log:
            with open(latest_log, 'r') as f:
                log_content = f.read()

        return {
            "has_results": os.path.exists(os.path.join(self.output_dir, 'summary_q_optimization.csv')),
            "latest_log_content": log_content,
            "files": {
                "summary": "summary_q_optimization.csv",
                "cache_hc": "dist_cache_hc.csv",
                "cache_pd": "dist_cache_pd.csv"
            }
        }
    

    def run_group_comparison(self, hc_data, pd_data):
        hc_signals, hc_names = hc_data
        pd_signals, pd_names = pd_data

        results = []

        def process_group(signals, names, group_label):
            group_res = []
            for s, name in zip(signals, names):
                _, d_ext = estimate_q_extensivity(s)
                _, d_fit = estimate_q_from_amplitude_qgaussian(s)
                group_res.append({
                    "Arquivo": name,
                    "Grupo": group_label,
                    "q_Extensividade": d_ext['q_ext_hat'],
                    "q_Gaussian_Fit": d_fit['q_qgauss_hat']
                })
            return group_res

        self.logger.info("Iniciando comparação por grupos...")
        results.extend(process_group(hc_signals, hc_names, "HC"))
        results.extend(process_group(pd_signals, pd_names, "PD"))

        df = pd.DataFrame(results)
        detailed_path = os.path.join(self.output_dir, 'group_comparison_detailed.csv')
        df.to_csv(detailed_path, index=False)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.boxplot(x='Grupo', y='q_Extensividade', data=df, hue='Grupo', palette='Set2', legend=False)
        plt.title('Comparação: q de Extensividade')

        plt.subplot(1, 2, 2)
        sns.boxplot(x='Grupo', y='q_Gaussian_Fit', data=df, hue='Grupo', palette='Set1', legend=False)
        plt.title('Comparação: q de Ajuste Gaussiano')

        plot_path = os.path.join(self.output_dir, 'comparison_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        summary = (
            df.groupby('Grupo')
            .agg(
                total_amostras=('Arquivo', 'count'),
                q_Extensividade_media=('q_Extensividade', 'mean'),
                q_Extensividade_std=('q_Extensividade', 'std'),
                q_Gaussian_Fit_media=('q_Gaussian_Fit', 'mean'),
                q_Gaussian_Fit_std=('q_Gaussian_Fit', 'std'),
            )
            .reset_index()
            .fillna('')
            .round(6)
        )

        summary_path = os.path.join(self.output_dir, 'group_stats_summary.csv')
        summary.to_csv(summary_path, index=False)

        return {
            "detailed_csv": "group_comparison_detailed.csv",
            "stats_csv": "group_stats_summary.csv",
            "plot_image": "comparison_plot.png",
            "summary_columns": summary.columns.tolist(),
            "summary_records": summary.to_dict(orient='records'),
            "detailed_records": df.to_dict(orient='records')
        }
    

    def get_latest_group_comparison_results(self) -> dict:
        detailed_csv = os.path.join(self.output_dir, 'group_comparison_detailed.csv')
        stats_csv = os.path.join(self.output_dir, 'group_stats_summary.csv')
        plot_image = os.path.join(self.output_dir, 'comparison_plot.png')

        has_results = (
            os.path.exists(detailed_csv)
            and os.path.exists(stats_csv)
            and os.path.exists(plot_image)
        )

        if not has_results:
            return {
                "has_results": False,
                "summary_columns": [],
                "summary_records": [],
                "detailed_records": [],
                "files": {
                    "detailed_csv": "group_comparison_detailed.csv",
                    "stats_csv": "group_stats_summary.csv",
                    "plot_image": "comparison_plot.png"
                }
            }

        df_summary = pd.read_csv(stats_csv).fillna('')
        df_detailed = pd.read_csv(detailed_csv).fillna('')

        return {
            "has_results": True,
            "summary_columns": df_summary.columns.tolist(),
            "summary_records": df_summary.to_dict(orient='records'),
            "detailed_records": df_detailed.to_dict(orient='records'),
            "files": {
                "detailed_csv": "group_comparison_detailed.csv",
                "stats_csv": "group_stats_summary.csv",
                "plot_image": "comparison_plot.png"
            }
        }