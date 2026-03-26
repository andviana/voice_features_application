from __future__ import annotations
import os
import pandas as pd
import numpy as np
from flask import render_template
from scipy import stats
from . import bp
from app.utils.path_utils import PathUtils

def calculate_cohen_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

def calculate_rosenthal_r(z_stat, n):
    return abs(z_stat) / np.sqrt(n) if n > 0 else 0

@bp.route("/", methods=["GET"])
def explore_data():
    manifest_path = PathUtils.manifest_filepath()
    
    # 1. LOAD & MERGE DATA
    df_merged = pd.DataFrame()
    
    if manifest_path.exists():
        try:
            df_manifest = pd.read_csv(manifest_path)
            # Ensure group is standardized
            df_manifest["group"] = df_manifest["group"].astype(str).str.upper().str.strip()
            df_manifest["group"] = df_manifest["group"].replace({"CONTROL": "HC", "PARKINSON": "PD", "HC_AH": "HC", "PD_AH": "PD"})
            
            # Prepare merge key
            df_manifest["merge_key"] = df_manifest["recording_id"].astype(str) + ".wav"
            
            # Load Features (HC and PD)
            dfs_feat = []
            for gid in ["HC_AH", "PD_AH"]:
                fpath = PathUtils.features_csv_for_group(gid)
                if fpath.exists():
                    dfs_feat.append(pd.read_csv(fpath))
            
            if dfs_feat:
                df_features = pd.concat(dfs_feat, ignore_index=True)
                # Drop redundant 'group' from features to avoid group_x/group_y
                if "group" in df_features.columns:
                    df_features = df_features.drop(columns=["group"])
                df_merged = pd.merge(df_manifest, df_features, left_on="merge_key", right_on="file_name", how="inner")
            
            # Load Tsallis results
            tsallis_path = PathUtils.project_root() / "data" / "optimization_results" / "group_comparison_detailed.csv"
            if tsallis_path.exists():
                df_tsallis = pd.read_csv(tsallis_path)
                # Drop redundant 'Grupo' if it exists or handle it
                if "Grupo" in df_tsallis.columns:
                    df_tsallis = df_tsallis.drop(columns=["Grupo"])
                df_merged = pd.merge(df_merged, df_tsallis, left_on="merge_key", right_on="Arquivo", how="left")
                
        except Exception as e:
            print(f"Erro no merge de dados: {e}")

    # 2. STATISTICAL ANALYSIS
    stats_results = []
    alpha = 0.01

    if not df_merged.empty:
        # Categorial: Sex
        groups = df_merged["group"].unique()
        if "HC" in groups and "PD" in groups:
            contingency = pd.crosstab(df_merged["group"], df_merged["sex"])
            # Fisher if N < 5 in any cell, else Chi2
            if (contingency < 5).any().any():
                try:
                    _, p_sex = stats.fisher_exact(contingency)
                    test_type = "Fisher's Exact"
                except:
                    # Fallback to Chi2 if Fisher logic fails (e.g. not 2x2)
                    res = stats.chi2_contingency(contingency)
                    p_sex = res.pvalue
                    test_type = "Chi-squared"
            else:
                res = stats.chi2_contingency(contingency)
                p_sex = res.pvalue
                test_type = "Chi-squared"
            
            stats_results.append({
                "variable": "Sex (Distribution)",
                "statistic": "-",
                "p_value": float(p_sex),
                "test": test_type,
                "effect_size": "-",
                "is_significant": bool(p_sex < alpha)
            })

        # Continuous Variables
        vars_to_test = {
            "age": "Idade",
            "hnr_mean_db": "HNR Mean (dB)",
            "jitter_local": "Local Jitter (%)",
            "shimmer_local": "Local Shimmer (%)",
            "mfcc1_mean": "MFCC 1 Mean",
            "mfcc2_mean": "MFCC 2 Mean",
            "mfcc3_mean": "MFCC 3 Mean",
            "q_Extensividade": "q-Extensividade",
            "q_Gaussian_Fit": "q-Gaussian Fit"
        }

        for col, label in vars_to_test.items():
            if col in df_merged.columns:
                data_hc = pd.to_numeric(df_merged[df_merged["group"] == "HC"][col], errors='coerce').dropna()
                data_pd = pd.to_numeric(df_merged[df_merged["group"] == "PD"][col], errors='coerce').dropna()

                if len(data_hc) > 3 and len(data_pd) > 3:
                    # Normality check (Shapiro)
                    _, p_norm_hc = stats.shapiro(data_hc)
                    _, p_norm_pd = stats.shapiro(data_pd)
                    is_normal = (p_norm_hc > 0.05) and (p_norm_pd > 0.05)

                    if is_normal:
                        # t-test
                        t_stat, p_val = stats.ttest_ind(data_hc, data_pd, equal_var=False)
                        effect = calculate_cohen_d(data_hc, data_pd)
                        applied_test = "Student's t"
                        stat_val = t_stat
                        effect_label = "Cohen's d"
                    else:
                        # Mann-Whitney U
                        u_stat, p_val = stats.mannwhitneyu(data_hc, data_pd, alternative='two-sided')
                        n1, n2 = len(data_hc), len(data_pd)
                        z_stat = (u_stat - (n1 * n2 / 2.0)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
                        effect = calculate_rosenthal_r(z_stat, n1 + n2)
                        applied_test = "Mann-Whitney U"
                        stat_val = u_stat
                        effect_label = "Rosenthal's r"

                    stats_results.append({
                        "variable": label,
                        "statistic": f"{stat_val:.3f}",
                        "p_value": float(p_val),
                        "test": applied_test,
                        "effect_size": f"{effect:.3f} ({effect_label})",
                        "is_significant": bool(p_val < alpha)
                    })

    # 3. PREPARE PAYLOAD FOR UI
    demo_data = {
        "groups": {"HC": 0, "PD": 0},
        "sex_total": {"Total": {"M": 0, "F": 0}, "HC": {"M": 0, "F": 0}, "PD": {"M": 0, "F": 0}},
        "age": {"Total": [], "HC": [], "PD": []}
    }
    
    if not df_merged.empty:
        counts = df_merged["group"].value_counts()
        demo_data["groups"]["HC"] = int(counts.get("HC", 0))
        demo_data["groups"]["PD"] = int(counts.get("PD", 0))
        
        if "age" in df_merged.columns:
            demo_data["age"]["Total"] = df_merged["age"].dropna().tolist()
            demo_data["age"]["HC"] = df_merged[df_merged["group"] == "HC"]["age"].dropna().tolist()
            demo_data["age"]["PD"] = df_merged[df_merged["group"] == "PD"]["age"].dropna().tolist()
        
        if "sex" in df_merged.columns:
            df_merged["sex_std"] = df_merged["sex"].astype(str).str.upper().str.strip().str[0]
            total_counts = df_merged["sex_std"].value_counts()
            demo_data["sex_total"]["Total"]["M"] = int(total_counts.get("M", 0))
            demo_data["sex_total"]["Total"]["F"] = int(total_counts.get("F", 0))
            
            hc_counts = df_merged[df_merged["group"] == "HC"]["sex_std"].value_counts()
            pd_counts = df_merged[df_merged["group"] == "PD"]["sex_std"].value_counts()
            demo_data["sex_total"]["HC"]["M"] = int(hc_counts.get("M", 0))
            demo_data["sex_total"]["HC"]["F"] = int(hc_counts.get("F", 0))
            demo_data["sex_total"]["PD"]["M"] = int(pd_counts.get("M", 0))
            demo_data["sex_total"]["PD"]["F"] = int(pd_counts.get("F", 0))

    features_data = {
        "pitch": {"HC": {}, "PD": {}},
        "jitter": {"HC": [], "PD": []},
        "shimmer": {"HC": [], "PD": []},
        "tsallis": {"HC": {"q_ext": [], "q_gauss": []}, "PD": {"q_ext": [], "q_gauss": []}},
        "mfccs": {"HC": {"m1": [], "m2": [], "m3": []}, "PD": {"m1": [], "m2": [], "m3": []}}
    }
    
    if not df_merged.empty:
        for prefix in ["HC", "PD"]:
            sub = df_merged[df_merged["group"] == prefix]
            for col in ["f0_mean_hz", "f0_min_hz", "f0_max_hz", "f0_cv"]:
                if col in sub.columns:
                    features_data["pitch"][prefix][col] = pd.to_numeric(sub[col], errors='coerce').dropna().tolist()
            
            if "jitter_local" in sub.columns:
                features_data["jitter"][prefix] = pd.to_numeric(sub["jitter_local"], errors='coerce').dropna().tolist()
            if "shimmer_local" in sub.columns:
                features_data["shimmer"][prefix] = pd.to_numeric(sub["shimmer_local"], errors='coerce').dropna().tolist()
            
            if "q_Extensividade" in sub.columns:
                features_data["tsallis"][prefix]["q_ext"] = pd.to_numeric(sub["q_Extensividade"], errors='coerce').dropna().tolist()
            if "q_Gaussian_Fit" in sub.columns:
                features_data["tsallis"][prefix]["q_gauss"] = pd.to_numeric(sub["q_Gaussian_Fit"], errors='coerce').dropna().tolist()
                
            if "mfcc1_mean" in sub.columns:
                features_data["mfccs"][prefix]["m1"] = pd.to_numeric(sub["mfcc1_mean"], errors='coerce').dropna().tolist()
            if "mfcc2_mean" in sub.columns:
                features_data["mfccs"][prefix]["m2"] = pd.to_numeric(sub["mfcc2_mean"], errors='coerce').dropna().tolist()
            if "mfcc3_mean" in sub.columns:
                features_data["mfccs"][prefix]["m3"] = pd.to_numeric(sub["mfcc3_mean"], errors='coerce').dropna().tolist()

    payload = {
        "demographics": demo_data,
        "features": features_data,
        "stats": stats_results
    }

    return render_template("data_exploration/view.html", data=payload)
