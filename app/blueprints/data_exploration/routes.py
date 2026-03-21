from flask import render_template
import pandas as pd
from app.utils.path_utils import PathUtils
import os
from . import bp

@bp.route("/", methods=["GET"])
def explore_data():
    manifest_path = PathUtils.manifest_filepath()
    
    # DEMOGRAPHICS
    demo_data = {
        "groups": {"HC": 0, "PD": 0},
        "sex_total": {"Total": {"M": 0, "F": 0}, "HC": {"M": 0, "F": 0}, "PD": {"M": 0, "F": 0}},
        "age": {"Total": [], "HC": [], "PD": []}
    }
    
    if manifest_path.exists():
        try:
            df = pd.read_csv(manifest_path)
            # Find appropriate columns flexibly
            age_col = next((c for c in df.columns if c.lower() in ["age", "idade"]), None)
            sex_col = next((c for c in df.columns if c.lower() in ["sex", "sexo", "gender", "genero"]), None)
            group_col = next((c for c in df.columns if c.lower() in ["label", "grupo", "group", "class", "diagnostico"]), None)
            
            if group_col:
                df[group_col] = df[group_col].astype(str).str.upper().str.strip()
                # Treat missing or generic "CONTROL" as "HC" and "PARKINSON" as "PD" for compatibility
                df[group_col] = df[group_col].replace({"CONTROL": "HC", "PARKINSON": "PD", "HC_AH": "HC", "PD_AH": "PD"})
                
                counts = df[group_col].value_counts()
                demo_data["groups"]["HC"] = int(counts.get("HC", 0))
                demo_data["groups"]["PD"] = int(counts.get("PD", 0))
            
            if age_col:
                df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
                demo_data["age"]["Total"] = df[age_col].dropna().tolist()
                if group_col:
                    demo_data["age"]["HC"] = df[df[group_col] == "HC"][age_col].dropna().tolist()
                    demo_data["age"]["PD"] = df[df[group_col] == "PD"][age_col].dropna().tolist()
                    
            if sex_col:
                # Padronizar sexo (M/F)
                df[sex_col] = df[sex_col].astype(str).str.upper().str.strip().str[0]
                total_counts = df[sex_col].value_counts()
                demo_data["sex_total"]["Total"]["M"] = int(total_counts.get("M", 0))
                demo_data["sex_total"]["Total"]["F"] = int(total_counts.get("F", 0))
                
                if group_col:
                    hc_counts = df[df[group_col] == "HC"][sex_col].value_counts()
                    pd_counts = df[df[group_col] == "PD"][sex_col].value_counts()
                    demo_data["sex_total"]["HC"]["M"] = int(hc_counts.get("M", 0))
                    demo_data["sex_total"]["HC"]["F"] = int(hc_counts.get("F", 0))
                    demo_data["sex_total"]["PD"]["M"] = int(pd_counts.get("M", 0))
                    demo_data["sex_total"]["PD"]["F"] = int(pd_counts.get("F", 0))
        except Exception as e:
            print(f"Erro ao processar manifest demográfico: {e}")

    # ACOUSTIC FEATURES (Jitter, Shimmer, Pitch)
    features_data = {
        "pitch": {"HC": {}, "PD": {}},
        "jitter": {"HC": [], "PD": []},
        "shimmer": {"HC": [], "PD": []}
    }
    
    def extract_group_features(group_id, prefix):
        features_csv = PathUtils.features_csv_for_group(group_id)
        if features_csv.exists():
            try:
                df_feat = pd.read_csv(features_csv)
                if "f0_mean_hz" in df_feat.columns:
                    features_data["pitch"][prefix]["f0_mean_hz"] = pd.to_numeric(df_feat["f0_mean_hz"], errors='coerce').dropna().tolist()
                if "f0_min_hz" in df_feat.columns:
                    features_data["pitch"][prefix]["f0_min_hz"] = pd.to_numeric(df_feat["f0_min_hz"], errors='coerce').dropna().tolist()
                if "f0_max_hz" in df_feat.columns:
                    features_data["pitch"][prefix]["f0_max_hz"] = pd.to_numeric(df_feat["f0_max_hz"], errors='coerce').dropna().tolist()
                if "f0_cv" in df_feat.columns:
                    features_data["pitch"][prefix]["f0_cv"] = pd.to_numeric(df_feat["f0_cv"], errors='coerce').dropna().tolist()

                if jitter_col:
                    features_data["jitter"][prefix] = pd.to_numeric(df_feat[jitter_col], errors='coerce').dropna().tolist()
                if shimmer_col:
                    features_data["shimmer"][prefix] = pd.to_numeric(df_feat[shimmer_col], errors='coerce').dropna().tolist()
            except Exception as e:
                print(f"Erro ao processar features de {group_id}: {e}")

    extract_group_features("HC_AH", "HC")
    extract_group_features("PD_AH", "PD")

    payload = {
        "demographics": demo_data,
        "features": features_data
    }

    return render_template("data_exploration/view.html", data=payload)
