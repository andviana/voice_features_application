from flask import render_template, request, jsonify
import shutil
import os
import pandas as pd
from pathlib import Path
from app.utils.path_utils import PathUtils
from . import bp

@bp.get('/')
def index():
    raw_root = PathUtils.raw_root()
    manifest_path = PathUtils.manifest_filepath()
    
    # Load manifest once to memory for quick lookup
    df = pd.DataFrame()
    if manifest_path.exists():
        try:
            df = pd.read_csv(manifest_path)
            possible_cols = ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id")
            
            file_col = None
            for c in df.columns:
                if str(c).strip().lower() in possible_cols:
                    file_col = c
                    break
            
            if file_col:
                df['_lookup_key'] = df[file_col].astype(str).str.strip().str.replace(".wav", "", regex=False)
        except Exception:
            pass

    def get_demographics(filename):
        if df.empty or '_lookup_key' not in df.columns:
            return {"age": "-", "sex": "-"}
            
        key = str(filename).replace(".wav", "")
        row = df[df['_lookup_key'] == key]
        
        if row.empty:
            return {"age": "-", "sex": "-"}
            
        data = row.iloc[0]
        
        age = "-"
        for c in ["age", "idade"]:
            if c in data and not pd.isna(data[c]):
                age = str(data[c])
                break
                
        sex = "-"
        for c in ["sex", "sexo", "gender", "genero"]:
            if c in data and not pd.isna(data[c]):
                sex = str(data[c])
                break
                
        return {"age": age, "sex": sex}

    groups = {}
    total_count = 0
    for group in PathUtils.ALLOWED_GROUPS_LIST:
        group_dir = PathUtils.safe_group_dir(raw_root, group)
        files_data = []
        if group_dir.exists():
            for filepath in sorted(group_dir.glob("*.wav"), key=lambda p: p.name.lower()):
                demo = get_demographics(filepath.name)
                files_data.append({
                    "filename": filepath.name,
                    "age": demo["age"],
                    "sex": demo["sex"]
                })
                total_count += 1
        groups[group] = files_data

    return render_template('audio_curation/index.html', groups=groups, total_count=total_count)


@bp.post('/reject')
def reject_audios():
    data = request.get_json()
    if not data or 'selections' not in data or len(data['selections']) == 0:
        return jsonify({"status": "error", "message": "Nenhum arquivo selecionado."}), 400
        
    selections = data['selections']
    raw_root = PathUtils.raw_root()
    rej_root = PathUtils.rejected_root()
    
    moved_count = 0
    errors = []
    
    for item in selections:
        filename = item.get("filename")
        group = item.get("group")
        
        if not filename or group not in PathUtils.ALLOWED_GROUPS_LIST:
            continue
            
        try:
            src = PathUtils.safe_wav_path(raw_root, group, filename)
            
            # Garante que o diretório de destino existe
            dst_dir = (rej_root / group).resolve()
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst = dst_dir / filename
            
            shutil.move(str(src), str(dst))
            moved_count += 1
        except Exception as e:
            errors.append(f"Erro movendo {filename}: {str(e)}")

    if errors:
        return jsonify({
            "status": "warning", 
            "message": f"{moved_count} movidos com sucesso. Alguns erros ocorreram.", 
            "errors": errors
        }), 207

    return jsonify({"status": "success", "message": f"{moved_count} arquivos movidos para Rejeitados com sucesso."})


@bp.post('/rebuild-manifest')
def rebuild_manifest():
    manifest_path = PathUtils.manifest_filepath()
    raw_root = PathUtils.raw_root()
    demog_xlsx = PathUtils.project_root() / "data" / "metadata" / "Demographics_age_sex.xlsx"
    
    try:
        # 1. Load Master Demographics
        demog_map = {}
        if demog_xlsx.exists():
            df_demog = pd.read_excel(demog_xlsx)
            # Standardize columns: Sample ID, Age, Sex
            for _, row in df_demog.iterrows():
                sid = str(row.get('Sample ID', '')).strip()
                if sid:
                    demog_map[sid] = {
                        "age": row.get('Age'),
                        "sex": row.get('Sex')
                    }

        # 2. Load Current Manifest (if exists)
        df_old = pd.DataFrame()
        if manifest_path.exists():
            df_old = pd.read_csv(manifest_path)
            # Ensure recording_id is standardized
            if 'recording_id' in df_old.columns:
                df_old['recording_id'] = df_old['recording_id'].astype(str).str.strip()

        # 3. Scan Files on Disk
        found_files = [] # list of dicts for the new manifest
        
        for group in PathUtils.ALLOWED_GROUPS_LIST:
            group_dir = PathUtils.safe_group_dir(raw_root, group)
            if not group_dir.exists():
                continue
                
            for filepath in group_dir.glob("*.wav"):
                fname = filepath.name
                rid = fname.replace(".wav", "")
                
                # Try to get data from old manifest first to preserve info
                existing_row = {}
                if not df_old.empty and 'recording_id' in df_old.columns:
                    hit = df_old[df_old['recording_id'] == rid]
                    if not hit.empty:
                        existing_row = hit.iloc[0].to_dict()
                
                # Fetch demographics from master XLSX (overwrites or fills)
                dg = demog_map.get(rid, {})
                age = dg.get("age", existing_row.get("age", "-"))
                sex = dg.get("sex", existing_row.get("sex", "-"))

                # Deriva participant_id: AH_PID_GUID -> PID
                # manifest usually has: participant_id, recording_id, group, age, sex, wav_path, sampling_rate
                pid = existing_row.get("participant_id")
                if not pid:
                    parts = rid.split("_")
                    if len(parts) > 1:
                        pid = parts[1]
                    else:
                        pid = rid
                
                new_row = {
                    "participant_id": pid,
                    "recording_id": rid,
                    "group": group.replace("_AH", ""), # Standardization: HC_AH -> HC
                    "age": age,
                    "sex": sex,
                    "wav_path": str(filepath.resolve()),
                    "sampling_rate": existing_row.get("sampling_rate", 48000)
                }
                found_files.append(new_row)

        if not found_files:
            return jsonify({"status": "warning", "message": "Nenhum arquivo .wav encontrado nas pastas brutas."}), 200

        # 4. Save New Manifest
        df_new = pd.DataFrame(found_files)
        # Ensure column order matches standard
        cols = ["participant_id", "recording_id", "group", "age", "sex", "wav_path", "sampling_rate"]
        df_new = df_new[cols]
        
        df_new.to_csv(manifest_path, index=False)
        
        return jsonify({
            "status": "success", 
            "message": f"Manifest reconstruído e enriquecido com sucesso! Total de {len(df_new)} arquivos registrados."
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Falha ao reconstruir manifest: {str(e)}"}), 500


@bp.get('/rejected')
def rejected_index():
    rej_root = PathUtils.rejected_root()
    manifest_path = PathUtils.manifest_filepath()
    
    # Load manifest once to memory for quick lookup
    df = pd.DataFrame()
    if manifest_path.exists():
        try:
            df = pd.read_csv(manifest_path)
            possible_cols = ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id")
            
            file_col = None
            for c in df.columns:
                if str(c).strip().lower() in possible_cols:
                    file_col = c
                    break
            
            if file_col:
                df['_lookup_key'] = df[file_col].astype(str).str.strip().str.replace(".wav", "", regex=False)
        except Exception:
            pass

    def get_demographics(filename):
        if df.empty or '_lookup_key' not in df.columns:
            return {"age": "-", "sex": "-"}
            
        key = str(filename).replace(".wav", "")
        row = df[df['_lookup_key'] == key]
        
        if row.empty:
            return {"age": "-", "sex": "-"}
            
        data = row.iloc[0]
        
        age = "-"
        for c in ["age", "idade"]:
            if c in data and not pd.isna(data[c]):
                age = str(data[c])
                break
                
        sex = "-"
        for c in ["sex", "sexo", "gender", "genero"]:
            if c in data and not pd.isna(data[c]):
                sex = str(data[c])
                break
                
        return {"age": age, "sex": sex}

    groups = {}
    total_count = 0
    for group in PathUtils.ALLOWED_GROUPS_LIST:
        group_dir = (rej_root / group).resolve()
        files_data = []
        if group_dir.exists():
            for filepath in sorted(group_dir.glob("*.wav"), key=lambda p: p.name.lower()):
                demo = get_demographics(filepath.name)
                files_data.append({
                    "filename": filepath.name,
                    "age": demo["age"],
                    "sex": demo["sex"]
                })
                total_count += 1
        groups[group] = files_data

    return render_template('audio_curation/rejected.html', groups=groups, total_count=total_count)


@bp.post('/restore')
def restore_audios():
    data = request.get_json()
    if not data or 'selections' not in data or len(data['selections']) == 0:
        return jsonify({"status": "error", "message": "Nenhum arquivo selecionado."}), 400
        
    selections = data['selections']
    raw_root = PathUtils.raw_root()
    rej_root = PathUtils.rejected_root()
    
    moved_count = 0
    errors = []
    
    for item in selections:
        filename = item.get("filename")
        group = item.get("group")
        
        if not filename or group not in PathUtils.ALLOWED_GROUPS_LIST:
            continue
            
        try:
            # Source no caso de restore recai na lixeira
            src = (rej_root / group / filename).resolve()
            if not src.exists() or src.suffix.lower() != ".wav":
                continue
            
            # Garante que o diretório raw (destino original) existe
            dst_dir = PathUtils.safe_group_dir(raw_root, group)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / filename
            
            shutil.move(str(src), str(dst))
            moved_count += 1
        except Exception as e:
            errors.append(f"Erro restaurando {filename}: {str(e)}")

    if errors:
        return jsonify({
            "status": "warning", 
            "message": f"{moved_count} restaurados. Alguns erros ocorreram.", 
            "errors": errors
        }), 207

    return jsonify({"status": "success", "message": f"{moved_count} arquivos restaurados para a pasta bruta."})
