from flask import Blueprint, render_template, jsonify
from pathlib import Path
import pandas as pd

from .services import *
from . import bp


DATA_RAW = Path("data/audio_raw")
DATA_PROCESSED = Path("data/audio_processed")

MANIFEST = Path("data/metadata/manifest.csv")
FEATURES = Path("data/features/dataset_voz_completo.csv")

def _find_filename_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("file_name", "filename", "arquivo", "nome_arquivo", "wav", "audio", "recording_id"):
            return c
    return None

def list_audios():

    data = {"HC_AH": [], "PD_AH": []}

    for group in data.keys():

        folder = DATA_RAW / group

        if not folder.exists():
            continue

        for f in folder.glob("*.wav"):
            data[group].append(f.name)

    return data


@bp.route("/")
def index():

    audios = list_audios()

    return render_template(
        "analysis/list.html",
        audios=audios
    )


@bp.route("/view/<group>/<filename>")
def view(group, filename):

    raw = DATA_RAW / group / filename
    processed = DATA_PROCESSED / group / filename

    y_raw, sr_raw, t_raw = load_audio(raw)
    y_proc, sr_proc, t_proc = load_audio(processed)

    wf_raw = waveform(y_raw, sr_raw)
    wf_proc = waveform(y_proc, sr_proc)

    spec_raw = spectrum(y_raw, sr_raw)
    spec_proc = spectrum(y_proc, sr_proc)

    spectro_raw = spectrogram(y_raw, sr_raw)
    spectro_proc = spectrogram(y_proc, sr_proc)

    butter = butterworth_response(sr_raw)

    psd_raw = psd(y_raw, sr_raw)

    auto = autocorrelation(y_raw)

    hist = amplitude_histogram(y_raw)

    demographics = {}
    features = []

    if MANIFEST.exists():
        df = pd.read_csv(MANIFEST)

        file_col = _find_filename_column(df)

        if file_col:
            key1 = filename
            key2 = filename.replace(".wav", "")

            row = df.loc[
                (df[file_col].astype(str).str.strip() == key1) |
                (df[file_col].astype(str).str.strip() == key2)
            ]

            if not row.empty:
                r = row.iloc[0]
                demographics = {
                    "label": r.get("label", r.get("grupo", r.get("group"))),
                    "age": r.get("age", r.get("idade")),
                    "sex": r.get("sex", r.get("sexo", r.get("gender"))),
                }

    if FEATURES.exists():
        df = pd.read_csv(FEATURES)

        file_col = _find_filename_column(df)

        if file_col:
            key1 = filename
            key2 = filename.replace(".wav", "")

            row = df.loc[
                (df[file_col].astype(str).str.strip() == key1) |
                (df[file_col].astype(str).str.strip() == key2)
            ]

            if not row.empty:
                r = row.iloc[0]

                used = set()

                for col in df.columns:
                    col_s = str(col)

                    if "_mean" in col_s:
                        std_col = col_s.replace("_mean_hz", "_std_hz").replace("_mean", "_std")
                        base = col_s.replace("_mean_hz", "").replace("_mean", "")

                        features.append({
                            "name": base,
                            "mean": r.get(col_s),
                            "std": r.get(std_col) if std_col in df.columns else None,
                        })

                        used.add(col_s)
                        used.add(std_col)

    return render_template(
        "analysis/view.html",

        filename=filename,
        group=group,

        wf_raw=wf_raw,
        wf_proc=wf_proc,

        spec_raw=spec_raw,
        spec_proc=spec_proc,

        spectro_raw=spectro_raw,
        spectro_proc=spectro_proc,

        butter=butter,
        psd=psd_raw,
        autocorr=auto,
        hist=hist,

        demographics=demographics,
        features=features
    )