import os
from pathlib import Path
from flask import render_template, abort, Blueprint, current_app
from pygments import highlight
from pygments.lexers import PythonLexer, get_lexer_by_name, get_lexer_for_filename
from pygments.util import ClassNotFound
from pygments.formatters import HtmlFormatter
from app.utils.path_utils import PathUtils
from . import bp


# Lista Branca de Arquivos Autorizados (Item 1 do pedido)
ALLOWED_CODES = {
    "pre_proccess": ["analise.py", "filters.py", "normalize.py", "windowing.py"],
    "extract_features": [
        "f0_features.py", "formants_lpc.py", "hnr_features.py", 
        "jitter_features.py", "mfcc_features.py", "shimmer_features.py",
        "spectral_features.py", "tsallis_amplitude_hist.py", "tsallis_f0_hist.py"
    ]
}

def get_data_files():
    """Busca dinamicamente arquivos CSV e XLSX na pasta data."""
    data_root = PathUtils.project_root() / "data"
    if not data_root.exists():
        return []
    
    files = []
    # Busca recursiva por .csv e .xlsx
    for ext in ["**/*.csv", "**/*.xlsx"]:
        for f in data_root.glob(ext):
            if not f.name.startswith('.'):
                # Retorna o caminho relativo à pasta data
                files.append(str(f.relative_to(data_root)))
    return sorted(files)

@bp.route('/codes')
@bp.route('/codes/<folder>/<path:filename>')
def view_code(folder=None, filename=None):
    # Pasta raiz do projeto (app está dentro, pre_proccess e extract_features ao lado)
    project_root = PathUtils.project_root()
    
    # Adiciona a pasta data dinamicamente à árvore
    tree = ALLOWED_CODES.copy()
    data_files = get_data_files()
    if data_files:
        tree["data"] = data_files

    code_content = ""
    table_data = None
    pygments_css = HtmlFormatter().get_style_defs('.highlight')
    selected_file = f"{folder}/{filename}" if folder else None
    file_type = "Python Source"

    # Validação de Segurança (Item 1)
    if folder and filename:
        is_allowed = False
        if folder in ALLOWED_CODES and filename in ALLOWED_CODES[folder]:
            is_allowed = True
            file_path = project_root / folder / filename
        elif folder == "data" and filename in data_files:
            is_allowed = True
            file_path = project_root / "data" / filename
        
        if not is_allowed:
            abort(403, "Acesso negado a este arquivo.")
        
        if file_path.exists():
            ext = file_path.suffix.lower()
            if ext == ".py":
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    code_content = highlight(content, PythonLexer(), HtmlFormatter(linenos=True, full=True, style="github-dark"))
                file_type = "Python Source"
            elif ext == ".csv":
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path, nrows=100)
                    table_data = {
                        "columns": df.columns.tolist(),
                        "rows": df.values.tolist()
                    }
                    file_type = "CSV Data"
                except Exception as e:
                    code_content = f"<div class='p-4 text-red-500'>Erro ao ler arquivo CSV: {str(e)}</div>"
                    file_type = "CSV Error"
            elif ext == ".xlsx":
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path, nrows=100)
                    table_data = {
                        "columns": df.columns.tolist(),
                        "rows": df.values.tolist()
                    }
                    file_type = "Excel Data"
                except Exception as e:
                    code_content = f"<div class='p-4 text-red-500'>Erro ao ler arquivo Excel: {str(e)}</div>"
                    file_type = "Excel Error"
        else:
            abort(404, "Arquivo físico não encontrado.")

    return render_template(
        "codes/view.html",
        tree=tree,
        code_content=code_content,
        table_data=table_data,
        pygments_css=pygments_css,
        selected_file=selected_file,
        file_type=file_type
    )