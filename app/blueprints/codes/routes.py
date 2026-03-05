import os
from pathlib import Path
from flask import render_template, abort, Blueprint, current_app
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
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

@bp.route('/codes')
@bp.route('/codes/<folder>/<filename>')
def view_code(folder=None, filename=None):
    # Pasta raiz do projeto (app está dentro, pre_proccess e extract_features ao lado)
    project_root = Path(current_app.root_path).parent
    
    code_content = ""
    pygments_css = HtmlFormatter().get_style_defs('.highlight')
    selected_file = f"{folder}/{filename}" if folder else None

    # Validação de Segurança (Item 1)
    if folder and filename:
        if folder not in ALLOWED_CODES or filename not in ALLOWED_CODES[folder]:
            abort(403, "Acesso negado a este arquivo.")
        
        file_path = project_root / folder / filename
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Destaque de Sintaxe (Item 2)
                code_content = highlight(content, PythonLexer(), HtmlFormatter(linenos=True, full=True, style="github-dark"))
        else:
            abort(404, "Arquivo físico não encontrado.")

    return render_template(
        "codes/view.html",
        tree=ALLOWED_CODES,
        code_content=code_content,
        pygments_css=pygments_css,
        selected_file=selected_file
    )