from flask import render_template, request, redirect, url_for, flash
from . import bp


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        # Aqui você conectaria com seu serviço de autenticação (ex.: checar hash no banco)
        if not email or not password:
            flash("Informe e-mail e senha.", "error")
            return redirect(url_for("auth.login"))

        flash("Login (exemplo) recebido com sucesso.", "success")
        return redirect(url_for("main.index"))

    return render_template("auth/login.html")