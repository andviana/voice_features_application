from flask import jsonify, render_template, request


def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not_found"}), 404
        return render_template("base.html", content="Página não encontrada"), 404

    @app.errorhandler(500)
    def internal_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal_server_error"}), 500
        return render_template("base.html", content="Erro interno"), 500