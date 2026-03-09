from flask import Flask
from sqlalchemy import inspect

import logging

from .config import get_config
from .extensions import db, migrate, api

from .logging_conf import configure_logging
from .error_handlers import register_error_handlers

from .repositories.params_repository import ParamsRepository
from .services.params_service import ParamsService


def create_app():
    app = Flask(__name__)
    app.config.from_object(get_config())

    configure_logging(app)

    # Extensions
    db.init_app(app)
    migrate.init_app(app, db)
    api.init_app(app)

    # Error handlers
    register_error_handlers(app)

    # Blueprints HTML
    from .blueprints.main import bp as main_bp
    from .blueprints.params import bp as params_bp
    from .blueprints.pipeline import bp as pipeline_bp
    from .blueprints.audio_raw import bp as audio_raw_bp
    from .blueprints.audio_processed import bp as audio_processed_bp
    from .blueprints.features import bp as features_bp
    from .blueprints.codes import bp as codes_bp
    from .blueprints.compare import bp as compare_bp
    from .blueprints.analysis import bp as analysis_bp
    from .blueprints.scientific_analysis import bp as scientific_analysis_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(params_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(audio_raw_bp)
    app.register_blueprint(audio_processed_bp)
    app.register_blueprint(features_bp)
    app.register_blueprint(codes_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(scientific_analysis_bp)

    # Blueprints API (Smorest)
    from .blueprints.api import params_api, pipeline_api
    api.register_blueprint(params_api)
    api.register_blueprint(pipeline_api)

    # Rotas simples de healthcheck
    @app.get("/health")
    def health():
        return {"status": "ok"}, 200

    # Carrega singleton de params no startup (garante row id=1 existe)
    with app.app_context():
        try:
            inspector = inspect(db.engine)
            if inspector.has_table("params"):
                service = ParamsService(ParamsRepository())
                service.load_to_singleton()
            else:
                logging.getLogger(__name__).info(
                    "Tabela 'params' ainda não existe (migrações não aplicadas). Pulando load do singleton."
                )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Não foi possível checar/carregar params no startup: %s", e
            )

    return app