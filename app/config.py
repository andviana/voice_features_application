import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]  # raiz do projeto (meu_projeto_flask/)
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).resolve()
AUDIO_RAW_DIR = (DATA_DIR / "audio_raw").resolve()

@dataclass
class BaseConfig:
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dissertacaomestradoppgee")
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    DATA_DIR = str(DATA_DIR)
    AUDIO_RAW_DIR = str(AUDIO_RAW_DIR)

    # Swagger / OpenAPI
    API_TITLE: str = "Voice Feautures"
    API_VERSION: str = "v1"
    OPENAPI_VERSION: str = "3.0.3"
    OPENAPI_URL_PREFIX: str = "/api"
    OPENAPI_SWAGGER_UI_PATH: str = "/swagger-ui"
    OPENAPI_SWAGGER_UI_URL: str = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"



@dataclass
class DevConfig(BaseConfig):
    DEBUG: bool = True
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///app.db",  # cria app.db na raiz do projeto
    )


@dataclass
class ProdConfig(BaseConfig):
    DEBUG: bool = False
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///app.db",
    )


def get_config():
    env = os.getenv("FLASK_ENV", "development").lower()
    return ProdConfig if env in ("production", "prod") else DevConfig