from flask import Blueprint

bp = Blueprint("audio_processed", __name__)

from . import routes 