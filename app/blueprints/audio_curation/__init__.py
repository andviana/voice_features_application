from flask import Blueprint

bp = Blueprint('audio_curation', __name__, url_prefix='/audio-curation')

from . import routes
