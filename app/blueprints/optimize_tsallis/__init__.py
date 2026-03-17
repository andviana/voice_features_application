from flask import Blueprint

bp = Blueprint("optimize_tsallis", __name__)

from . import routes 