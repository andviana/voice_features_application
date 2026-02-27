from flask import Blueprint

bp = Blueprint("pipeline", __name__)

from . import routes