from flask import Blueprint

bp = Blueprint("params", __name__)

from . import routes