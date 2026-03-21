from flask import Blueprint

bp = Blueprint("data_exploration", __name__, url_prefix="/data-exploration")

from . import routes
