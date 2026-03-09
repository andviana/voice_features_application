from flask import Blueprint

bp = Blueprint("scientific_analysis", __name__, url_prefix="/scientific-analysis")

from . import routes