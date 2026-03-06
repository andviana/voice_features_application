from flask import Blueprint

bp = Blueprint("compare", __name__)

from . import routes  # noqa