from flask import Blueprint

bp = Blueprint('group_comparison', __name__)

from . import routes
