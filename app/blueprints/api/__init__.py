from flask import Blueprint
from flask_smorest import Blueprint as SmorestBlueprint

api_bp = Blueprint("api_root", __name__)  # só para organização se quiser

params_api = SmorestBlueprint("params_api", __name__, url_prefix="/api/params", description="Params")
pipeline_api = SmorestBlueprint("pipeline_api", __name__, url_prefix="/api/pipeline", description="Pipeline")

from . import params_api as _p 
from . import pipeline_api as _q 