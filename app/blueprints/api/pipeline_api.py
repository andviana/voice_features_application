from . import pipeline_api
from ...services.pipeline_service import pipeline_manager, example_pipeline


@pipeline_api.route("/start")
class PipelineStart:
    @pipeline_api.response(202, description="Started")
    def post(self):
        run_id = pipeline_manager.start(example_pipeline)
        return {"run_id": run_id}


@pipeline_api.route("/status/<run_id>")
class PipelineStatus:
    @pipeline_api.response(200)
    def get(self, run_id: str):
        run = pipeline_manager.get_run(run_id)
        if run is None:
            return {"error": "run_not_found"}, 404
        return {"run_id": run_id, "finished": run.finished, "error": run.error}