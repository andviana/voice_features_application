import time
from flask import Blueprint, Response, render_template, stream_with_context, jsonify
from ...services.pipeline_service import pipeline_manager, example_pipeline

bp = Blueprint("pipeline", __name__)


@bp.get("/pipeline")
def live_page():
    return render_template("pipeline/live.html")


@bp.post("/pipeline/start")
def start_pipeline():
    run_id = pipeline_manager.start(example_pipeline)
    return jsonify({"run_id": run_id}), 202


@bp.get("/pipeline/stream/<run_id>")
def stream_pipeline(run_id: str):
    run = pipeline_manager.get_run(run_id)
    if run is None:
        return jsonify({"error": "run_not_found"}), 404

    def event_stream():
        # recomendações SSE
        yield "retry: 1000\n\n"

        while True:
            try:
                line = run.q.get(timeout=15)
            except Exception:
                # keep-alive a cada ~15s
                yield ": keep-alive\n\n"
                continue

            if line == "__PIPELINE_DONE__":
                yield "event: done\ndata: pipeline_finished\n\n"
                break

            # cada linha vira um evento
            # IMPORTANTE: SSE exige \n\n no fim do evento
            safe = line.replace("\r", "")
            yield f"data: {safe}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")