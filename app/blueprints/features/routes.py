from __future__ import annotations

from flask import jsonify
from . import bp
from app.services.features_extract_service import start_extract_features_run
from app.services.pipeline_service import pipeline_manager


@bp.post("/features/extract/<group>")
def extract_features_group(group: str):
    run_id = start_extract_features_run(group=group, filename=None)
    return jsonify({"run_id": run_id}), 202


@bp.post("/features/extract/<group>/<path:filename>")
def extract_features_file(group: str, filename: str):
    run_id = start_extract_features_run(group=group, filename=filename)
    return jsonify({"run_id": run_id}), 202


@bp.get("/features/stream/<run_id>")
def features_stream(run_id: str):
    run = pipeline_manager.get_run(run_id)
    if run is None:
        return jsonify({"error": "run_not_found"}), 404

    def event_stream():
        yield "retry: 1000\n\n"
        while True:
            try:
                line = run.q.get(timeout=15)
            except Exception:
                yield ": keep-alive\n\n"
                continue

            if line == "__PIPELINE_DONE__":
                yield "event: done\ndata: features_finished\n\n"
                break

            safe = str(line).replace("\r", "")
            yield f"data: {safe}\n\n"

    from flask import Response, stream_with_context
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")