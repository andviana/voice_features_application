import logging
import sys


def configure_logging(app) -> None:
    level = logging.DEBUG if app.debug else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # Evita handlers duplicados em reload do debug
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Reduz ruído de libs
    logging.getLogger("werkzeug").setLevel(logging.INFO)