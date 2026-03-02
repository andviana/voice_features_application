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

    # ✅ silencia numba (responsável pelo flood que você mostrou)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
    logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)

    # (opcional) outras libs comuns de áudio que podem ficar verbosas
    logging.getLogger("librosa").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)