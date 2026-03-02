import contextlib
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Optional


class QueueWriter:
    """
    Recebe writes do print (stdout) e empurra linhas completas para a Queue.
    """
    def __init__(self, q: queue.Queue[str]):
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.q.put(line)
        return len(s)

    def flush(self) -> None:
        # Se sobrou algo sem \n, não força, deixa quando completar.
        pass


@dataclass
class PipelineRun:
    run_id: str
    q: queue.Queue[str]
    thread: threading.Thread
    started_at: float
    finished: bool = False
    error: Optional[str] = None


class PipelineManager:
    def __init__(self):
        self._runs: Dict[str, PipelineRun] = {}
        self._lock = threading.Lock()

    def start(self, pipeline_fn: Callable[[], None]) -> str:
        run_id = uuid.uuid4().hex
        q: queue.Queue[str] = queue.Queue()

        def runner():
            try:
                writer = QueueWriter(q)
                with contextlib.redirect_stdout(writer):
                    pipeline_fn()
                # finaliza: sinal de fim
                q.put("__PIPELINE_DONE__")
            except Exception as e:
                q.put(f"ERRO: {type(e).__name__}: {e}")
                q.put("__PIPELINE_DONE__")
                with self._lock:
                    self._runs[run_id].error = f"{type(e).__name__}: {e}"
            finally:
                with self._lock:
                    self._runs[run_id].finished = True

        t = threading.Thread(target=runner, daemon=True)
        run = PipelineRun(run_id=run_id, q=q, thread=t, started_at=time.time())

        with self._lock:
            self._runs[run_id] = run

        t.start()
        return run_id

    def get_run(self, run_id: str) -> PipelineRun | None:
        with self._lock:
            return self._runs.get(run_id)


pipeline_manager = PipelineManager()


# Exemplo de pipeline interno (substitua pelo seu)
def example_pipeline():
    from ..globals import params_singleton

    p = params_singleton.get()
    print(f"Iniciando pipeline com SR={int(p.SR)} duration={p.duration}")
    for i in range(1, 11):
        print(f"Processando etapa {i}/10 ...")
        time.sleep(0.4)
    print("Finalizado com sucesso.")