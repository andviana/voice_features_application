from dataclasses import dataclass
from threading import RLock


@dataclass
class ParamsSnapshot:
    SR: float
    duration: float
    f_low_woman: float
    f_high_woman: float
    f_low_man: float
    f_high_man: float
    target_db: float
    path_audio: str
    path_demographics: str
    tsallis_q: float


class ParamsSingleton:
    def __init__(self):
        self._lock = RLock()
        self._value: ParamsSnapshot | None = None

    def set(self, snapshot: ParamsSnapshot) -> None:
        with self._lock:
            self._value = snapshot

    def get(self) -> ParamsSnapshot:
        with self._lock:
            if self._value is None:
                raise RuntimeError("ParamsSingleton não inicializado")
            return self._value


params_singleton = ParamsSingleton()