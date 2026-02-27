from ..globals import ParamsSnapshot, params_singleton
from ..repositories.params_repository import ParamsRepository


class ParamsService:
    def __init__(self, repo: ParamsRepository):
        self.repo = repo

    def load_to_singleton(self) -> None:
        obj = self.repo.get_single()
        params_singleton.set(
            ParamsSnapshot(
                SR=obj.SR,
                duration=obj.duration,
                f_low_woman=obj.f_low_woman,
                f_high_woman=obj.f_high_woman,
                f_low_man=obj.f_low_man,
                f_high_man=obj.f_high_man,
                target_db =obj.target_db,
                path_audio=obj.path_audio,
                path_demographics=obj.path_demographics,
            )
        )

    def get(self):
        return self.repo.get_single()

    def update(self, data: dict):
        obj = self.repo.update_single(data)
        self.load_to_singleton()  # mantém singleton sempre atualizado
        return obj