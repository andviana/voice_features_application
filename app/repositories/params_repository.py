from ..extensions import db
from ..models.params import Params


class ParamsRepository:
    def get_single(self) -> Params:
        obj = db.session.get(Params, 1)
        if obj is None:
            obj = Params(id=1)
            db.session.add(obj)
            db.session.commit()
        return obj

    def update_single(self, data: dict) -> Params:
        obj = self.get_single()
        for k, v in data.items():
            setattr(obj, k, v)
        db.session.commit()
        return obj