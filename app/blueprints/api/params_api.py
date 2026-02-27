from . import params_api
from ...repositories.params_repository import ParamsRepository
from ...services.params_service import ParamsService
from ...schemas.params_schema import ParamsSchema, ParamsUpdateSchema

service = ParamsService(ParamsRepository())


@params_api.route("/")
class ParamsResource:
    @params_api.response(200, ParamsSchema)
    def get(self):
        obj = service.get()
        return {
            "SR": obj.SR,
            "duration": obj.duration,
            "f_low_woman": obj.f_low_woman,
            "f_high_woman": obj.f_high_woman,
            "f_low_man": obj.f_low_man,
            "f_high_man": obj.f_high_man,
            "target_db": obj.target_db,
            "path_audio": obj.path_audio,
            "path_demographics": obj.path_demographics,
        }


    @params_api.arguments(ParamsUpdateSchema)
    @params_api.response(200, ParamsSchema)
    def put(self, args):
        obj = service.update(args)
        return {
            "SR": obj.SR,
            "duration": obj.duration,
             "f_low_woman": obj.f_low_woman,
            "f_high_woman": obj.f_high_woman,
            "f_low_man": obj.f_low_man,
            "f_high_man": obj.f_high_man,
            "target_db": obj.target_db,
            "path_audio": obj.path_audio,
            "path_demographics": obj.path_demographics,
        }
