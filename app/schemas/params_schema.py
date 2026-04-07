from marshmallow import Schema, fields


class ParamsSchema(Schema):
    SR = fields.Float(required=True)
    duration = fields.Float(required=True)
    f_low_woman = fields.Float(required=True)
    f_high_woman = fields.Float(required=True)
    f_low_man = fields.Float(required=True)
    f_high_man = fields.Float(required=True)
    target_db = fields.Float(required=True)
    path_audio = fields.Str(required=True)
    path_demographics = fields.Str(required=True)
    tsallis_q = fields.Float(required=True)


class ParamsUpdateSchema(ParamsSchema):
    # permite PATCH parcial se quiser (aqui deixei igual ao completo por simplicidade)
    pass