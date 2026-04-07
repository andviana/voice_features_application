from ..extensions import db


class Params(db.Model):
    __tablename__ = "params"

    id = db.Column(db.Integer, primary_key=True)  # sempre 1 (single row)
    SR = db.Column(db.Float, nullable=False, default=44100.0)
    duration = db.Column(db.Float, nullable=False, default=2.0)
    f_low_woman = db.Column(db.Float, nullable=False, default=100.0)
    f_high_woman = db.Column(db.Float, nullable=False, default=600.0)
    f_low_man = db.Column(db.Float, nullable=False, default=75.0)
    f_high_man = db.Column(db.Float, nullable=False, default=300.0)
    target_db = db.Column(db.Float, nullable=False, default=-1.0)
    path_audio = db.Column(db.String(500), nullable=False, default="data/audio_raw")
    path_demographics = db.Column(db.String(500), nullable=False, default="data/metadata")
    tsallis_q = db.Column(db.Float, nullable=False, default=1.3)


