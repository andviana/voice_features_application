from werkzeug.security import generate_password_hash

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.params import Params


def seed_users() -> None:
    """
    Cria usuário admin padrão se não existir.
    """
    admin_email = "admin@example.com"

    existing = db.session.query(User).filter_by(email=admin_email).first()
    if existing:
        print("Usuário admin já existe.")
        return

    user = User(
        email=admin_email,
        password_hash=generate_password_hash("admin123"),
    )

    db.session.add(user)
    db.session.commit()
    print("Usuário admin criado com sucesso.")


def seed_params() -> None:
    """
    Garante que exista a linha singleton id=1 em params.
    """
    params = db.session.get(Params, 1)

    if params:
        print("Params já existem.")
        return

    params = Params(
        id=1,
        SR=44100.0,
        duration=2.0,
        f_low_woman=100.0,
        f_high_woman=600.0,
        f_low_man=75.0,
        f_high_man=300.0,
        target_db=-1.0,
        path_audio="data/audio_raw",
        path_demographics="data/metadata",
    )

    db.session.add(params)
    db.session.commit()
    print("Params iniciais criados.")


def run():
    app = create_app()

    with app.app_context():
        print("Iniciando seed...")
        seed_users()
        seed_params()
        print("Seed finalizado.")


if __name__ == "__main__":
    run()