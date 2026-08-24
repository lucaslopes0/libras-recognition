"""Factory da API Flask — autenticação + reconhecimento de sinais e letras."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import Flask, jsonify
from flask_cors import CORS

from libras.api.extensions import bcrypt, db
from libras.config import PATHS


def create_app() -> Flask:
    app = Flask(__name__)

    # SECRET_KEY não assina nada sensível hoje (auth usa token opaco salvo
    # no banco, não sessão de cookie assinada), mas fica configurável via
    # env var em vez de hardcoded no código-fonte.
    app.config["SECRET_KEY"] = os.environ.get("LIBRAS_SECRET_KEY", "dev-only-change-me")

    PATHS["instance"].mkdir(parents=True, exist_ok=True)
    db_path = PATHS["instance"] / "site.db"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    CORS(app, supports_credentials=True)

    db.init_app(app)
    bcrypt.init_app(app)

    with app.app_context():
        from libras.api import models  # noqa: F401 — registra o model User antes do create_all
        db.create_all()
        print(f"Banco de dados: {db_path}")

    from libras.api.auth import auth_bp
    from libras.api.letter_routes import letter_bp
    from libras.api.sign_routes import sign_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(sign_bp)
    app.register_blueprint(letter_bp)

    @app.route("/health", methods=["GET"])
    def health():
        from libras.api import letter_routes, sign_routes

        return jsonify({
            "status": "ok",
            "lstm_loaded": sign_routes.is_loaded(),
            "lstm_classes": sign_routes.num_classes(),
            "letter_loaded": letter_routes.is_loaded(),
            "letter_classes": letter_routes.num_classes(),
            "buffer_size": sign_routes.buffer_size(),
        })

    return app
