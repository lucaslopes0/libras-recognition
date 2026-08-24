"""Rotas de autenticação — registro, login, logout, sessão atual."""
from __future__ import annotations

import secrets

from flask import Blueprint, jsonify, request

from libras.api.extensions import bcrypt, db
from libras.api.models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Dados obrigatorios."}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"message": "Usuario e senha obrigatorios."}), 400

    if len(username) < 2 or len(username) > 20:
        return jsonify({"message": "Usuario deve ter entre 2 e 20 caracteres."}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Este nome de usuario ja esta em uso."}), 409

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    db.session.add(User(username=username, password=hashed))
    db.session.commit()

    return jsonify({"message": "Conta criada com sucesso!"}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Dados obrigatorios."}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"message": "Usuario e senha obrigatorios."}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Usuario ou senha incorretos."}), 401

    user.token = secrets.token_hex(32)
    db.session.commit()

    return jsonify({
        "access_token": user.token,
        "username": user.username,
        "message": "Login bem-sucedido!",
    }), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    return jsonify({"message": "Logout realizado."}), 200


@auth_bp.route("/me", methods=["GET"])
def me():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        user = User.query.filter_by(token=token).first()
        if user:
            return jsonify({"username": user.username, "id": user.id}), 200
    return jsonify({"message": "Nao autenticado."}), 401
