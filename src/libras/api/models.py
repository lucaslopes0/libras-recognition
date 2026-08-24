"""Modelos de banco de dados da API."""
from __future__ import annotations

from libras.api.extensions import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    token = db.Column(db.String(64), unique=True, nullable=True)
