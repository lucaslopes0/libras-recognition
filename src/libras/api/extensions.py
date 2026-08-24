"""Extensões Flask compartilhadas entre os módulos da API (evita import circular)."""
from __future__ import annotations

from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
bcrypt = Bcrypt()
