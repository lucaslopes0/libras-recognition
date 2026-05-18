"""Funções de I/O — salvar/carregar JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Salva dicionário como JSON com encoding UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Carrega JSON UTF-8."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
