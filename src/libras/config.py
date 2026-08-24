"""
Loader de configuração.

Carrega YAML em `configs/` e expõe caminhos padronizados do projeto.
Toda configuração passa por aqui — nenhum hardcode espalhado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ── Diretórios base ──────────────────────────────────────────
# Resolve em tempo de import: src/libras/config.py → raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "configs":               PROJECT_ROOT / "configs",
    "data_raw":              PROJECT_ROOT / "data" / "raw",
    "data_interim":          PROJECT_ROOT / "data" / "interim",
    "data_processed":        PROJECT_ROOT / "data" / "processed",
    "data_raw_custom":       PROJECT_ROOT / "data" / "raw_custom",
    "data_processed_custom": PROJECT_ROOT / "data" / "processed_custom",
    "data_letters_captured": PROJECT_ROOT / "data" / "raw_custom" / "letters",
    "artifacts":             PROJECT_ROOT / "artifacts",
    "models":                PROJECT_ROOT / "artifacts" / "models",
    "models_letters":        PROJECT_ROOT / "artifacts" / "models" / "letters",
    "logs":                  PROJECT_ROOT / "artifacts" / "logs",
    "reports":               PROJECT_ROOT / "artifacts" / "reports",
    "instance":              PROJECT_ROOT / "instance",
}


@dataclass
class Config:
    """Representação tipada da configuração carregada do YAML."""
    raw: dict[str, Any]

    @property
    def dataset(self) -> dict:
        return self.raw["dataset"]

    @property
    def preprocess(self) -> dict:
        return self.raw["preprocess"]

    @property
    def training(self) -> dict:
        return self.raw["training"]

    @property
    def inference(self) -> dict:
        return self.raw["inference"]

    @property
    def feedback(self) -> dict:
        return self.raw["feedback"]


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Carrega configuração YAML."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(raw=data)


def ensure_directories() -> None:
    """Cria todos os diretórios padrão do projeto."""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
