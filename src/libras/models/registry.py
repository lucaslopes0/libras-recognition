"""
Registry de arquiteturas.

Permite adicionar facilmente novos modelos (ex: Transformer, CNN+LSTM)
sem mexer no código de treino. Basta registrar aqui.
"""

from __future__ import annotations

from typing import Callable

from tensorflow.keras.models import Model

from libras.models.lstm import build_bidirectional_lstm

MODEL_REGISTRY: dict[str, Callable[..., Model]] = {
    "bi_lstm": build_bidirectional_lstm,
}


def build_model(name: str, **kwargs) -> Model:
    """
    Constrói modelo pelo nome registrado.

    >>> model = build_model("bi_lstm", num_classes=100)
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Modelo '{name}' não registrado. Disponíveis: {available}")
    return MODEL_REGISTRY[name](**kwargs)
