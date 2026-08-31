"""
LSTM Bidirecional para reconhecimento de sinais.

Arquitetura escolhida porque:
  • LSTM Bidirecional capta padrões temporais nos dois sentidos
  • BatchNormalization estabiliza o treino
  • Dropout regulariza contra overfitting
  • Activation 'tanh' nas LSTMs (padrão correto para gates)
"""

from __future__ import annotations

from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Dense, Dropout, LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from libras.utils.mediapipe_holistic import TOTAL_DIM


def build_bidirectional_lstm(
    num_classes: int,
    seq_len: int = 30,
    features: int = TOTAL_DIM,
    learning_rate: float = 1e-3,
) -> Sequential:
    """
    Constrói rede LSTM Bidirecional para classificação de sinais.

    Entrada: (batch, seq_len, features)
    Saída:   (batch, num_classes) — probabilidades softmax
    """
    model = Sequential(
        [
            # Bloco 1 — padrões de curto prazo
            Bidirectional(
                LSTM(64, return_sequences=True, activation="tanh"),
                input_shape=(seq_len, features),
            ),
            BatchNormalization(),
            Dropout(0.3),

            # Bloco 2 — padrões de médio prazo
            Bidirectional(LSTM(128, return_sequences=True, activation="tanh")),
            BatchNormalization(),
            Dropout(0.3),

            # Bloco 3 — comprime para vetor final
            Bidirectional(LSTM(64, return_sequences=False, activation="tanh")),
            BatchNormalization(),
            Dropout(0.3),

            # Cabeça classificadora
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ],
        name="libras_bi_lstm",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
