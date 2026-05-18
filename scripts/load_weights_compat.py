"""
Carrega o modelo recriando a arquitetura local e injetando os pesos.

Use quando houver incompatibilidade de versão entre TF do treino e do uso.

Uso:
    python scripts/load_weights_compat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Dense, Dropout, LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from libras.config import PATHS
from libras.utils.io import load_json


def build_compatible_model(num_classes: int) -> Sequential:
    """
    REPRODUZ EXATAMENTE a arquitetura do Colab (treino_colab.ipynb célula 8).
    Importante: tem que ser idêntica para os pesos casarem!
    """
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True, activation='tanh',
                 recurrent_dropout=0.2),
            input_shape=(30, 1662),
        ),
        BatchNormalization(),
        Dropout(0.5),

        Bidirectional(LSTM(128, return_sequences=False, activation='tanh',
                           recurrent_dropout=0.2)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main() -> None:
    print("=" * 55)
    print("  Reconstruindo modelo + carregando pesos")
    print("=" * 55)

    models_dir = PATHS["models"]

    # Carrega metadata
    meta = load_json(models_dir / "metadata.json")
    num_classes = meta["num_classes"]
    print(f"Classes: {num_classes}")
    print(f"Lista:   {meta['classes']}")

    # Reconstrói arquitetura
    print("\n🔧 Reconstruindo arquitetura...")
    model = build_compatible_model(num_classes)
    model.summary()

    # Carrega pesos
    weights_path = models_dir / "best_model.weights.h5"
    if not weights_path.exists():
        print(f"\n❌ Arquivo de pesos não encontrado: {weights_path}")
        print("   Volte ao Colab e gere best_model.weights.h5")
        sys.exit(1)

    print(f"\n📥 Carregando pesos de {weights_path}...")
    model.load_weights(str(weights_path))

    # Salva no formato local que funciona
    output = models_dir / "best_model_local.keras"
    model.save(str(output))
    print(f"\n✅ Modelo salvo em formato compatível: {output}")
    print("   Renomeie para best_model.keras para usar com o pipeline normal.")


if __name__ == "__main__":
    main()