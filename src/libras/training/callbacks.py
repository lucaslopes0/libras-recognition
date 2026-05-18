"""Callbacks padrão para o treino."""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard,
)


def build_callbacks(
    model_path: Path,
    logs_dir: Path,
    early_stopping_patience: int = 30,
    reduce_lr_patience: int = 10,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list[Callback]:
    """
    Lista padrão de callbacks:

    • EarlyStopping     — para o treino se não houver melhora
    • ReduceLROnPlateau — reduz learning rate quando estagna
    • ModelCheckpoint   — salva o melhor modelo automaticamente
    • TensorBoard       — `tensorboard --logdir artifacts/logs`
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=str(logs_dir), histogram_freq=1),
    ]
