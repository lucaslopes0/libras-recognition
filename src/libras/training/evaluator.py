"""Avaliação do modelo treinado."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from tensorflow.keras.models import load_model

from libras.config import PATHS
from libras.utils.io import load_json


def _top_k_accuracy(y_true_int: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    top_k = tf.keras.metrics.sparse_top_k_categorical_accuracy(
        y_true_int, y_pred_probs, k=k,
    )
    return float(np.mean(top_k.numpy()))


def _plot_confusion(cm: np.ndarray, classes: list[str], save_path: Path) -> None:
    n = len(classes)
    figsize = (max(10, n // 2), max(8, n // 2))
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, annot=(n <= 30), fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n // 20))
    plt.yticks(rotation=0, fontsize=max(6, 10 - n // 20))
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate() -> dict:
    """Avalia o melhor modelo salvo e gera matriz de confusão."""
    print("=" * 55)
    print("  AVALIAÇÃO — Reconhecimento de Libras")
    print("=" * 55)

    models_dir = PATHS["models"]
    model_path = models_dir / "best_model.keras"
    meta_path  = models_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    meta = load_json(meta_path)
    classes = meta["classes"]

    print(f"Modelo:  {model_path}")
    print(f"Classes: {len(classes)}\n")

    model = load_model(model_path)
    test_data = np.load(models_dir / "test_data.npz")
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    print(f"Amostras de teste: {len(X_test)}")

    print("\nPredizendo...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = {
        "accuracy_top_1": float(accuracy_score(y_true, y_pred)),
        "accuracy_top_3": _top_k_accuracy(y_true, y_pred_probs, k=3),
        "accuracy_top_5": _top_k_accuracy(y_true, y_pred_probs, k=5),
    }

    print("\n" + "─" * 40)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}  ({v * 100:.2f}%)")
    print("─" * 40)

    print("\nRelatório por classe:")
    report = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0,
    )
    print(report)

    if len(classes) <= 80:
        cm = confusion_matrix(y_true, y_pred)
        cm_path = PATHS["reports"] / "confusion_matrix.png"
        _plot_confusion(cm, classes, cm_path)
        print(f"✓  Matriz salva: {cm_path}")

    return metrics
