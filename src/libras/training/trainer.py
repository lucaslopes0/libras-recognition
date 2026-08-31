"""Orquestrador do treino com data augmentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from libras.config import PATHS, PROJECT_ROOT, Config
from libras.data.augmentation import augment_training_set
from libras.data.loader import (
    _required_min_samples, list_classes, load_arrays, split_data,
)
from libras.models import build_model
from libras.training.callbacks import build_callbacks
from libras.utils.io import save_json


def _plot_history(history, save_path: Path) -> None:
    """Salva curvas de acurácia e loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Treino")
    axes[0].plot(history.history["val_accuracy"], label="Validação")
    axes[0].set_title("Acurácia por época")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Acurácia")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Treino")
    axes[1].plot(history.history["val_loss"], label="Validação")
    axes[1].set_title("Loss por época")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(config: Config, model_name: str = "bi_lstm") -> None:
    """
    Pipeline completo de treino:
      1. Carrega classes (respeita max_classes do config!) e dados
      2. Divide em treino/val/teste
      3. Aplica data augmentation no treino
      4. Constrói modelo
      5. Treina com callbacks
      6. Salva metadados, conjunto de teste e curvas
    """
    print("=" * 55)
    print("  TREINO — Reconhecimento de Libras")
    print("=" * 55)

    cfg_pre = config.preprocess
    cfg_train = config.training
    cfg_dataset = config.dataset

    # Pasta de keypoints processados: configs/default.yaml -> dataset.processed_dir
    # (custom, por padrão) ou PATHS["data_processed"] (saída do 02_preprocess.py
    # sobre o V-LIBRASIL) quando não especificado.
    processed_dir_cfg = cfg_dataset.get("processed_dir")
    processed_dir = (PROJECT_ROOT / processed_dir_cfg) if processed_dir_cfg else None

    # Calcula min_samples necessário para o split estratificado
    min_samples = _required_min_samples(
        test_size=cfg_train["test_size"],
        val_size=cfg_train["val_size"],
    )

    # 1. Carrega dados
    print("\n[1/5] Carregando dataset...")
    print(f"      Pasta:                         {processed_dir or PATHS['data_processed']}")
    print(f"      Mínimo de amostras por classe: {min_samples}")
    classes = list_classes(processed_dir=processed_dir, min_samples=min_samples)
    print(f"      Classes encontradas: {len(classes)}")

    # ✅ CORREÇÃO: aplica o filtro max_classes do config
    max_classes = cfg_dataset.get("max_classes")
    if max_classes is not None and max_classes < len(classes):
        print(f"      📌 Aplicando max_classes={max_classes} (de {len(classes)})")
        classes = classes[:max_classes]
        print(f"      Classes selecionadas: {len(classes)}")
        print(f"      Primeiras: {classes[:5]}...")
    else:
        print(f"      Usando todas as {len(classes)} classes")

    if len(classes) < 2:
        raise RuntimeError(
            f"Apenas {len(classes)} classes têm amostras suficientes.\n"
            "Reduza test_size/val_size no configs/default.yaml"
        )

    X, y = load_arrays(
        classes,
        processed_dir=processed_dir,
        expected_shape=(cfg_pre["num_frames"], cfg_pre["keypoint_dim"]),
    )
    print(f"      X shape: {X.shape}")

    # 2. Split
    print("\n[2/5] Dividindo dados...")
    split = split_data(
        X, y, classes,
        test_size=cfg_train["test_size"],
        val_size=cfg_train["val_size"],
        random_seed=cfg_train["random_seed"],
    )
    split.summary()

    # Salva conjunto de teste para evaluate.py (formato consumido por
    # libras.training.evaluator.evaluate: um único .npz com X_test/y_test)
    models_dir = PATHS["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(models_dir / "test_data.npz", X_test=split.X_test, y_test=split.y_test)

    # 3. Data augmentation (apenas no treino!)
    augmentation_cfg = cfg_train.get("augmentation", {})
    use_aug = augmentation_cfg.get("enabled", True)
    multiplier = augmentation_cfg.get("multiplier", 5)

    if use_aug:
        print(f"\n[3/5] Aplicando data augmentation (multiplicador {multiplier}x)...")
        X_train_aug, y_train_aug = augment_training_set(
            split.X_train, split.y_train,
            multiplier=multiplier,
            random_seed=cfg_train["random_seed"],
        )
        print(f"      Treino antes: {len(split.X_train)}")
        print(f"      Treino depois: {len(X_train_aug)}")
    else:
        print("\n[3/5] Data augmentation desativado.")
        X_train_aug, y_train_aug = split.X_train, split.y_train

    # 4. Modelo
    print("\n[4/5] Construindo modelo...")
    model = build_model(
        model_name,
        num_classes=split.num_classes,
        seq_len=cfg_pre["num_frames"],
        features=cfg_pre["keypoint_dim"],
        learning_rate=cfg_train["learning_rate"],
    )
    model.summary()

    # 5. Treino
    print(f"\n[5/5] Treinando (máx {cfg_train['epochs']} épocas)...")
    model_path = models_dir / "best_model.keras"
    callbacks_list = build_callbacks(
        model_path=model_path,
        logs_dir=PATHS["logs"],
        early_stopping_patience=cfg_train["callbacks"]["early_stopping_patience"],
        reduce_lr_patience=cfg_train["callbacks"]["reduce_lr_patience"],
        reduce_lr_factor=cfg_train["callbacks"]["reduce_lr_factor"],
        min_lr=cfg_train["callbacks"]["min_lr"],
    )

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(split.X_val, split.y_val),
        epochs=cfg_train["epochs"],
        batch_size=cfg_train["batch_size"],
        callbacks=callbacks_list,
    )

    # Salva artefatos
    save_json(
        {
            "classes": split.classes,
            "num_classes": split.num_classes,
            "num_frames": cfg_pre["num_frames"],
            "keypoint_dim": cfg_pre["keypoint_dim"],
            "threshold": config.inference["threshold"],
            "epochs_trained": len(history.history["accuracy"]),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "augmentation_used": use_aug,
            "augmentation_multiplier": multiplier if use_aug else 1,
            "max_classes_applied": max_classes,
        },
        models_dir / "metadata.json",
    )

    _plot_history(history, PATHS["reports"] / "training_curves.png")

    print("\n✅ Treino concluído!")
    print(f"   Melhor val_accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"   Modelo salvo:        {model_path}")