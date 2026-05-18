"""
Loader de dados processados.

Lê os arrays .npy de data/processed/ e prepara para o treino:
  • Filtra classes com amostras insuficientes
  • Faz split estratificado treino/validação/teste
  • Retorna container tipado DataSplit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from libras.config import PATHS


@dataclass
class DataSplit:
    """Container tipado para os splits de dados."""
    X_train: np.ndarray
    X_val:   np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_val:   np.ndarray
    y_test:  np.ndarray
    classes: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def summary(self) -> None:
        print(f"Classes:   {self.num_classes}")
        print(f"Treino:    {len(self.X_train)} amostras")
        print(f"Validação: {len(self.X_val)} amostras")
        print(f"Teste:     {len(self.X_test)} amostras")


def _required_min_samples(test_size: float, val_size: float) -> int:
    """
    Calcula o mínimo de amostras por classe necessário para o split.

    Precisa de pelo menos 1 amostra no treino e 1 para val+test.
    Com a alocação manual, 2 amostras já são suficientes
    (1 treino + 1 val, teste fica sem — mas aceitável).
    Para garantir representação nos 3 conjuntos, idealmente 3.
    """
    return 3


def list_classes(
    processed_dir: Path | None = None,
    min_samples: int = 3,
) -> list[str]:
    """
    Lista classes válidas (com ao menos `min_samples` amostras).

    Padrão: 3 amostras (1 treino + 1 val + 1 teste).
    Classes com menos amostras são descartadas porque o split
    precisa de pelo menos 1 amostra por conjunto.
    """
    processed_dir = processed_dir or PATHS["data_processed"]

    if not processed_dir.is_dir():
        raise FileNotFoundError(
            f"data/processed/ não encontrado em {processed_dir}\n"
            "Execute primeiro: python scripts/02_preprocess.py"
        )

    valid = []
    skipped = []
    for cls_dir in sorted(processed_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        n = len(list(cls_dir.glob("*.npy")))
        if n >= min_samples:
            valid.append(cls_dir.name)
        else:
            skipped.append((cls_dir.name, n))

    if skipped:
        print(f"⚠️  {len(skipped)} classes descartadas (< {min_samples} amostras)")
        if len(skipped) <= 10:
            for name, n in skipped:
                print(f"     {name}: {n} amostras")
        else:
            print(f"     Exemplos: {[s[0] for s in skipped[:5]]}...")

    return valid


def load_arrays(
    classes: list[str],
    processed_dir: Path | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega todos os .npy das classes fornecidas.

    Returns
    -------
    X : array (N, num_frames, keypoint_dim) float32
    y : array one-hot (N, num_classes)
    """
    processed_dir = processed_dir or PATHS["data_processed"]
    label_map = {cls: idx for idx, cls in enumerate(classes)}

    X_list, y_list = [], []
    for cls in tqdm(classes, desc="Carregando dados"):
        for npy_path in (processed_dir / cls).glob("*.npy"):
            arr = np.load(npy_path)
            if expected_shape and arr.shape != expected_shape:
                print(f"⚠️  Shape inválido ignorado: {npy_path.name} → {arr.shape}")
                continue
            X_list.append(arr)
            y_list.append(label_map[cls])

    X = np.array(X_list, dtype=np.float32)
    y = to_categorical(y_list, num_classes=len(classes))
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> DataSplit:
    """
    Split treino / validação / teste.

    Usa alocação manual por classe para garantir que cada classe
    aparece em todos os 3 conjuntos, mesmo com poucas amostras
    (≥3 por classe). Isso evita o erro do sklearn quando
    num_classes > tamanho de algum split.

    Para classes com amostras suficientes, respeita as proporções
    test_size/val_size. Para classes com poucas amostras, distribui
    1 amostra por conjunto (treino recebe o excedente).
    """
    rng = np.random.RandomState(random_seed)
    y_int = np.argmax(y, axis=1)

    train_idx, val_idx, test_idx = [], [], []

    for cls_id in range(len(classes)):
        idxs = np.where(y_int == cls_id)[0]
        rng.shuffle(idxs)
        n = len(idxs)

        if n == 0:
            continue

        # Número de amostras para test e val (pelo menos 1 cada)
        n_test = max(1, round(n * test_size))
        n_val = max(1, round(n * val_size))

        # Garantir que sobra pelo menos 1 para treino
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
            # Se ainda não sobra para treino (n < 3), ajustar
            if n_test + n_val >= n:
                n_val = 0 if n < 3 else 1
                n_test = 0 if n < 2 else 1

        n_train = n - n_test - n_val

        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    # Embaralha dentro de cada split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return DataSplit(
        X_train=X[train_idx], X_val=X[val_idx], X_test=X[test_idx],
        y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
        classes=classes,
    )