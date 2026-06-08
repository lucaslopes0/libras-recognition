"""
Wrapper de predição.

Encapsula modelo + metadados + buffer de frames, oferecendo
interface simples para inferência em streaming.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow.keras.models import load_model

from libras.config import PATHS
from libras.utils.io import load_json


class StreamPredictor:
    """
    Mantém buffer deslizante de frames e prediz quando cheio.

    Uso típico
    ----------
    >>> p = StreamPredictor()
    >>> while webcam:
    ...     p.push(extract_keypoints(...))
    ...     res = p.predict()  # None se buffer ainda não está cheio
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        predict_interval: int = 5,
        activity_threshold: float = 0.01,
    ) -> None:
        models_dir = models_dir or PATHS["models"]
        model_path = models_dir / "best_model.keras"
        meta_path  = models_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}\n"
                "Execute primeiro: python scripts/03_train.py"
            )

        self.meta       = load_json(meta_path)
        self.classes    = self.meta["classes"]
        self.num_frames = self.meta["num_frames"]
        self.threshold  = self.meta["threshold"]

        self.model = load_model(model_path)
        self.buffer: deque = deque(maxlen=self.num_frames)

        self._predict_interval = max(1, predict_interval)
        self._activity_threshold = activity_threshold
        self._frame_count = 0
        self._last_result: Optional[np.ndarray] = None

    # ── Buffer ──────────────────────────────────────────────

    def push(self, keypoints: np.ndarray) -> None:
        self.buffer.append(keypoints)

    def clear(self) -> None:
        self.buffer.clear()
        self._frame_count = 0
        self._last_result = None

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) == self.num_frames

    @property
    def buffer_progress(self) -> float:
        return len(self.buffer) / self.num_frames

    @property
    def is_active(self) -> bool:
        """True se há movimento suficiente nas mãos para iniciar predição."""
        return self.is_ready and self._is_active()

    # ── Predição ────────────────────────────────────────────

    def _is_active(self) -> bool:
        # Avalia apenas os keypoints das mãos (últimas 126 dims: mão esq + dir).
        # Mede o desvio padrão temporal — baixo std indica mãos paradas.
        hands = np.array(self.buffer)[:, -126:]
        return float(np.std(hands)) > self._activity_threshold

    def predict(self) -> Optional[np.ndarray]:
        """Retorna probabilidades, ou None se buffer não está cheio ou mãos inativas."""
        if not self.is_ready:
            return None
        if not self._is_active():
            self._last_result = None
            return None
        if self._frame_count % self._predict_interval == 0:
            inp = np.expand_dims(np.array(self.buffer), axis=0)
            self._last_result = self.model.predict(inp, verbose=0)[0]
        self._frame_count += 1
        return self._last_result

    def get_buffer_array(self) -> np.ndarray:
        """Retorna o buffer atual como array (útil para feedback)."""
        return np.array(self.buffer)
