"""
Sistema de feedback por keypoints.

Compara a execução do usuário com a referência média do dataset
e gera sugestões específicas por região (mãos, postura, expressão).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from libras.config import PATHS
from libras.utils.mediapipe_holistic import REGION_SLICES

REGION_LABELS = {
    "right_hand": "🤚 Mão direita",
    "left_hand":  "🤚 Mão esquerda",
    "pose":       "🧍 Postura corporal",
    "face":       "😐 Expressão facial",
}


class FeedbackEngine:
    """
    Mantém cache de referências e gera feedback em tempo real.

    Uso típico
    ----------
    >>> engine = FeedbackEngine(classes=["Abacaxi", "Abraço", ...])
    >>> msgs  = engine.feedback(user_seq, "Abacaxi")
    >>> score = engine.score(user_seq, "Abacaxi")
    """

    def __init__(
        self,
        classes: list[str],
        processed_dir: Path | None = None,
        error_threshold: float = 0.05,
    ) -> None:
        self.processed_dir = processed_dir or PATHS["data_processed"]
        self.error_threshold = error_threshold
        self.references = self._preload_references(classes)

    # ── Construção das referências ────────────────────────────

    def _build_reference(self, class_name: str) -> np.ndarray | None:
        """Calcula keypoint médio de uma classe."""
        cls_dir = self.processed_dir / class_name
        if not cls_dir.is_dir():
            return None
        arrays = [np.load(p) for p in cls_dir.glob("*.npy")]
        return np.mean(arrays, axis=0) if arrays else None

    def _preload_references(self, classes: list[str]) -> dict[str, np.ndarray | None]:
        return {
            cls: self._build_reference(cls)
            for cls in tqdm(classes, desc="Carregando referências")
        }

    # ── API pública ────────────────────────────────────────────

    def feedback(self, user_seq: np.ndarray, class_name: str) -> list[str]:
        """Gera lista de sugestões para o sinal `class_name`."""
        ref = self.references.get(class_name)
        if ref is None:
            return [f"⚠️  Referência não disponível para '{class_name}'"]

        if user_seq.shape != ref.shape:
            return [f"⚠️  Shape incompatível: {user_seq.shape} vs {ref.shape}"]

        diff = np.abs(user_seq - ref).mean(axis=0)

        msgs = []
        for region, (start, end) in REGION_SLICES.items():
            err = diff[start:end].mean()
            if err > self.error_threshold:
                label = REGION_LABELS[region]
                msgs.append(f"{label}: ajuste necessário (desvio: {err:.3f})")

        if not msgs:
            msgs.append(f'✅ Sinal "{class_name}" executado corretamente!')

        return msgs

    def score(self, user_seq: np.ndarray, class_name: str) -> float | None:
        """Pontuação 0.0–1.0 da execução (1.0 = perfeita)."""
        ref = self.references.get(class_name)
        if ref is None or user_seq.shape != ref.shape:
            return None
        diff = np.abs(user_seq - ref).mean()
        return round(max(0.0, 1.0 - diff * 10), 3)
