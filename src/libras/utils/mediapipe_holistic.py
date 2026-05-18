"""
Wrapper sobre o MediaPipe Holistic.

Centraliza:
  • Extração de keypoints (1662 dimensões)
  • Normalização temporal de sequências
  • Desenho de landmarks
  • Parsing de nomes de arquivo do V-LIBRASIL

Compatível com mediapipe==0.10.9
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

# Handles globais do MediaPipe
_mp_holistic = mp.solutions.holistic
_mp_drawing = mp.solutions.drawing_utils

# Dimensões esperadas
POSE_DIM = 33 * 4       # x, y, z, visibility
FACE_DIM = 468 * 3
HAND_DIM = 21 * 3
TOTAL_DIM = POSE_DIM + FACE_DIM + HAND_DIM + HAND_DIM  # 1662


# ── Detecção ────────────────────────────────────────────────

def detect(image: np.ndarray, model: Any) -> tuple[np.ndarray, Any]:
    """Processa um frame BGR com o modelo MediaPipe Holistic."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def create_holistic(
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
    model_complexity: int = 1,
):
    """Cria instância do Holistic. Use como context manager."""
    return _mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )


# ── Extração de keypoints ────────────────────────────────────

def extract_keypoints(results: Any) -> np.ndarray:
    """
    Extrai vetor de 1662 keypoints de um resultado do MediaPipe.

    Composição:
      pose        33 × 4  = 132
      face       468 × 3  = 1404
      left_hand   21 × 3  = 63
      right_hand  21 × 3  = 63
    """
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility]
                  for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(POSE_DIM)
    )
    face = (
        np.array([[r.x, r.y, r.z]
                  for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(FACE_DIM)
    )
    lh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(HAND_DIM)
    )
    rh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(HAND_DIM)
    )
    return np.concatenate([pose, face, lh, rh])


# Índices das regiões no vetor de 1662 dimensões
REGION_SLICES = {
    "pose":       (0,                              POSE_DIM),
    "face":       (POSE_DIM,                       POSE_DIM + FACE_DIM),
    "left_hand":  (POSE_DIM + FACE_DIM,            POSE_DIM + FACE_DIM + HAND_DIM),
    "right_hand": (POSE_DIM + FACE_DIM + HAND_DIM, TOTAL_DIM),
}


# ── Normalização temporal ────────────────────────────────────

def normalize_sequence(frames: np.ndarray, target: int) -> np.ndarray:
    """
    Garante exatamente `target` frames por sequência.
    Mais frames: amostragem uniforme. Menos: padding com último frame.
    """
    if len(frames) == 0:
        return np.zeros((target, TOTAL_DIM))
    if len(frames) >= target:
        idx = np.linspace(0, len(frames) - 1, target, dtype=int)
        return frames[idx]
    pad = np.repeat(frames[-1:], target - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)


# ── Dataset V-LIBRASIL ────────────────────────────────────────

def parse_class_name(filename: str | Path) -> str:
    """
    Extrai a classe do padrão V-LIBRASIL.

    >>> parse_class_name("Abacaxi_Articulador1.mp4")
    'Abacaxi'
    >>> parse_class_name("À noite toda_Articulador2.mp4")
    'À noite toda'
    """
    name = Path(filename).stem
    parts = name.rsplit("_Articulador", 1)
    return parts[0].strip() if len(parts) == 2 else name


# ── Visualização ────────────────────────────────────────────

def draw_landmarks(image: np.ndarray, results: Any) -> None:
    """Desenha todos os landmarks no frame com cores por região."""
    if results.face_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.face_landmarks,
            _mp_holistic.FACEMESH_CONTOURS,
            _mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            _mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.pose_landmarks, _mp_holistic.POSE_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            _mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, _mp_holistic.HAND_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            _mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        _mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, _mp_holistic.HAND_CONNECTIONS,
            _mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            _mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )
