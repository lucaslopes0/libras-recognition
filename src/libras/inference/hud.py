"""
HUD (Heads-Up Display) — desenho da interface da webcam.

VERSÃO MELHORADA PARA DEMO:
  • Destaca as TOP-3 predições (não só a top-1)
  • Mostra a top-1 grande no topo
  • Top-2 e Top-3 menores, indicando alternativas
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_top_predictions(
    image: np.ndarray,
    res: np.ndarray,
    classes: list[str],
    threshold: float = 0.3,
) -> str | None:
    """
    Mostra as TOP-3 predições em destaque.
    Retorna a classe top-1 se passar do threshold, senão None.
    """
    top_idx = np.argsort(res)[::-1][:3]
    top_probs = [float(res[i]) for i in top_idx]
    top_names = [classes[i] for i in top_idx]

    # ── Faixa preta de fundo ──
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 130), (30, 30, 30), -1)

    # ── Top-1 GRANDE ──
    is_confident = top_probs[0] >= threshold
    color = (0, 255, 0) if is_confident else (100, 100, 255)
    cv2.putText(
        image, f"#1: {top_names[0]}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, f"{top_probs[0]*100:.1f}%",
        (w - 130, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
    )

    # ── Top-2 e Top-3 menores ──
    cv2.putText(
        image,
        f"#2: {top_names[1]} ({top_probs[1]*100:.0f}%)",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"#3: {top_names[2]} ({top_probs[2]*100:.0f}%)",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA,
    )

    return top_names[0] if is_confident else None


def draw_history(image: np.ndarray, sentence: list[str]) -> None:
    """Faixa lateral direita com histórico de palavras reconhecidas."""
    if not sentence:
        return

    h, w = image.shape[:2]
    # Faixa à direita
    box_x = w - 200
    cv2.rectangle(image, (box_x, 140), (w, 140 + len(sentence) * 35 + 20),
                  (50, 50, 50), -1)
    cv2.putText(
        image, "Historico:",
        (box_x + 5, 165),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    for i, word in enumerate(sentence[-5:]):
        cv2.putText(
            image, f"- {word}",
            (box_x + 5, 195 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA,
        )


def draw_feedback(
    image: np.ndarray,
    msgs: list[str],
    score: float | None = None,
) -> None:
    """Mensagens de feedback + score na parte inferior."""
    h, w = image.shape[:2]
    base_y = h - 30 - len(msgs) * 22

    if score is not None:
        score_color = (
            (0, 255, 0) if score >= 0.8
            else (0, 200, 255) if score >= 0.5
            else (0, 100, 255)
        )
        cv2.putText(
            image, f"Score: {score:.2f}", (10, base_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA,
        )

    for i, msg in enumerate(msgs[:4]):
        color = (0, 220, 100) if "✅" in msg else (0, 200, 255)
        cv2.putText(
            image, msg[:60], (10, base_y + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )


def draw_buffer_bar(image: np.ndarray, progress: float) -> None:
    """Barra de progresso do buffer (0.0–1.0)."""
    h, w = image.shape[:2]
    filled = int(progress * w)
    cv2.rectangle(image, (0, h - 8), (w, h), (60, 60, 60), -1)
    cv2.rectangle(image, (0, h - 8), (filled, h), (0, 220, 100), -1)


def draw_controls(image: np.ndarray) -> None:
    """Legenda dos controles."""
    controls = ["Q: sair", "C: limpar historico"]
    h, w = image.shape[:2]
    for i, txt in enumerate(controls):
        cv2.putText(
            image, txt, (10, h - 60 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )