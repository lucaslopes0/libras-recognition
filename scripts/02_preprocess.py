"""
Passo 2 — Extrai keypoints dos vídeos com MediaPipe.

⏱️  Pode levar 1–3 horas no dataset completo. É retomável.

Uso:
    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --test          # só 30 vídeos
    python scripts/02_preprocess.py --limit 100     # só os primeiros 100
"""

from __future__ import annotations

import argparse

from libras.config import ensure_directories, load_config
from libras.data.preprocess import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-processa o dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test", action="store_true", help="Processa apenas 30 vídeos")
    parser.add_argument("--limit", type=int, default=None, help="Limita N vídeos")
    args = parser.parse_args()

    ensure_directories()
    cfg = load_config(args.config)

    limit = 30 if args.test else args.limit

    preprocess_dataset(
        num_frames=cfg.preprocess["num_frames"],
        max_classes=cfg.dataset["max_classes"],
        mp_config=cfg.preprocess["mediapipe"],
        limit=limit,
    )


if __name__ == "__main__":
    main()
