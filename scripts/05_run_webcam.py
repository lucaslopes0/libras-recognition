"""
Passo 5 — Inferência em tempo real com webcam.

⚠️  Execução local apenas — não funciona em Colab/Kaggle.

Uso:
    python scripts/05_run_webcam.py
    python scripts/05_run_webcam.py --no-feedback
    python scripts/05_run_webcam.py --camera 1
"""

from __future__ import annotations

import argparse

from libras.config import load_config
from libras.inference.webcam import run_webcam


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência em tempo real")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-feedback", action="store_true",
                        help="Desativa o sistema de feedback")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_webcam(cfg, camera_index=args.camera, use_feedback=not args.no_feedback)


if __name__ == "__main__":
    main()
