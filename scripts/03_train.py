"""
Passo 3 — Treina o modelo.

Uso:
    python scripts/03_train.py
    python scripts/03_train.py --model bi_lstm
    python scripts/03_train.py --config configs/small.yaml
"""

from __future__ import annotations

import argparse

from libras.config import ensure_directories, load_config
from libras.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="bi_lstm",
                        help="Arquitetura registrada em libras.models")
    args = parser.parse_args()

    ensure_directories()
    cfg = load_config(args.config)
    train(cfg, model_name=args.model)


if __name__ == "__main__":
    main()
