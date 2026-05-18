"""
Passo 1 — Download do dataset V-LIBRASIL.

Uso:
    python scripts/01_download.py
    python scripts/01_download.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from libras.config import ensure_directories, load_config
from libras.data.download import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa o dataset V-LIBRASIL")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    ensure_directories()
    cfg = load_config(args.config)
    download_dataset(cfg.dataset["kaggle_id"])


if __name__ == "__main__":
    main()
