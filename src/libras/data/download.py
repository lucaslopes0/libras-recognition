"""
Download do dataset V-LIBRASIL via kagglehub.

Pré-requisito: credenciais em ~/.kaggle/kaggle.json
Obtenha em kaggle.com → Settings → API → Create New Token
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from libras.config import PATHS


def _find_video_folder(base: Path) -> Path | None:
    """Percorre subpastas até encontrar uma com .mp4."""
    for path in base.rglob("*.mp4"):
        return path.parent
    return None


def download_dataset(kaggle_id: str, destination: Path | None = None) -> Path:
    """
    Baixa o dataset do Kaggle e copia para `destination` (ou `data/raw/`).
    """
    destination = destination or PATHS["data_raw"]

    print(f"📥 Baixando dataset: {kaggle_id}")
    kaggle_path = Path(kagglehub.dataset_download(kaggle_id))
    print(f"✓  Download concluído em: {kaggle_path}")

    video_folder = _find_video_folder(kaggle_path)
    if video_folder is None:
        raise RuntimeError("Nenhum .mp4 encontrado no dataset baixado.")

    n_videos = len(list(video_folder.glob("*.mp4")))
    print(f"✓  {n_videos} vídeos em: {video_folder}")

    print(f"📂 Copiando para {destination} ...")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(video_folder, destination, dirs_exist_ok=True)

    final = len(list(destination.glob("*.mp4")))
    print(f"✅ {final} vídeos disponíveis em {destination}")

    return destination
