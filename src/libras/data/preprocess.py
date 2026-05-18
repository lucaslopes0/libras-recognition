"""
Pré-processamento: vídeos .mp4 → keypoints .npy.

Lê todos os vídeos do dataset, extrai keypoints com MediaPipe e
salva como arrays NumPy organizados em pastas por classe.

É retomável: pula arquivos já processados.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from libras.config import PATHS
from libras.utils.mediapipe_holistic import (
    create_holistic, detect, extract_keypoints,
    normalize_sequence, parse_class_name, TOTAL_DIM,
)


def process_video(video_path: Path, num_frames: int, mp_config: dict) -> np.ndarray | None:
    """
    Extrai keypoints de um vídeo e normaliza para `num_frames`.

    Returns
    -------
    Array (num_frames, TOTAL_DIM) ou None se o vídeo não pôde ser lido.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames_kp = []
    with create_holistic(
        detection_confidence=mp_config["detection_confidence"],
        tracking_confidence=mp_config["tracking_confidence"],
        model_complexity=mp_config["model_complexity"],
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, results = detect(frame, holistic)
            frames_kp.append(extract_keypoints(results))

    cap.release()

    if not frames_kp:
        return None

    seq = normalize_sequence(np.array(frames_kp), target=num_frames)
    assert seq.shape == (num_frames, TOTAL_DIM), f"Shape inesperado: {seq.shape}"
    return seq


def preprocess_dataset(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    num_frames: int = 30,
    max_classes: int | None = None,
    mp_config: dict | None = None,
    limit: int | None = None,
) -> dict:
    """
    Processa todo o dataset.

    Parameters
    ----------
    raw_dir     : pasta com os vídeos .mp4. Padrão: data/raw/
    output_dir  : pasta para os .npy.       Padrão: data/processed/
    num_frames  : frames por sequência
    max_classes : limita classes (None = todas)
    mp_config   : config do MediaPipe (dict)
    limit       : limita total de vídeos a processar (útil para teste)
    """
    raw_dir = raw_dir or PATHS["data_raw"]
    output_dir = output_dir or PATHS["data_processed"]
    mp_config = mp_config or {
        "detection_confidence": 0.5,
        "tracking_confidence": 0.5,
        "model_complexity": 1,
    }

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Diretório de vídeos não encontrado: {raw_dir}\n"
            "Execute primeiro: python scripts/01_download.py"
        )

    all_videos = sorted(raw_dir.glob("*.mp4"))
    if not all_videos:
        raise RuntimeError(f"Nenhum .mp4 em {raw_dir}")

    # Filtra classes
    all_classes = sorted({parse_class_name(v.name) for v in all_videos})
    if max_classes:
        all_classes = all_classes[:max_classes]
    selected_classes = set(all_classes)

    videos = [v for v in all_videos if parse_class_name(v.name) in selected_classes]
    if limit:
        videos = videos[:limit]
        print(f"[TEST] Modo limitado: {len(videos)} videos")

    print(f"[INFO] Classes: {len(all_classes)} | Videos: {len(videos)}")

    stats = {"saved": 0, "skipped": 0, "errors": []}
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in tqdm(videos, desc="Extraindo keypoints"):
        class_name = parse_class_name(video_path.name)
        save_path = output_dir / class_name / f"{video_path.stem}.npy"

        if save_path.exists():
            stats["skipped"] += 1
            continue

        keypoints = process_video(video_path, num_frames, mp_config)
        if keypoints is None:
            stats["errors"].append(video_path.name)
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, keypoints)
        stats["saved"] += 1

    stats["classes"] = len(all_classes)

    print(
        f"\n[OK] Concluido!\n"
        f"   Salvos:  {stats['saved']}\n"
        f"   Pulados: {stats['skipped']} (ja existiam)\n"
        f"   Erros:   {len(stats['errors'])}"
    )
    if stats["errors"]:
        print(f"   Exemplos: {stats['errors'][:5]}")

    return stats
