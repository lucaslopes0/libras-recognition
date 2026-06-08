"""
Passo 7 — Pré-processa o dataset customizado gravado com 06_record_dataset.py.

Lê videos de data/raw_custom/{Classe}/{arquivo}.mp4,
extrai keypoints com MediaPipe e salva em data/processed_custom/{Classe}/{arquivo}.npy.

É retomável: pula arquivos já processados.

Uso:
    python scripts/07_preprocess_custom.py
    python scripts/07_preprocess_custom.py --person lucas   # filtra por sinalizador
"""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from libras.config import PATHS, load_config
from libras.data.preprocess import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-processa dataset customizado")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--person", type=str, default=None,
                        help="Processa apenas videos deste sinalizador (ex: lucas)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_frames = cfg.preprocess["num_frames"]
    mp_config = cfg.preprocess["mediapipe"]

    raw_dir = PATHS["data_raw_custom"]
    out_dir = PATHS["data_processed_custom"]

    if not raw_dir.is_dir():
        print(f"Nenhum video encontrado em {raw_dir}")
        print("Execute primeiro: python scripts/06_record_dataset.py")
        return

    videos = sorted(raw_dir.rglob("*.mp4"))
    if args.person:
        videos = [v for v in videos if f"_{args.person}_" in v.stem]

    if not videos:
        print("Nenhum video encontrado com os filtros aplicados.")
        return

    print(f"Videos encontrados: {len(videos)}")

    saved = skipped = errors = 0

    for video_path in tqdm(videos, desc="Extraindo keypoints"):
        class_name = video_path.parent.name
        save_path = out_dir / class_name / f"{video_path.stem}.npy"

        if save_path.exists():
            skipped += 1
            continue

        keypoints = process_video(video_path, num_frames, mp_config)
        if keypoints is None:
            print(f"  Erro ao processar: {video_path.name}")
            errors += 1
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, keypoints)
        saved += 1

    print(f"\nConcluido!")
    print(f"  Salvos:  {saved}")
    print(f"  Pulados: {skipped} (ja existiam)")
    print(f"  Erros:   {errors}")
    if saved > 0:
        print(f"\nKeypoints salvos em: {out_dir}")
        print("Para incluir no treino, adicione data/processed_custom/ ao loader.")


if __name__ == "__main__":
    main()
