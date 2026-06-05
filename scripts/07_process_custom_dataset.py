"""
Processa vídeos gravados (celular ou webcam) → keypoints .npy

Lê a estrutura de pastas:
    data/raw_custom/
    ├── Ola/
    │   ├── Ola_lucas_01.mp4
    │   └── ...
    └── ...

Gera:
    data/processed_custom/
    ├── Ola/
    │   ├── Ola_lucas_01.npy    (shape: 30, 1662)
    │   └── ...
    └── ...

Uso:
    python scripts/07_process_custom_dataset.py
    python scripts/07_process_custom_dataset.py --input data/raw_custom --output data/processed_custom
    python scripts/07_process_custom_dataset.py --num-frames 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


def extract_keypoints_from_results(results) -> np.ndarray:
    """Extrai 1662 keypoints de um frame (mesmo formato do pipeline original)."""
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)

    face = np.array([[r.x, r.y, r.z]
                     for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[r.x, r.y, r.z]
                   for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[r.x, r.y, r.z]
                   for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])  # 132 + 1404 + 63 + 63 = 1662


def process_video(
    video_path: Path,
    num_frames: int = 30,
    holistic=None,
) -> np.ndarray | None:
    """
    Processa um vídeo e retorna array de keypoints (num_frames, 1662).

    Amostra `num_frames` frames uniformemente distribuídos ao longo do vídeo.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        return None

    # Seleciona frames uniformemente espaçados
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    keypoints_sequence = []
    frame_idx = 0
    target_idx = 0

    while cap.isOpened() and target_idx < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == frame_indices[target_idx]:
            # Converte BGR → RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            kp = extract_keypoints_from_results(results)
            keypoints_sequence.append(kp)
            target_idx += 1

        frame_idx += 1

    cap.release()

    if len(keypoints_sequence) < num_frames:
        # Preenche com último frame se necessário
        while len(keypoints_sequence) < num_frames:
            keypoints_sequence.append(keypoints_sequence[-1]
                                      if keypoints_sequence
                                      else np.zeros(1662))

    return np.array(keypoints_sequence[:num_frames], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Processa vídeos de Libras → keypoints .npy"
    )
    parser.add_argument("--input", type=str, default="data/raw_custom",
                        help="Pasta com vídeos organizados por classe")
    parser.add_argument("--output", type=str, default="data/processed_custom",
                        help="Pasta de saída para keypoints")
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Número de frames por vídeo (padrão: 30)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Pasta não encontrada: {input_dir}")
        return

    # Lista classes (subpastas)
    classes = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    print(f"📂 Classes encontradas: {len(classes)}")
    for c in classes:
        videos = list((input_dir / c).glob("*.mp4"))
        print(f"   {c}: {len(videos)} vídeos")

    # Processa com MediaPipe Holistic
    print(f"\n🤖 Processando com MediaPipe Holistic...")
    print(f"   Frames por vídeo: {args.num_frames}")
    print(f"   Saída: {output_dir}\n")

    mp_holistic = mp.solutions.holistic

    total_ok = 0
    total_err = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for cls in classes:
            cls_input = input_dir / cls
            cls_output = output_dir / cls
            cls_output.mkdir(parents=True, exist_ok=True)

            videos = sorted(cls_input.glob("*.mp4"))
            print(f"\n📹 {cls} ({len(videos)} vídeos):")

            for video_path in tqdm(videos, desc=f"   {cls}"):
                output_path = cls_output / f"{video_path.stem}.npy"

                # Pula se já processado
                if output_path.exists():
                    total_ok += 1
                    continue

                result = process_video(video_path, args.num_frames, holistic)

                if result is not None and result.shape == (args.num_frames, 1662):
                    np.save(output_path, result)
                    total_ok += 1
                else:
                    print(f"   ⚠️  Erro ao processar: {video_path.name}")
                    total_err += 1

    # Resumo
    print(f"\n{'=' * 55}")
    print(f"  PROCESSAMENTO CONCLUÍDO")
    print(f"{'=' * 55}")
    print(f"   ✅ Processados: {total_ok}")
    print(f"   ❌ Erros:       {total_err}")
    print(f"   📂 Saída:       {output_dir}")

    # Estatísticas por classe
    print(f"\n📊 Resumo por classe:")
    for cls in classes:
        cls_output = output_dir / cls
        n = len(list(cls_output.glob("*.npy")))
        print(f"   {cls}: {n} amostras")

    print(f"\n📋 Próximo passo:")
    print(f"   Atualize configs/default.yaml para apontar para {output_dir}")
    print(f"   E rode: python scripts/03_train.py")


if __name__ == "__main__":
    main()