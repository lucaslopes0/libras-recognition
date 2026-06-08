"""
Gravação guiada de sinais em Libras via webcam.

Mostra qual palavra gravar, conta regressiva, e salva automaticamente
na estrutura correta de pastas.

Uso:
    python scripts/06_record_dataset.py
    python scripts/06_record_dataset.py --camera 1         # webcam externa
    python scripts/06_record_dataset.py --person lucas      # nome do sinalizador
    python scripts/06_record_dataset.py --duration 4        # segundos por vídeo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ─── Vocabulário ───
VOCABULARY = [
    "Casa",
    "Desculpa",
    "Trabalho",
    "De nada",
    "Amigo",
    "Família",
]

REPS_PER_WORD = 10  # repetições por sessão


def draw_text_centered(img, text, y, scale=1.0, color=(255, 255, 255), thickness=2):
    """Desenha texto centralizado horizontalmente."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(img, text_lines, alpha=0.6):
    """Desenha fundo escuro com texto centralizado."""
    overlay = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    y_start = h // 2 - len(text_lines) * 30
    for i, (text, scale, color) in enumerate(text_lines):
        draw_text_centered(img, text, y_start + i * 60, scale, color)


def countdown(cap, seconds=3):
    """Mostra contagem regressiva."""
    for i in range(seconds, 0, -1):
        start = time.time()
        while time.time() - start < 1.0:
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            draw_overlay(frame, [
                (str(i), 3.0, (0, 255, 255)),
                ("Prepare-se...", 0.8, (200, 200, 200)),
            ], alpha=0.4)
            cv2.imshow("Gravacao Libras", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
    return True


def record_clip(cap, duration_sec, output_path, fps=30):
    """Grava um clipe de duração fixa."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    n_frames = int(duration_sec * fps)
    recorded = 0
    start = time.time()

    while recorded < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        frame = cv2.flip(frame, 1)
        writer.write(original)  # salva sem espelhar

        # HUD de gravação
        elapsed = time.time() - start
        remaining = max(0, duration_sec - elapsed)
        progress = min(1.0, elapsed / duration_sec)

        # Bola vermelha pulsante (gravando)
        radius = 15 + int(5 * np.sin(elapsed * 6))
        cv2.circle(frame, (40, 40), radius, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (65, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Timer
        cv2.putText(frame, f"{remaining:.1f}s",
                    (w - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Barra de progresso
        bar_w = int(progress * w)
        cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 255, 0), -1)

        cv2.imshow("Gravacao Libras", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            writer.release()
            return False

        recorded += 1

    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Gravação guiada de sinais")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--person", type=str, default="user")
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Duração de cada clipe em segundos")
    parser.add_argument("--output", type=str, default="data/raw_custom",
                        help="Pasta de saída")
    parser.add_argument("--start-word", type=int, default=0,
                        help="Índice da palavra inicial (0-19)")
    parser.add_argument("--start-rep", type=int, default=1,
                        help="Número da repetição inicial")
    args = parser.parse_args()

    output_base = Path(args.output)
    person = args.person

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Câmera {args.camera} indisponível.")
        return

    # Resolução
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Câmera: {w}×{h}")
    print(f"👤 Pessoa: {person}")
    print(f"⏱️  Duração: {args.duration}s por clipe")
    print(f"📂 Saída:  {output_base}")
    print(f"\nPressione Q a qualquer momento para sair.")
    print(f"Pressione S para pular uma palavra.")
    print(f"Pressione R para regravar a última.\n")

    total_recorded = 0

    for word_idx in range(args.start_word, len(VOCABULARY)):
        word = VOCABULARY[word_idx]
        word_dir = output_base / word
        word_dir.mkdir(parents=True, exist_ok=True)

        # Conta quantos já existem dessa pessoa
        existing = list(word_dir.glob(f"{word}_{person}_*.mp4"))
        start_rep = len(existing) + 1

        for rep in range(start_rep, start_rep + REPS_PER_WORD):
            filename = f"{word}_{person}_{rep:02d}.mp4"
            filepath = word_dir / filename

            # Tela de instrução
            key = None
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                draw_overlay(frame, [
                    (f"Palavra: {word}", 1.5, (0, 255, 100)),
                    (f"Repeticao {rep}/{start_rep + REPS_PER_WORD - 1}", 0.7, (200, 200, 200)),
                    (f"({word_idx + 1}/{len(VOCABULARY)})", 0.6, (150, 150, 150)),
                    ("", 0.5, (0, 0, 0)),
                    ("ESPACO = gravar | S = pular | Q = sair", 0.6, (0, 200, 255)),
                ], alpha=0.5)

                cv2.imshow("Gravacao Libras", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"\n✅ Encerrado. Total gravado: {total_recorded} vídeos.")
                    return
                elif key == ord("s"):
                    print(f"   ⏭️  Pulando {word}")
                    waiting = False
                    break
                elif key == ord(" "):
                    waiting = False

            if key == ord("s"):
                break  # pula para próxima palavra

            # Contagem regressiva
            if not countdown(cap, seconds=3):
                cap.release()
                cv2.destroyAllWindows()
                return

            # Grava
            print(f"   🎬 Gravando: {filename}...", end=" ")
            ok = record_clip(cap, args.duration, filepath)
            if not ok:
                cap.release()
                cv2.destroyAllWindows()
                return

            total_recorded += 1
            print(f"✅ ({total_recorded} total)")

            # Pausa entre repetições
            pause_start = time.time()
            while time.time() - pause_start < 1.0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                draw_text_centered(frame, "OK! Preparando proxima...",
                                   frame.shape[0] // 2, 0.8, (0, 255, 0))
                cv2.imshow("Gravacao Libras", frame)
                cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 Gravação completa! Total: {total_recorded} vídeos em {output_base}")


if __name__ == "__main__":
    main()