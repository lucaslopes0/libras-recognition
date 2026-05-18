"""
Loop principal da webcam — VERSÃO DEMO.

Mostra TOP-3 predições em destaque (compensa baixa acurácia top-1).
"""

from __future__ import annotations

import cv2
import numpy as np

from libras.config import Config
from libras.inference.feedback import FeedbackEngine
from libras.inference.hud import (
    draw_buffer_bar, draw_controls, draw_feedback,
    draw_history, draw_top_predictions,
)
from libras.inference.predictor import StreamPredictor
from libras.utils.mediapipe_holistic import (
    create_holistic, detect, draw_landmarks, extract_keypoints,
)


def run_webcam(
    config: Config,
    camera_index: int = 0,
    use_feedback: bool = True,
) -> None:
    """Loop principal — encerra com Q, limpa histórico com C."""
    # Inicializa componentes
    predictor = StreamPredictor()
    feedback_engine = (
        FeedbackEngine(
            classes=predictor.classes,
            error_threshold=config.feedback["error_threshold"],
        )
        if use_feedback else None
    )

    stability_window = config.inference["stability_window"]
    threshold = 0.30   # ← MAIS BAIXO para mostrar mais predições
    mp_config = config.preprocess["mediapipe"]

    # Estado
    sentence:      list[str] = []
    predictions:   list[int] = []
    feedback_msgs: list[str] = []
    exec_score: float | None = None
    last_class: str | None = None

    # Câmera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Câmera {camera_index} indisponível.")
    print("✅ Câmera aberta.")
    print("   Pressione Q para sair, C para limpar histórico.")

    with create_holistic(
        detection_confidence=mp_config["detection_confidence"],
        tracking_confidence=mp_config["tracking_confidence"],
        model_complexity=mp_config["model_complexity"],
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Espelha (mais natural para a pessoa)
            frame = cv2.flip(frame, 1)

            # MediaPipe
            image, results = detect(frame, holistic)
            draw_landmarks(image, results)

            # Push keypoints no buffer
            predictor.push(extract_keypoints(results))

            # Predição (se buffer cheio)
            res = predictor.predict()
            if res is not None:
                predictions.append(int(np.argmax(res)))

                # Desenha TOP-3 em destaque
                detected = draw_top_predictions(
                    image, res, predictor.classes, threshold=threshold,
                )

                # Estabilização: aceita só se últimos N frames concordam
                if len(predictions) >= stability_window and detected:
                    most_common = int(
                        np.bincount(predictions[-stability_window:]).argmax()
                    )
                    if res[most_common] > threshold:
                        palavra = predictor.classes[most_common]

                        # Novo sinal → gera feedback
                        if feedback_engine and palavra != last_class:
                            last_class = palavra
                            user_arr = predictor.get_buffer_array()
                            feedback_msgs = feedback_engine.feedback(user_arr, palavra)
                            exec_score = feedback_engine.score(user_arr, palavra)

                        if not sentence or sentence[-1] != palavra:
                            sentence.append(palavra)

                predictions = predictions[-30:]

            sentence = sentence[-5:]

            # HUD
            draw_history(image, sentence)
            if feedback_engine and feedback_msgs:
                draw_feedback(image, feedback_msgs, exec_score)
            draw_buffer_bar(image, predictor.buffer_progress)
            draw_controls(image)

            cv2.imshow("Libras — Reconhecimento em Tempo Real (TCC)", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                predictor.clear()
                sentence, predictions = [], []
                feedback_msgs, exec_score, last_class = [], None, None

    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado.")