"""Rotas de reconhecimento de sinais (LSTM) — /predict, /clear, /classes.

Reaproveita libras.inference.predictor.StreamPredictor e
libras.utils.mediapipe_holistic.extract_keypoints em vez de reimplementar
a extração de keypoints (como o backend/app.py original fazia).
"""
from __future__ import annotations

import base64

import cv2
import mediapipe as mp
import numpy as np
from flask import Blueprint, jsonify, request

from libras.config import load_config
from libras.inference.predictor import StreamPredictor
from libras.utils.mediapipe_holistic import extract_keypoints

sign_bp = Blueprint("sign", __name__)

_config = load_config()
_mp_cfg = _config.preprocess["mediapipe"]

_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=_mp_cfg["model_complexity"],
    min_detection_confidence=_mp_cfg["detection_confidence"],
    min_tracking_confidence=_mp_cfg["tracking_confidence"],
)

print("\n--- IA de Palavras (LSTM) ---")
try:
    _predictor: StreamPredictor | None = StreamPredictor()
    print(f"Modelo LSTM carregado: {len(_predictor.classes)} classes")
    print(f"   Classes: {_predictor.classes}")
except FileNotFoundError as e:
    _predictor = None
    print(f"Erro ao carregar modelo LSTM: {e}")
    print("   Rota /predict ficara desativada.")

_last_prediction: dict | None = None


def _decode_image(image_b64: str) -> np.ndarray | None:
    """Decodifica base64 (com ou sem prefixo data:image/...) para array BGR."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@sign_bp.route("/predict", methods=["POST"])
def predict_word():
    global _last_prediction

    if _predictor is None:
        return jsonify({"error": "Modelo LSTM nao carregado."}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Campo image obrigatorio"}), 400

    frame = _decode_image(data["image"])
    if frame is None:
        return jsonify({"error": "Imagem invalida ou corrompida"}), 400

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = _holistic.process(rgb)

        _predictor.push(extract_keypoints(results))
        progress = _predictor.buffer_progress

        if _predictor.is_ready:
            # Força a predição com o buffer cheio: frames chegam isolados via
            # HTTP, então o gate de atividade do StreamPredictor (pensado pro
            # loop contínuo da webcam local) fica desligado aqui de propósito.
            inp = np.expand_dims(_predictor.get_buffer_array(), axis=0)
            probs = _predictor.model.predict(inp, verbose=0)[0]
            top_indices = np.argsort(probs)[::-1][:3]

            predictions = [
                {"word": _predictor.classes[idx], "confidence": float(probs[idx]), "rank": rank + 1}
                for rank, idx in enumerate(top_indices)
            ]

            _last_prediction = {
                "buffer_progress": 1.0,
                "ready": True,
                "predictions": predictions,
                "top1_word": _predictor.classes[top_indices[0]],
                "top1_confidence": float(probs[top_indices[0]]),
                "is_confident": float(probs[top_indices[0]]) >= 0.25,
            }
            return jsonify(_last_prediction)

        return jsonify({
            "buffer_progress": progress,
            "ready": False,
            "predictions": _last_prediction["predictions"] if _last_prediction else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sign_bp.route("/clear", methods=["POST"])
def clear_buffer():
    global _last_prediction
    if _predictor is not None:
        _predictor.clear()
    _last_prediction = None
    return jsonify({"status": "ok", "message": "Buffer limpo"})


@sign_bp.route("/classes", methods=["GET"])
def get_classes():
    classes = _predictor.classes if _predictor is not None else []
    return jsonify({"classes": classes, "num_classes": len(classes)})


def is_loaded() -> bool:
    return _predictor is not None


def num_classes() -> int:
    return len(_predictor.classes) if _predictor is not None else 0


def buffer_size() -> int:
    return len(_predictor.buffer) if _predictor is not None else 0
