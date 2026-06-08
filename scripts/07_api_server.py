"""
Servidor Flask que expõe a inferência LSTM como API REST para o frontend Angular.

Instale as dependências extras antes de rodar:
  pip install flask flask-cors

Uso:
  python scripts/07_api_server.py

Endpoints:
  POST /predict_sign  - Recebe frame base64 e retorna predição da sequência
  POST /reset         - Limpa o buffer do predictor (trocar de sinal alvo)
  GET  /classes       - Lista os sinais que o modelo conhece
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, jsonify, request
from flask_cors import CORS

import mediapipe as mp
from libras.config import load_config, PATHS
from libras.inference.predictor import StreamPredictor
from libras.utils.mediapipe_holistic import extract_keypoints

app = Flask(__name__)
CORS(app,
     origins=["http://localhost:4200"],
     supports_credentials=True,
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

config = load_config()
mp_cfg = config.preprocess["mediapipe"]
inf_cfg = config.inference

# static_image_mode=False: mantém tracking entre frames HTTP consecutivos,
# o que melhora muito a detecção de mãos (mesmo fundamento do modo webcam).
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=mp_cfg["model_complexity"],
    min_detection_confidence=mp_cfg["detection_confidence"],
    min_tracking_confidence=mp_cfg["tracking_confidence"],
)
predictor = StreamPredictor(
    predict_interval=inf_cfg["predict_interval"],
    activity_threshold=inf_cfg["activity_threshold"],
)


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


@app.route("/predict_sign", methods=["POST"])
def predict_sign():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Campo 'image' é obrigatório"}), 400

    frame = _decode_image(data["image"])
    if frame is None:
        return jsonify({"error": "Imagem inválida ou corrompida"}), 400

    # Normaliza resolução: evita que mudança de tamanho entre frames (comum
    # durante inicialização da webcam) trave o tracking do MediaPipe.
    frame = cv2.resize(frame, (640, 480))

    # Extrai keypoints e alimenta o buffer deslizante
    global holistic
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = holistic.process(img_rgb)
    except Exception:
        holistic.close()
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=mp_cfg["model_complexity"],
            min_detection_confidence=mp_cfg["detection_confidence"],
            min_tracking_confidence=mp_cfg["tracking_confidence"],
        )
        results = holistic.process(img_rgb)
    keypoints = extract_keypoints(results)
    predictor.push(keypoints)

    # Força a predição quando o buffer está cheio, independente de is_active.
    # No contexto web frames chegam isolados — is_active pode ser False mesmo
    # com mãos visíveis se o movimento entre frames for pequeno.
    probs = None
    if predictor.is_ready:
        inp = np.expand_dims(predictor.get_buffer_array(), axis=0)
        probs = predictor.model.predict(inp, verbose=0)[0]

    frames_needed = predictor.num_frames - len(predictor.buffer)
    if not predictor.is_ready:
        status_msg = f"Mostre suas mãos na câmera — coletando frames ({len(predictor.buffer)}/{predictor.num_frames})"
    elif not predictor.is_active:
        status_msg = "Mãos não detectadas — levante as mãos na frente da câmera"
    else:
        status_msg = "Analisando sinal..."

    response: dict = {
        "buffer_progress": predictor.buffer_progress,
        "is_active": predictor.is_active,
        "predicted_sign": None,
        "confidence": 0.0,
        "correct": False,
        "top_predictions": [],
        "message": status_msg,
    }

    if probs is not None:
        top_n = inf_cfg.get("top_n_display", 3)
        top_idx = np.argsort(probs)[::-1][:top_n]
        best_idx = int(top_idx[0])
        best_sign = predictor.classes[best_idx]
        best_conf = float(probs[best_idx])
        target = str(data.get("target_sign", "")).strip().lower()
        is_correct = best_sign.lower() == target and best_conf >= predictor.threshold

        response.update(
            {
                "predicted_sign": best_sign,
                "confidence": best_conf,
                "correct": is_correct,
                "top_predictions": [
                    {"sign": predictor.classes[i], "confidence": float(probs[i])}
                    for i in top_idx
                ],
                "message": (
                    "Correto! Muito bem!"
                    if is_correct
                    else f"Detectado: {best_sign} ({best_conf:.0%})"
                ),
            }
        )

    return jsonify(response)


@app.route("/reset", methods=["POST"])
def reset_predictor():
    predictor.clear()
    return jsonify({"success": True})


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": predictor.classes})


if __name__ == "__main__":
    print("\nSinais disponíveis:", predictor.classes)
    print("Acesse: http://localhost:5000\n")
    # threaded=False evita conflitos de thread no MediaPipe / TensorFlow
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
