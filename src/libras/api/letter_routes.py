"""Rotas de reconhecimento de letras estáticas (Keras + MediaPipe Hands).

/predict_letter e /record_data. Modelo separado do LSTM de sinais —
classifica um único frame de landmarks da mão (sem dimensão temporal).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile

import cv2
import h5py
import joblib
import mediapipe as mp
import numpy as np
from flask import Blueprint, jsonify, request
from tensorflow.keras.layers import InputLayer as _InputLayer
from tensorflow.keras.models import load_model as keras_load_model

from libras.config import PATHS

letter_bp = Blueprint("letter", __name__)

_MODEL_PATH = PATHS["models_letters"] / "keras_landmarks_model.h5"
_ENCODER_PATH = PATHS["models_letters"] / "label_encoder.pkl"
_CAPTURE_DIR = PATHS["data_letters_captured"]
_CONFIDENCE_THRESHOLD = 0.8


class _CompatInputLayer(_InputLayer):
    """Modelo foi salvo com `batch_shape` (formato antigo do Keras);
    a versão local do Keras espera `input_shape`."""

    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs.setdefault("input_shape", batch_shape[1:])
        super().__init__(**kwargs)


def _patch_h5_dtype_policy(src_path) -> str:
    """Copia o .h5 para um temp e remove o campo DTypePolicy do config —
    incompatível entre a versão do Keras usada no treino (Colab) e a local."""

    def _patch(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "DTypePolicy" and "config" in obj:
                return obj["config"].get("name", "float32")
            return {k: _patch(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_patch(i) for i in obj]
        return obj

    tmp = tempfile.mktemp(suffix=".h5")
    shutil.copy2(str(src_path), tmp)
    with h5py.File(tmp, "r+") as f:
        raw = f.attrs.get("model_config")
        if raw is not None:
            cfg_str = raw if isinstance(raw, str) else raw.decode("utf-8")
            f.attrs["model_config"] = json.dumps(_patch(json.loads(cfg_str))).encode("utf-8")
    return tmp


def _load_letter_model():
    if not _MODEL_PATH.exists() or not _ENCODER_PATH.exists():
        if not _MODEL_PATH.exists():
            print(f"Modelo de letras nao encontrado: {_MODEL_PATH}")
        if not _ENCODER_PATH.exists():
            print(f"Encoder nao encontrado: {_ENCODER_PATH}")
        return None, None

    tmp_path = _patch_h5_dtype_policy(_MODEL_PATH)
    try:
        model = keras_load_model(
            tmp_path, custom_objects={"InputLayer": _CompatInputLayer}, compile=False,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    encoder = joblib.load(str(_ENCODER_PATH))
    return model, encoder


print("\n--- IA de Letras (Keras) ---")
_hands = None
try:
    _letter_model, _label_encoder = _load_letter_model()
    _letter_classes = _label_encoder.classes_ if _label_encoder is not None else None
    if _letter_model is not None:
        print(f"Modelo de letras carregado: {len(_letter_classes)} classes")
        print(f"   Classes: {list(_letter_classes)}")
        _hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
        )
    else:
        print("   Rota /predict_letter ficara desativada.")
except Exception as e:
    _letter_model = None
    _label_encoder = None
    _letter_classes = None
    print(f"Erro ao carregar modelo de letras: {e}")
    print("   Rota /predict_letter ficara desativada.")


def _decode_image(image_b64: str) -> np.ndarray | None:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _hand_bounding_box(landmarks, w: int, h: int, margin: int = 20) -> tuple[int, int, int, int]:
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
    return (
        max(0, x_min - margin), max(0, y_min - margin),
        min(w, x_max + margin), min(h, y_max + margin),
    )


@letter_bp.route("/predict_letter", methods=["POST"])
def predict_letter():
    if _hands is None or _letter_model is None:
        return jsonify({"success": False, "message": "Modelo de letras nao carregado."}), 503

    data = request.get_json()
    if not data or "image" not in data or "target_letter" not in data:
        return jsonify({"success": False, "message": "Dados de imagem ou letra alvo faltando."}), 400

    try:
        img_bgr = _decode_image(data["image"])
        target_letter = data["target_letter"]
        if img_bgr is None:
            return jsonify({"success": False, "message": "Erro ao processar a imagem."}), 400

        results = _hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return jsonify({
                "success": True,
                "correct": False,
                "predicted_letter": "Nenhuma mao detectada",
                "message": "Nenhuma mao detectada.",
            })

        landmarks = results.multi_hand_landmarks[0].landmark
        x = np.array([[lm.x, lm.y] for lm in landmarks]).flatten().reshape(1, -1)

        probas = _letter_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probas))
        confidence = float(probas[idx])
        predicted_letter = _label_encoder.inverse_transform([idx])[0]

        is_correct = False
        if confidence < _CONFIDENCE_THRESHOLD:
            predicted_letter = "Movimento Invalido"
            message = "Movimento Invalido."
        elif predicted_letter.upper() != target_letter.upper():
            message = f"Gesto incorreto. Voce fez a letra: {predicted_letter}"
        else:
            is_correct = True
            message = "Gesto Correto!"

        h, w, _ = img_bgr.shape
        box = _hand_bounding_box(landmarks, w, h)

        return jsonify({
            "success": True,
            "correct": is_correct,
            "predicted_letter": predicted_letter,
            "target_letter": target_letter,
            "box": list(box),
            "confidence": confidence,
            "message": message,
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@letter_bp.route("/record_data", methods=["POST"])
def record_data():
    if _hands is None:
        return jsonify({"success": False, "message": "MediaPipe nao carregado."}), 503

    data = request.get_json()
    if not data or "image" not in data or "label" not in data:
        return jsonify({"success": False, "message": "Dados faltando."}), 400

    try:
        img_bgr = _decode_image(data["image"])
        label_text = data["label"].upper()
        if img_bgr is None:
            return jsonify({"success": False, "message": "Erro ao processar imagem."}), 400

        results = _hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return jsonify({"success": False, "message": "Nenhuma mao detectada."})

        h, w, _ = img_bgr.shape
        x_min, y_min, x_max, y_max = _hand_bounding_box(
            results.multi_hand_landmarks[0].landmark, w, h,
        )
        hand_img = img_bgr[y_min:y_max, x_min:x_max]

        folder_path = _CAPTURE_DIR / label_text
        folder_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{len(list(folder_path.glob('*.png'))) + 1}.png"
        cv2.imwrite(str(folder_path / file_name), hand_img)

        return jsonify({"success": True, "message": f"Imagem salva como {file_name}."})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


def is_loaded() -> bool:
    return _letter_model is not None


def num_classes() -> int:
    return len(_letter_classes) if _letter_classes is not None else 0
