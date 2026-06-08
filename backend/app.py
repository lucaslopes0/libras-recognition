import base64
import json
import os
import secrets
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import load_model as keras_load_model

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

# Modelo LSTM (palavras)
LSTM_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_model.keras"
LSTM_METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "metadata.json"

# Modelo Keras (letras) - na pasta backend/ml_model/
LETTER_MODEL_PATH = BACKEND_DIR / "ml_model" / "keras_landmarks_model.h5"
LETTER_ENCODER_PATH = BACKEND_DIR / "ml_model" / "label_encoder.pkl"

DB_PATH = BACKEND_DIR / "instance" / "site.db"

NUM_FRAMES = 30
KEYPOINT_DIM = 1662
BUFFER_SIZE = NUM_FRAMES

# --- Flask App ---
app = Flask(__name__)
app.config["SECRET_KEY"] = "librakids-secret-key-2024"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


# --- Database ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    token = db.Column(db.String(64), unique=True, nullable=True)


DB_PATH.parent.mkdir(parents=True, exist_ok=True)
with app.app_context():
    db.create_all()
    print(f"Banco de dados: {DB_PATH}")


# =============================================
#   MODELO 1: LSTM (palavras/sinais completos)
# =============================================
print("\n--- IA de Palavras (LSTM) ---")
lstm_model = None
LSTM_CLASSES = []
LSTM_NUM_CLASSES = 0

try:
    lstm_model = keras_load_model(str(LSTM_MODEL_PATH))
    with open(LSTM_METADATA_PATH, "r", encoding="utf-8") as f:
        lstm_metadata = json.load(f)
    LSTM_CLASSES = lstm_metadata["classes"]
    LSTM_NUM_CLASSES = lstm_metadata["num_classes"]
    print(f"Modelo LSTM carregado: {LSTM_NUM_CLASSES} classes")
    print(f"   Classes: {LSTM_CLASSES}")
except Exception as e:
    print(f"Erro ao carregar modelo LSTM: {e}")
    print("   Rota /predict ficara desativada.")

# MediaPipe Holistic (para palavras)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

frame_buffer = deque(maxlen=BUFFER_SIZE)
last_prediction = None


# =============================================
#   MODELO 2: Keras (letras estaticas)
# =============================================
print("\n--- IA de Letras (Keras) ---")
letter_model = None
label_encoder = None
letter_classes = None
hands = None

try:
    import joblib
    import h5py
    import json
    import tempfile
    import shutil
    from tensorflow.keras.layers import InputLayer as _InputLayer

    class _CompatInputLayer(_InputLayer):
        def __init__(self, batch_shape=None, **kwargs):
            if batch_shape is not None:
                kwargs.setdefault('input_shape', batch_shape[1:])
            super().__init__(**kwargs)

    def _patch_h5_config(src_path):
        """Copia o .h5 para um temp, remove DTypePolicy do config JSON e retorna o path."""
        def _patch(obj):
            if isinstance(obj, dict):
                if obj.get('class_name') == 'DTypePolicy' and 'config' in obj:
                    return obj['config'].get('name', 'float32')
                return {k: _patch(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_patch(i) for i in obj]
            return obj

        tmp = tempfile.mktemp(suffix='.h5')
        shutil.copy2(str(src_path), tmp)
        with h5py.File(tmp, 'r+') as f:
            raw = f.attrs.get('model_config')
            if raw is not None:
                cfg_str = raw if isinstance(raw, str) else raw.decode('utf-8')
                patched = json.dumps(_patch(json.loads(cfg_str)))
                f.attrs['model_config'] = patched.encode('utf-8')
        return tmp

    if LETTER_MODEL_PATH.exists() and LETTER_ENCODER_PATH.exists():
        _tmp_model = _patch_h5_config(LETTER_MODEL_PATH)
        try:
            letter_model = keras_load_model(
                _tmp_model,
                custom_objects={'InputLayer': _CompatInputLayer},
                compile=False,
            )
        finally:
            try:
                os.unlink(_tmp_model)
            except OSError:
                pass
        label_encoder = joblib.load(str(LETTER_ENCODER_PATH))
        letter_classes = label_encoder.classes_
        print(f"Modelo de letras carregado: {len(letter_classes)} classes")
        print(f"   Classes: {list(letter_classes)}")

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
    else:
        if not LETTER_MODEL_PATH.exists():
            print(f"Modelo nao encontrado: {LETTER_MODEL_PATH}")
        if not LETTER_ENCODER_PATH.exists():
            print(f"Encoder nao encontrado: {LETTER_ENCODER_PATH}")
        print("   Rota /predict_letter ficara desativada.")
except Exception as e:
    print(f"Erro ao carregar modelo de letras: {e}")
    print("   Rota /predict_letter ficara desativada.")


# --- Helpers ---
def extract_holistic_keypoints(results):
    pose = (
        np.array(
            [[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[r.x, r.y, r.z] for r in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def decode_image(image_data):
    if "," in image_data:
        image_data = image_data.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


# =============================================
#   AUTH ROUTES
# =============================================

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Dados obrigatorios."}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"message": "Usuario e senha obrigatorios."}), 400

    if len(username) < 2 or len(username) > 20:
        return jsonify({"message": "Usuario deve ter entre 2 e 20 caracteres."}), 400

    existing = User.query.filter_by(username=username).first()
    if existing:
        return jsonify({"message": "Este nome de usuario ja esta em uso."}), 409

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(username=username, password=hashed)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Conta criada com sucesso!"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Dados obrigatorios."}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"message": "Usuario e senha obrigatorios."}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Usuario ou senha incorretos."}), 401

    token = secrets.token_hex(32)
    user.token = token
    db.session.commit()

    return jsonify({
        "access_token": token,
        "username": user.username,
        "message": "Login bem-sucedido!"
    }), 200


@app.route("/logout", methods=["POST"])
def logout():
    return jsonify({"message": "Logout realizado."}), 200


@app.route("/me", methods=["GET"])
def me():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        user = User.query.filter_by(token=token).first()
        if user:
            return jsonify({"username": user.username, "id": user.id}), 200
    return jsonify({"message": "Nao autenticado."}), 401


# =============================================
#   IA PALAVRAS (LSTM) - /predict
# =============================================

@app.route("/predict", methods=["POST"])
def predict_word():
    global last_prediction

    if lstm_model is None:
        return jsonify({"error": "Modelo LSTM nao carregado."}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Campo image obrigatorio"}), 400

    try:
        frame = decode_image(data["image"])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)

        kp = extract_holistic_keypoints(results)
        frame_buffer.append(kp)

        progress = len(frame_buffer) / BUFFER_SIZE

        if len(frame_buffer) >= BUFFER_SIZE:
            sequence = np.array(list(frame_buffer), dtype=np.float32)
            sequence = sequence.reshape(1, NUM_FRAMES, KEYPOINT_DIM)

            probs = lstm_model.predict(sequence, verbose=0)[0]
            top_indices = np.argsort(probs)[::-1][:3]

            predictions = []
            for idx in top_indices:
                predictions.append({
                    "word": LSTM_CLASSES[idx],
                    "confidence": float(probs[idx]),
                    "rank": len(predictions) + 1,
                })

            last_prediction = {
                "buffer_progress": 1.0,
                "ready": True,
                "predictions": predictions,
                "top1_word": LSTM_CLASSES[top_indices[0]],
                "top1_confidence": float(probs[top_indices[0]]),
                "is_confident": float(probs[top_indices[0]]) >= 0.25,
            }
            return jsonify(last_prediction)

        return jsonify({
            "buffer_progress": progress,
            "ready": False,
            "predictions": last_prediction["predictions"] if last_prediction else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_buffer():
    global last_prediction
    frame_buffer.clear()
    last_prediction = None
    return jsonify({"status": "ok", "message": "Buffer limpo"})


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": LSTM_CLASSES, "num_classes": LSTM_NUM_CLASSES})


# =============================================
#   IA LETRAS (Keras) - /predict_letter
# =============================================

@app.route("/predict_letter", methods=["POST"])
def predict_letter():
    if hands is None or letter_model is None:
        return jsonify({
            "success": False,
            "message": "Modelo de letras nao carregado."
        }), 503

    data = request.get_json()
    if not data or "image" not in data or "target_letter" not in data:
        return jsonify({
            "success": False,
            "message": "Dados de imagem ou letra alvo faltando."
        }), 400

    try:
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        target_letter = data["target_letter"]

        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({
                "success": False,
                "message": "Erro ao processar a imagem."
            }), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({
                "success": True,
                "correct": False,
                "predicted_letter": "Nenhuma mao detectada",
                "message": "Nenhuma mao detectada."
            })

        # Extrai 21 landmarks da mao (x, y)
        landmarks = results.multi_hand_landmarks[0].landmark
        processed_landmarks = np.array(
            [[lm.x, lm.y] for lm in landmarks]
        ).flatten()

        # Predicao com modelo Keras
        x = processed_landmarks.reshape(1, -1)
        probas = letter_model.predict(x, verbose=0)[0]
        predicted_label_index = int(np.argmax(probas))
        confidence_score = float(probas[predicted_label_index])
        predicted_letter = label_encoder.inverse_transform([predicted_label_index])[0]

        LETTER_THRESHOLD = 0.8
        is_correct = False

        if confidence_score < LETTER_THRESHOLD:
            predicted_letter = "Movimento Invalido"
            message = "Movimento Invalido."
        elif predicted_letter.upper() != target_letter.upper():
            message = f"Gesto incorreto. Voce fez a letra: {predicted_letter}"
        else:
            is_correct = True
            message = "Gesto Correto!"

        # Bounding box da mao
        h, w, _ = img_bgr.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for landmark in results.multi_hand_landmarks[0].landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        return jsonify({
            "success": True,
            "correct": is_correct,
            "predicted_letter": predicted_letter,
            "target_letter": target_letter,
            "box": [x_min, y_min, x_max, y_max],
            "confidence": confidence_score,
            "message": message,
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# =============================================
#   RECORD DATA (coleta de dados)
# =============================================

@app.route("/record_data", methods=["POST"])
def record_data():
    if hands is None:
        return jsonify({"success": False, "message": "MediaPipe nao carregado."}), 503

    data = request.get_json()
    if not data or "image" not in data or "label" not in data:
        return jsonify({"success": False, "message": "Dados faltando."}), 400

    try:
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        label_text = data["label"].upper()

        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"success": False, "message": "Erro ao processar imagem."}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({"success": False, "message": "Nenhuma mao detectada."})

        h, w, _ = img_bgr.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for landmark in results.multi_hand_landmarks[0].landmark:
            lx, ly = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, lx)
            y_min = min(y_min, ly)
            x_max = max(x_max, lx)
            y_max = max(y_max, ly)

        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        hand_img = img_bgr[y_min:y_max, x_min:x_max]

        folder_path = BACKEND_DIR / "data" / label_text
        folder_path.mkdir(parents=True, exist_ok=True)

        existing = list(folder_path.glob("*.png"))
        file_count = len(existing) + 1
        file_name = f"{file_count}.png"
        file_path = folder_path / file_name

        cv2.imwrite(str(file_path), hand_img)

        return jsonify({"success": True, "message": f"Imagem salva como {file_name}."})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# =============================================
#   HEALTH
# =============================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "lstm_loaded": lstm_model is not None,
        "lstm_classes": LSTM_NUM_CLASSES,
        "letter_loaded": letter_model is not None,
        "letter_classes": len(letter_classes) if letter_classes is not None else 0,
        "buffer_size": len(frame_buffer),
    })


# =============================================
#   STARTUP
# =============================================

if __name__ == "__main__":
    print("")
    print("Backend LibraKids rodando em http://localhost:5001")
    print("   Auth:    POST /register, /login, /logout, GET /me")
    print("   Letras:  POST /predict_letter, /record_data  (Keras)")
    print("   Sinais:  POST /predict, /clear, GET /classes  (LSTM)")
    print("   Status:  GET /health")
    app.run(host="0.0.0.0", port=5001, debug=False)