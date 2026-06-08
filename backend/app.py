import base64
import json
import os
import secrets
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

# Suprime avisos do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import load_model

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
MODEL_PATH = MODELS_DIR / "best_model.keras"
METADATA_PATH = MODELS_DIR / "metadata.json"
DB_PATH = PROJECT_ROOT / "backend" / "instance" / "site.db"

NUM_FRAMES = 30
KEYPOINT_DIM = 1662
BUFFER_SIZE = NUM_FRAMES
CONFIDENCE_THRESHOLD = 0.25

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


# --- Database Model ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    token = db.Column(db.String(64), unique=True, nullable=True)

    def __repr__(self):
        return f"User('{self.username}')"


# --- Create DB ---
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
with app.app_context():
    db.create_all()
    print(f"Banco de dados: {DB_PATH}")


# --- Load AI Model ---
print("Carregando modelo de IA...")
modelo = load_model(str(MODEL_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

CLASSES = metadata["classes"]
NUM_CLASSES = metadata["num_classes"]
print(f"Modelo carregado: {NUM_CLASSES} classes")
print(f"   Classes: {CLASSES}")

# --- MediaPipe ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# --- Frame Buffer ---
frame_buffer = deque(maxlen=BUFFER_SIZE)
last_prediction = None


# --- Helper Functions ---
def extract_keypoints(results):
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


def get_current_user():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        return User.query.filter_by(token=token).first()
    return None


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

    if len(password) < 6:
        return jsonify({"message": "Senha deve ter no minimo 6 caracteres."}), 400

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
    user = get_current_user()
    if user:
        user.token = None
        db.session.commit()
    return jsonify({"message": "Logout realizado."}), 200


@app.route("/me", methods=["GET"])
def me():
    user = get_current_user()
    if not user:
        return jsonify({"message": "Nao autenticado."}), 401
    return jsonify({"username": user.username, "id": user.id}), 200


# =============================================
#   AI ROUTES
# =============================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": modelo is not None,
        "classes": NUM_CLASSES,
        "buffer_size": len(frame_buffer),
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": CLASSES, "num_classes": NUM_CLASSES})


@app.route("/clear", methods=["POST"])
def clear_buffer():
    global last_prediction
    frame_buffer.clear()
    last_prediction = None
    return jsonify({"status": "ok", "message": "Buffer limpo"})


@app.route("/predict", methods=["POST"])
def predict():
    global last_prediction

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Campo image obrigatorio"}), 400

    try:
        frame = decode_image(data["image"])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)

        kp = extract_keypoints(results)
        frame_buffer.append(kp)

        progress = len(frame_buffer) / BUFFER_SIZE

        if len(frame_buffer) >= BUFFER_SIZE:
            sequence = np.array(list(frame_buffer), dtype=np.float32)
            sequence = sequence.reshape(1, NUM_FRAMES, KEYPOINT_DIM)

            probs = modelo.predict(sequence, verbose=0)[0]
            top_indices = np.argsort(probs)[::-1][:3]

            predictions = []
            for idx in top_indices:
                predictions.append({
                    "word": CLASSES[idx],
                    "confidence": float(probs[idx]),
                    "rank": len(predictions) + 1,
                })

            last_prediction = {
                "buffer_progress": 1.0,
                "ready": True,
                "predictions": predictions,
                "top1_word": CLASSES[top_indices[0]],
                "top1_confidence": float(probs[top_indices[0]]),
                "is_confident": float(probs[top_indices[0]]) >= CONFIDENCE_THRESHOLD,
            }
            return jsonify(last_prediction)

        return jsonify({
            "buffer_progress": progress,
            "ready": False,
            "predictions": last_prediction["predictions"] if last_prediction else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================
#   STARTUP
# =============================================

if __name__ == "__main__":
    print("")
    print("Backend LibraKids rodando em http://localhost:5001")
    print("   Auth:  POST /register, /login, /logout, GET /me")
    print("   IA:    POST /predict, /clear, GET /classes, /health")
    app.run(host="0.0.0.0", port=5001, debug=False)