"""Ponto de entrada do backend Flask (autenticação + inferência de sinais/letras).

Requer `pip install -e .[api]` (instala o pacote `libras` + Flask/SQLAlchemy/bcrypt).
"""
from libras.api import create_app

app = create_app()

if __name__ == "__main__":
    print("\nBackend Libras rodando em http://localhost:5001")
    print("   Auth:    POST /register, /login, /logout, GET /me")
    print("   Letras:  POST /predict_letter, /record_data")
    print("   Sinais:  POST /predict, /clear, GET /classes")
    print("   Status:  GET /health")
    app.run(host="0.0.0.0", port=5001, debug=False)