# Arquitetura do Projeto

Este documento explica as decisões de design e como os módulos se conectam.

## Princípios

1. **Separação de responsabilidades** — cada módulo faz uma coisa só
2. **Configuração externa** — toda config em YAML, nunca hardcoded
3. **Scripts finos** — `scripts/` apenas faz parse de args e delega
4. **Pacote instalável** — `pip install -e .` permite `import libras` de qualquer lugar
5. **Dados ≠ código** — dados em `data/`, código em `src/`, saídas em `artifacts/`

---

## Fluxo de Dados

```
┌──────────────────┐   01_download    ┌──────────────────┐
│   Kaggle Hub     │ ───────────────► │   data/raw/      │
│   V-LIBRASIL     │                  │   *.mp4          │
└──────────────────┘                  └──────────────────┘
                                              │
                                              │  02_preprocess
                                              ▼
                                      ┌──────────────────┐
                                      │ data/processed/  │
                                      │ <classe>/*.npy   │
                                      └──────────────────┘
                                              │
                                              │  03_train
                                              ▼
                                      ┌──────────────────┐
                                      │ artifacts/       │
                                      │   models/        │
                                      │   ├ best_model   │
                                      │   ├ metadata.json│
                                      │   └ X_test.npy   │
                                      └──────────────────┘
                                              │
                                              │  04_evaluate + 05_run_webcam
                                              ▼
                                      ┌──────────────────┐
                                      │ artifacts/       │
                                      │   reports/       │
                                      │   ├ curves.png   │
                                      │   └ confusion.png│
                                      └──────────────────┘
```

---

## Camadas do Pacote `libras`

### 🟦 Utils — `libras/utils/`

Funções puras sem estado. Reutilizáveis em todos os outros módulos.

- **`mediapipe_holistic.py`** — wrapper sobre MediaPipe. Centraliza extração de keypoints (1662 dims), normalização temporal, parsing de nomes do V-LIBRASIL.
- **`io.py`** — JSON load/save.

### 🟩 Data — `libras/data/`

Manipulação de dados em todos os estágios.

- **`download.py`** — Kaggle → `data/raw/`
- **`preprocess.py`** — `data/raw/` (vídeos) → `data/processed/` (keypoints .npy)
- **`loader.py`** — `data/processed/` → arrays em memória + split estratificado

### 🟨 Models — `libras/models/`

Apenas arquiteturas neurais. Sem código de treino.

- **`lstm.py`** — implementação da LSTM bidirecional
- **`registry.py`** — registro nome → builder (facilita adicionar novos modelos)

### 🟥 Training — `libras/training/`

- **`callbacks.py`** — callbacks padrão (EarlyStopping, ReduceLR, etc.)
- **`trainer.py`** — orquestra todo o pipeline de treino
- **`evaluator.py`** — métricas, matriz de confusão, classification report

### 🟪 Inference — `libras/inference/`

- **`predictor.py`** — `StreamPredictor`: modelo + buffer deslizante
- **`feedback.py`** — `FeedbackEngine`: compara execução com referência
- **`hud.py`** — funções de desenho da interface
- **`webcam.py`** — orquestra captura + predição + feedback + HUD

---

## Como adicionar uma nova arquitetura

1. Crie `src/libras/models/transformer.py` com função `build_transformer(...)`
2. Registre em `src/libras/models/registry.py`:
   ```python
   from libras.models.transformer import build_transformer
   MODEL_REGISTRY["transformer"] = build_transformer
   ```
3. Use:
   ```bash
   python scripts/03_train.py --model transformer
   ```

---

## Como adicionar um novo passo no pipeline

1. Crie a função em `src/libras/<categoria>/<nome>.py`
2. Crie `scripts/0X_<nome>.py` que apenas parseia args e chama a função
3. Atualize o README
