"""
Prepara o dataset mesclado:
  1. Copia classes do V-LIBRASIL (já processadas) para o dataset custom
  2. Lista o que falta gravar com webcam

Uso:
    python scripts/10_prepare_merged.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from libras.config import PATHS

# Classes que vêm do V-LIBRASIL (já processadas como .npy)
VLIBRASIL_CLASSES = ["Amigo", "Família"]

# Todas as classes do vocabulário final
ALL_CLASSES = [
    "Sim", "Oi", "Obrigado", "Por favor", "Bom dia",
    "Casa", "Desculpa", "Trabalho", "De nada",
    "Amigo", "Família",
]

# Classes que precisam ser gravadas com webcam
RECORD_CLASSES = ["Casa", "Desculpa", "Trabalho", "De nada", "Amigo", "Família"]


def main() -> None:
    print("=" * 55)
    print("  Preparando dataset mesclado")
    print("=" * 55)

    vlibrasil_processed = PATHS["data_processed"]   # data/processed/
    custom_processed = Path("data/processed_custom")  # destino dos .npy

    # 1. Copia classes do V-LIBRASIL → processed_custom
    print("\n📂 Copiando classes do V-LIBRASIL...")
    for cls in VLIBRASIL_CLASSES:
        src = vlibrasil_processed / cls
        dst = custom_processed / cls

        if not src.is_dir():
            print(f"   ⚠️  {cls} não encontrada em {vlibrasil_processed}")
            continue

        # Cria pasta destino se não existir
        dst.mkdir(parents=True, exist_ok=True)

        # Copia .npy com prefixo para não sobrescrever
        copied = 0
        for npy in src.glob("*.npy"):
            dest_file = dst / f"vlibrasil_{npy.name}"
            if not dest_file.exists():
                shutil.copy2(npy, dest_file)
                copied += 1

        total = len(list(dst.glob("*.npy")))
        print(f"   ✅ {cls}: copiou {copied} novos ({total} total na pasta)")

    # 2. Relatório do estado atual
    print(f"\n{'=' * 55}")
    print(f"  ESTADO ATUAL DO DATASET")
    print(f"{'=' * 55}")

    raw_custom = Path("data/raw_custom")

    for cls in ALL_CLASSES:
        # Vídeos brutos
        raw_dir = raw_custom / cls
        n_raw = len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0

        # Keypoints processados
        proc_dir = custom_processed / cls
        n_proc = len(list(proc_dir.glob("*.npy"))) if proc_dir.exists() else 0

        status = "✅" if (n_raw + n_proc) >= 10 else "🟡"
        print(f"   {status} {cls:12s} | vídeos: {n_raw:3d} | keypoints: {n_proc:3d}")

    # 3. O que falta
    print(f"\n{'=' * 55}")
    print(f"  PRÓXIMOS PASSOS")
    print(f"{'=' * 55}")
    print(f"\n   📹 Gravar com webcam (15 reps cada):")
    for cls in RECORD_CLASSES:
        print(f"      - {cls}")

    print(f"\n   Comando para gravar:")
    print(f"   python scripts/06_record_dataset.py --person lucas")
    print(f"\n   Depois processar:")
    print(f"   python scripts/07_process_custom_dataset.py")
    print(f"\n   Depois empacotar:")
    print(f"   python scripts/09_package_custom.py")


if __name__ == "__main__":
    main()