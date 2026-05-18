"""
Filtra apenas as 10 classes escolhidas dos keypoints processados,
copia para uma pasta separada e gera um .tar.gz pronto para o Colab.

Uso:
    python scripts/prepare_for_colab.py
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from libras.config import PATHS

# As 10 palavras escolhidas
SELECTED_CLASSES = [
    "Abacaxi",
    "Abraço",
    "Amigo",
    "Banana",
    "Casa",
    "Comer",
    "Dinheiro",
    "Escola",
    "Estudo",
    "Família",
]


def main() -> None:
    print("=" * 55)
    print("  Preparando dataset para o Colab")
    print("=" * 55)

    src = PATHS["data_processed"]
    dst = src.parent / "processed_10classes"
    archive = src.parent / "processed_10classes.tar.gz"

    if not src.is_dir():
        print(f"❌ Pasta não encontrada: {src}")
        return

    # Remove versões antigas
    if dst.exists():
        print(f"🗑️  Removendo pasta antiga: {dst}")
        shutil.rmtree(dst)
    if archive.exists():
        print(f"🗑️  Removendo arquivo antigo: {archive}")
        archive.unlink()

    # Copia as 10 classes
    dst.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Copiando {len(SELECTED_CLASSES)} classes...")

    missing = []
    total_samples = 0

    for cls in SELECTED_CLASSES:
        src_cls = src / cls
        if not src_cls.is_dir():
            missing.append(cls)
            continue

        dst_cls = dst / cls
        shutil.copytree(src_cls, dst_cls)

        n = len(list(dst_cls.glob("*.npy")))
        total_samples += n
        print(f"  ✓ {cls}: {n} amostras")

    if missing:
        print(f"\n⚠️  Classes não encontradas: {missing}")
        print("    Verifique se elas existem em data/processed/")
        return

    print(f"\n📊 Total: {total_samples} amostras em {len(SELECTED_CLASSES)} classes")

    # Cria o tar.gz
    print(f"\n📦 Compactando em {archive}...")
    with tarfile.open(archive, "w:gz") as tar:
        # arcname='processed' para extrair como /processed/
        tar.add(dst, arcname="processed")

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"\n✅ Arquivo pronto:")
    print(f"   {archive}")
    print(f"   Tamanho: {size_mb:.2f} MB")

    print(f"\n📋 PRÓXIMOS PASSOS:")
    print(f"   1. Faça upload deste arquivo para a raiz do seu Google Drive:")
    print(f"      {archive}")
    print(f"   2. Abra o notebook treino_colab.ipynb no Google Colab")
    print(f"   3. Vá em Runtime → Change runtime type → T4 GPU")
    print(f"   4. Execute todas as células em ordem")


if __name__ == "__main__":
    main()