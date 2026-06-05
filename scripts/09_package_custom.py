"""
Empacota o dataset custom processado para upload no Colab.

Uso:
    python scripts/09_package_custom.py
"""

from __future__ import annotations

import tarfile
from pathlib import Path


def main() -> None:
    processed = Path("data/processed_custom")

    if not processed.exists():
        print("❌ Pasta data/processed_custom/ não encontrada.")
        print("   Rode primeiro: python scripts/07_process_custom_dataset.py")
        return

    # Lista classes
    classes = sorted([d.name for d in processed.iterdir() if d.is_dir()])
    print(f"📂 Classes encontradas: {len(classes)}")
    total = 0
    for cls in classes:
        n = len(list((processed / cls).glob("*.npy")))
        total += n
        print(f"   {cls}: {n} amostras")

    print(f"\n📊 Total: {total} amostras")

    # Compacta
    archive = Path("data/custom_dataset.tar.gz")
    if archive.exists():
        archive.unlink()

    print(f"\n📦 Compactando...")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(processed, arcname="processed")

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"\n✅ Pronto: {archive} ({size_mb:.2f} MB)")
    print(f"\n📋 Próximos passos:")
    print(f"   1. Upload de {archive} para a raiz do Google Drive")
    print(f"   2. No Colab, mude ARCHIVE para:")
    print(f"      ARCHIVE = '/content/drive/MyDrive/custom_dataset.tar.gz'")
    print(f"   3. Rode o notebook")


if __name__ == "__main__":
    main()