"""
Procura por classes similares às que estão faltando.
Útil quando o nome no V-LIBRASIL não bate exatamente.

Uso:
    python scripts/find_classes.py
"""

from __future__ import annotations

from pathlib import Path

from libras.config import PATHS

# Classes que queremos encontrar (parcial ou completo)
SEARCH_TERMS = [
    "brincar", "brinca", "brinquedo",
    "cachorro", "cão", "cao",
    "estudar", "estudo", "estuda",
    "escola",
]


def main() -> None:
    src = PATHS["data_processed"]
    if not src.is_dir():
        print(f"❌ Pasta não encontrada: {src}")
        return

    all_classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    print(f"Total de classes disponíveis: {len(all_classes)}\n")

    # Busca case-insensitive e parcial
    print("=" * 55)
    print("  Classes encontradas para cada termo")
    print("=" * 55)

    for term in SEARCH_TERMS:
        term_lower = term.lower()
        matches = [c for c in all_classes if term_lower in c.lower()]
        print(f"\n🔎 Termo: '{term}'")
        if matches:
            for m in matches:
                n = len(list((src / m).glob("*.npy")))
                print(f"   ✓ '{m}' ({n} amostras)")
        else:
            print(f"   (nenhuma classe encontrada)")

    print("\n" + "=" * 55)
    print("  Dica: classes começadas com B, C ou E (primeiras 20 de cada)")
    print("=" * 55)

    for letter in ["B", "C", "E"]:
        starting = [c for c in all_classes if c.upper().startswith(letter)][:20]
        print(f"\n📚 Iniciando com '{letter}':")
        for c in starting:
            print(f"   {c}")


if __name__ == "__main__":
    main()