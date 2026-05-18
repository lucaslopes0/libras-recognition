"""
Script de diagnóstico — verifica o estado real do pipeline.

Uso:
    python scripts/00_diagnose.py
"""

from __future__ import annotations

import numpy as np

from libras.config import PATHS, load_config
from libras.data.loader import (
    _required_min_samples, list_classes, load_arrays, split_data,
)


def main() -> None:
    print("=" * 60)
    print("  DIAGNÓSTICO DO PIPELINE")
    print("=" * 60)

    # 1. Config
    cfg = load_config("configs/default.yaml")
    print("\n[1] CONFIGURAÇÃO CARREGADA:")
    print(f"    max_classes:       {cfg.dataset['max_classes']}")
    print(f"    test_size:         {cfg.training['test_size']}")
    print(f"    val_size:          {cfg.training['val_size']}")
    print(f"    batch_size:        {cfg.training['batch_size']}")
    aug = cfg.training.get('augmentation', {})
    print(f"    augmentation:      {aug.get('enabled')} (×{aug.get('multiplier')})")

    # 2. Dados processados
    processed = PATHS['data_processed']
    print(f"\n[2] DADOS EM {processed}:")
    if not processed.exists():
        print("    ❌ Pasta não existe! Rode o preprocess primeiro.")
        return

    all_classes = sorted([d.name for d in processed.iterdir() if d.is_dir()])
    print(f"    Total de classes no disco: {len(all_classes)}")

    counts = [len(list((processed / c).glob('*.npy'))) for c in all_classes]
    print(f"    Amostras por classe — min: {min(counts)} | max: {max(counts)} | média: {np.mean(counts):.1f}")

    distribution = {}
    for n in counts:
        distribution[n] = distribution.get(n, 0) + 1
    print(f"    Distribuição:")
    for n_amostras, n_classes in sorted(distribution.items()):
        print(f"      {n_amostras} amostras → {n_classes} classes")

    # 3. O que o list_classes vai filtrar
    min_s = _required_min_samples(
        test_size=cfg.training['test_size'],
        val_size=cfg.training['val_size'],
    )
    print(f"\n[3] FILTRO DE CLASSES:")
    print(f"    min_samples calculado: {min_s}")
    valid = list_classes(min_samples=min_s)
    print(f"    Classes que passam:    {len(valid)}")

    # 4. Verificar se max_classes está sendo aplicado
    max_cls = cfg.dataset.get('max_classes')
    print(f"\n[4] LIMITE DE CLASSES:")
    if max_cls is None:
        print(f"    ⚠️  max_classes = null — usando TODAS as {len(valid)} classes")
    else:
        print(f"    ✅ max_classes = {max_cls}")
        print(f"    Mas o trainer.py não aplica esse filtro automaticamente!")
        print(f"    → Precisamos aplicar manualmente.")

    # 5. Tamanho esperado do treino
    n_train_expected = int(len(valid) * (1 - cfg.training['test_size'] - cfg.training['val_size']))
    n_aug = n_train_expected * aug.get('multiplier', 1) if aug.get('enabled') else n_train_expected
    n_steps = n_aug // cfg.training['batch_size']
    print(f"\n[5] PREVISÃO DO TREINO (sem max_classes):")
    print(f"    Treino:        ~{n_train_expected} amostras")
    print(f"    Após aug.:     ~{n_aug} amostras")
    print(f"    Steps/época:   ~{n_steps}")

    # 6. Artefatos antigos
    print(f"\n[6] ARTEFATOS EM {PATHS['models']}:")
    if PATHS['models'].exists():
        for f in PATHS['models'].iterdir():
            size = f.stat().st_size / 1024
            print(f"    {f.name} ({size:.1f} KB)")
    else:
        print("    (vazio)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()