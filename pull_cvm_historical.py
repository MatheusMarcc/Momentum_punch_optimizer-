"""
Puxa múltiplos anos de Fato Relevante da CVM — SEM rate limit (é download em
bloco por ano, já confirmado funcionando: 28.590 registros só pra 2026).

Uso pensado: sinal agregado de "pulso de governança corporativa" do mercado
(aproximação pro GOVE11, não atribuição exata de constituintes do índice —
não consegui confirmar a lista de empresas do IGCT sem renderização JS na
página oficial; documentado como limitação conhecida).

Uso:
    python pull_cvm_historical.py --anos 2023 2024 2025 2026
"""
from __future__ import annotations

import argparse

from momentum_punch.collectors import cvm_fatos_relevantes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anos", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    parser.add_argument("--out", default="data/raw/cvm_fatos_relevantes.csv")
    args = parser.parse_args()

    df = cvm_fatos_relevantes.fetch_fatos_relevantes(anos=args.anos)
    print(f"\n[pull_cvm_historical] Colunas disponíveis: {list(df.columns)}")
    df.to_csv(args.out, index=False)
    print(f"[pull_cvm_historical] Salvo em {args.out} ({len(df)} registros, {args.anos[0]}-{args.anos[-1]})")


if __name__ == "__main__":
    main()
