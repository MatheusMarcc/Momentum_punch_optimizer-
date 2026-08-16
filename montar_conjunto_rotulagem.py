"""
Monta o conjunto CONGELADO de rotulagem manual — seção 4 do pré-relatório
("será mantido um conjunto congelado de manchetes [...] mudanças de prompt ou
modelo deverão reexecutar essa suíte e registrar divergências").

O conjunto serve pra medir DUAS coisas que hoje são só suposição:

  1. O FILTRO acerta? (text_filter.py) — por isso a amostra inclui textos que o
     filtro ACEITOU e textos que ele REJEITOU. Só com os aceitos dá pra medir
     precisão, nunca recall: um filtro que aceita 3 textos perfeitos e descarta
     900 pertinentes teria precisão 100% e seria inútil.

  2. O CLASSIFICADOR acerta? (motor GenAI) — sentimento e relevância que a LLM
     atribui, comparados com julgamento humano no mesmo texto.

A amostra é sorteada com semente fixa e gravada com hash. Rotular, reamostrar e
rotular de novo até o número agradar é o mesmo vício que a gente corrigiu no
tilt — o conjunto é escolhido antes de ver o resultado e não muda depois.

Uso:
    python montar_conjunto_rotulagem.py
    python montar_conjunto_rotulagem.py --n-por-ticker 30

Saída: data/processed/conjunto_rotulagem.csv  (preencher as 2 colunas vazias)
"""
from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

from momentum_punch import config, text_filter

SEMENTE = 20260816  # data da montagem; fixa e declarada

INSTRUCOES = """\
COMO ROTULAR (preencha só as duas últimas colunas, deixe o resto intacto)

  relevante_humano : 1 se o texto tem relação defensável com o TEMA do ticker
                     indicado na linha; 0 se não tem.
                     Julgue relação com o TEMA (governança, ESG, energia,
                     macro), não com a empresa específica.

  sentimento_humano: -1 negativo | 0 neutro/ambíguo | +1 positivo,
                     PARA O TEMA/ATIVO indicado — não "a notícia é boa pra
                     humanidade". Deixe em branco se relevante_humano = 0.

Regras que evitam contaminar a medição:
  - Não olhe as colunas de saída do modelo antes de rotular (elas nem estão
    aqui, de propósito).
  - Se ficar em dúvida entre neutro e um extremo, marque 0. A taxa de
    ambiguidade é resultado, não problema.
  - Não pule linhas: linha não rotulada vira dado faltante e reduz a amostra.
"""


def montar(n_por_ticker: int, cvm_csv: str) -> pd.DataFrame:
    df = pd.read_csv(cvm_csv, low_memory=False)
    col = next((c for c in df.columns if "assunto" in c.lower()), None)
    if col is None:
        raise ValueError(f"Coluna de assunto não encontrada em {cvm_csv}")

    textos = sorted({t.strip() for t in df[col].fillna("").astype(str) if len(t.strip()) > 15})
    print(f"[rotulagem] {len(textos)} textos únicos no corpus")

    linhas = []
    for ticker in config.TICKERS:
        aceitos = [t for t in textos if text_filter.texto_e_relevante(t, ticker)]
        rejeitados = [t for t in textos if not text_filter.texto_e_relevante(t, ticker)]

        # 2/3 aceitos, 1/3 rejeitados: precisão precisa de mais massa, mas sem
        # rejeitados não há como estimar o que o filtro está deixando passar
        n_ac = min(len(aceitos), int(n_por_ticker * 2 / 3))
        n_rj = min(len(rejeitados), n_por_ticker - n_ac)

        amostra_ac = pd.Series(aceitos).sample(n_ac, random_state=SEMENTE).tolist() if n_ac else []
        amostra_rj = pd.Series(rejeitados).sample(n_rj, random_state=SEMENTE).tolist() if n_rj else []

        for t in amostra_ac:
            linhas.append({"ticker": ticker, "texto": t, "filtro_aceitou": 1})
        for t in amostra_rj:
            linhas.append({"ticker": ticker, "texto": t, "filtro_aceitou": 0})

        print(f"[rotulagem] {ticker}: {n_ac} aceitos + {n_rj} rejeitados "
              f"(universo: {len(aceitos)} aceitos / {len(rejeitados)} rejeitados)")

    conjunto = pd.DataFrame(linhas)
    # embaralha pra quem rotula não perceber o padrão "primeiros são aceitos" e
    # deixar isso influenciar o julgamento
    conjunto = conjunto.sample(frac=1.0, random_state=SEMENTE).reset_index(drop=True)
    conjunto.insert(0, "id", range(1, len(conjunto) + 1))
    conjunto["relevante_humano"] = ""
    conjunto["sentimento_humano"] = ""
    return conjunto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-por-ticker", type=int, default=30)
    parser.add_argument("--cvm-csv", default="data/raw/cvm_fatos_relevantes.csv")
    parser.add_argument("--out", default="data/processed/conjunto_rotulagem.csv")
    args = parser.parse_args()

    if os.path.exists(args.out):
        print(f"[rotulagem] {args.out} JÁ EXISTE — não vou sobrescrever.")
        print("O conjunto é congelado: regerar depois de ver resultado invalida a medição.")
        print("Se quiser mesmo recomeçar, apague o arquivo à mão, conscientemente.")
        return

    conjunto = montar(args.n_por_ticker, args.cvm_csv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    conjunto.to_csv(args.out, index=False, encoding="utf-8-sig")  # BOM: Excel abre com acento certo

    caminho_instrucoes = args.out.replace(".csv", "_INSTRUCOES.txt")
    with open(caminho_instrucoes, "w", encoding="utf-8") as f:
        f.write(INSTRUCOES)

    h = hashlib.sha256(conjunto[["id", "ticker", "texto"]].to_csv(index=False).encode()).hexdigest()
    print(f"\n[rotulagem] {len(conjunto)} textos -> {args.out}")
    print(f"[rotulagem] instruções -> {caminho_instrucoes}")
    print(f"[rotulagem] sha256 do conjunto (sem os rótulos): {h[:32]}")
    print(f"[rotulagem] semente: {SEMENTE}")
    print("\nAbra no Excel, preencha 'relevante_humano' e 'sentimento_humano', salve como CSV,")
    print("e rode: python avaliar_classificador.py")


if __name__ == "__main__":
    main()
