"""
Roda validate_signal em matriz: todos os tickers de sentiment x vários
horizontes, pra avaliar COERÊNCIA (não pra caçar um p<0.05 isolado).

Por que isso importa: com N testes, ~N*5% vão dar "significativo" só por
acaso (correção de comparação múltipla). Um sinal real tende a aparecer de
forma consistente em horizontes vizinhos, com o MESMO sinal em treino e
teste — não isolado numa única combinação. Este script existe pra distinguir
"padrão coerente" (evidência de sinal real) de "acerto isolado" (ruído).

Uso:
    python validate_signal_matrix.py
    python validate_signal_matrix.py --horizons 1 3 5 10 --tickers ISUS11 GOVE11 REVE11 BOVA11
"""
from __future__ import annotations

import argparse

import pandas as pd

from momentum_punch import config
from validate_signal import (
    compute_forward_returns,
    train_test_split_temporal,
    information_coefficient,
    newey_west_regression,
    hit_rate,
)


def run_matrix(tickers: list[str], horizons: list[int], prices_csv: str, price_target: str | None,
               scores_csv: str = "data/processed/sentiment_scores.csv"):
    prices_long = pd.read_csv(prices_csv, parse_dates=["data"])
    prices_wide = prices_long.pivot(index="data", columns="ticker", values="close")
    scores = pd.read_csv(scores_csv, index_col=0, parse_dates=True)

    rows = []
    for ticker in tickers:
        if ticker not in scores.columns:
            print(f"[aviso] {ticker} não está em sentiment_scores.csv, pulando")
            continue
        sinal_completo = scores[ticker]

        # CORREÇÃO: por padrão, cada ticker testa contra o PRÓPRIO retorno
        # futuro (bug anterior fixava sempre BOVA11 pra todo mundo, o que
        # testava "sentiment do GOVE11 prevê o Ibovespa" em vez de "...prevê
        # o próprio GOVE11" — invalidava a leitura anterior desses resultados)
        alvo = price_target or ticker
        if alvo not in prices_wide.columns:
            print(f"[aviso] {alvo} não encontrado nos preços, pulando {ticker}")
            continue

        for horizon in horizons:
            retornos_futuros = compute_forward_returns(prices_wide[alvo], horizon)
            sinal, retornos = sinal_completo.align(retornos_futuros, join="inner")
            train_s, test_s = train_test_split_temporal(sinal.to_frame("s"))
            train_r, test_r = train_test_split_temporal(retornos.to_frame("r"))

            ic_train, p_train, n_train = information_coefficient(train_s["s"], train_r["r"])
            ic_test, p_test, n_test = information_coefficient(test_s["s"], test_r["r"])
            hr_train, _ = hit_rate(train_s["s"], train_r["r"])
            hr_test, _ = hit_rate(test_s["s"], test_r["r"])

            mesmo_sinal = (ic_train > 0) == (ic_test > 0) if not (pd.isna(ic_train) or pd.isna(ic_test)) else None

            rows.append({
                "ticker": ticker,
                "alvo_retorno": alvo,
                "horizonte": horizon,
                "n_treino": n_train,
                "n_teste": n_test,
                "IC_treino": round(ic_train, 3) if not pd.isna(ic_train) else None,
                "p_treino": round(p_train, 3) if not pd.isna(p_train) else None,
                "IC_teste": round(ic_test, 3) if not pd.isna(ic_test) else None,
                "p_teste": round(p_test, 3) if not pd.isna(p_test) else None,
                "hit_rate_teste": round(hr_test, 3) if not pd.isna(hr_test) else None,
                "mesmo_sinal_treino_teste": mesmo_sinal,
                "sig_teste_p<0.05": (p_test < 0.05) if not pd.isna(p_test) else False,
            })

    resultado = pd.DataFrame(rows)
    n_testes = len(resultado)
    alpha_bonferroni = 0.05 / n_testes if n_testes > 0 else 0.05

    print(f"\n=== Matriz de validação: {n_testes} combinações testadas ===")
    print(resultado.to_string(index=False))

    print(f"\n--- Leitura ---")
    print(f"Com {n_testes} testes, o limiar corrigido (Bonferroni) pra considerar um "
          f"resultado individual genuinamente significativo é p < {alpha_bonferroni:.4f}, "
          f"não p < 0.05 simples (esse limiar mais frouxo produziria ~{n_testes * 0.05:.1f} "
          f"'falso positivo' só por acaso, mesmo sem nenhum sinal real).")

    coerentes = resultado.dropna(subset=["mesmo_sinal_treino_teste"])
    n_mesmo_sinal = coerentes["mesmo_sinal_treino_teste"].sum()
    print(f"Sinal manteve a mesma direção (treino e teste) em {n_mesmo_sinal}/{len(coerentes)} combinações "
          f"— isso é a evidência de coerência mais relevante que o p-valor isolado de qualquer célula.")

    sobrevive_bonferroni = resultado[resultado["p_teste"] < alpha_bonferroni] if n_testes > 0 else resultado.iloc[0:0]
    if len(sobrevive_bonferroni) > 0:
        print(f"\nCombinações que sobrevivem à correção de Bonferroni:")
        print(sobrevive_bonferroni.to_string(index=False))
    else:
        print(f"\nNenhuma combinação sobrevive à correção de Bonferroni — consistente com "
              f"'sem sinal direcional robusto detectável', não com bug de código.")

    resultado.to_csv("data/processed/validate_signal_matrix.csv", index=False)
    print(f"\nSalvo em data/processed/validate_signal_matrix.csv")

    # ---- Calibração do tilt: SÓ COM TREINO ---------------------------------
    # A recomendação abaixo olha exclusivamente as colunas de TREINO. As de
    # teste estão na tabela acima como diagnóstico posterior, mas usá-las pra
    # escolher o tilt é o vazamento que invalidava a calibração anterior: os
    # kappas do config foram fixados olhando a amostra inteira, e depois o
    # backtest foi avaliado nessa mesma amostra.
    print(f"\n--- Calibração recomendada (olhando SÓ treino, corte {config.DATA_CORTE_TREINO_TESTE}) ---")
    print("Regra, nesta ordem:")
    print("  1. amostra suficiente no treino (n >= 100);")
    print("  2. direção do IC coerente entre horizontes;")
    print("  3. IC POSITIVO — a hipótese do projeto é direcional (sentimento bom")
    print("     -> retorno maior). IC negativo coerente NÃO vira tilt negativo:")
    print("     inverter o sinal depois de ver o dado é escolher a direção pelo")
    print("     resultado, e uma estratégia contrária a notícia precisaria de")
    print("     hipótese própria, declarada antes. IC negativo -> tilt 0 e");
    print("     'hipótese não sustentada para este ativo';")
    print("  4. magnitude relevante (IC >= 0.03).\n")

    for ticker in resultado["ticker"].unique():
        linhas_t = resultado[resultado["ticker"] == ticker].dropna(subset=["IC_treino"])
        if linhas_t.empty:
            print(f"  {ticker:8s} tilt=0.00   sem observação suficiente no treino")
            continue
        ics = linhas_t["IC_treino"]
        n_obs = int(linhas_t["n_treino"].max())
        coerente = (ics > 0).all() or (ics < 0).all()
        ic_medio = ics.mean()

        if n_obs < 100:
            veredito, tilt = f"amostra pequena no treino (n={n_obs})", 0.0
        elif not coerente:
            veredito, tilt = "direção do IC troca entre horizontes", 0.0
        elif ic_medio < 0:
            veredito, tilt = f"IC coerente mas NEGATIVO ({ic_medio:+.3f}) — contradiz a hipótese", 0.0
        elif ic_medio < 0.03:
            veredito, tilt = f"coerente e positivo, mas fraco (IC {ic_medio:+.3f})", 0.0
        else:
            tilt = round(min(0.30, ic_medio * 3), 2)
            veredito = f"coerente, positivo, IC médio {ic_medio:+.3f}, n={n_obs}"
        print(f"  {ticker:8s} tilt={tilt:.2f}   {veredito}")

    print("\nCopie os valores acima pro config.SENTIMENT_TILT_STRENGTH_POR_TICKER e")
    print("CONGELE antes de rodar o período de teste. Não reajuste depois de ver o teste.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["ISUS11", "GOVE11", "REVE11", "BOVA11"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--prices-csv", default="data/raw/etf_prices.csv")
    parser.add_argument("--price-target", default=None, help="fixa um único ativo-alvo pra todos (default: cada ticker testa contra o próprio retorno)")
    parser.add_argument("--scores", default="data/processed/sentiment_scores.csv")
    args = parser.parse_args()

    run_matrix(args.tickers, args.horizons, args.prices_csv, args.price_target, args.scores)


if __name__ == "__main__":
    main()