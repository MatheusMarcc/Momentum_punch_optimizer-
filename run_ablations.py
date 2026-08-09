"""
Matriz de ablação REDUZIDA — cobre o critério de aceitação da seção 6.1 do
pré-relatório: "a tese é apoiada somente se sentimento+circuit breaker
superar o baseline (sem nenhum dos dois) de forma economicamente relevante".

Não é a matriz completa A0-A12 do documento (covariância regularizada/EWMA,
tilt multiplicativo vs aditivo como comparação, custo ligado/desligado como
eixo separado) — é o subconjunto que isola a contribuição marginal de cada
módulo com o que já está implementado:

  A0: baseline puro (sem sentimento, sem circuit breaker)
  A2: só sentimento (circuit breaker desligado)
  A3: só circuit breaker (sentimento neutro)
  A4: sentimento + circuit breaker (sistema completo)

Uso:
    python run_ablations.py
    python run_ablations.py --cost-bps 10
"""
from __future__ import annotations

import argparse

import pandas as pd

from momentum_punch import backtest, config
from run_real_backtest import load_prices, load_cdi, load_sentiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--mu-method", default="tilt_linear", choices=["tilt_linear", "black_litterman"])
    args = parser.parse_args()

    prices = load_prices()
    cdi = load_cdi().reindex(prices.index).ffill().fillna(0.0)

    try:
        sentiment_real = load_sentiment("data/processed/sentiment_scores.csv", prices.index)
    except FileNotFoundError:
        print("[run_ablations] sentiment_scores.csv não encontrado, usando neutro em todas as configs")
        sentiment_real = pd.DataFrame(0.0, index=prices.index, columns=config.TICKERS)

    try:
        stress_real = load_sentiment("data/processed/stress_index.csv", prices.index).iloc[:, 0]
    except FileNotFoundError:
        print("[run_ablations] stress_index.csv não encontrado, usando 0.0 (sempre Risk-On) em todas as configs")
        stress_real = pd.Series(0.0, index=prices.index)

    sentiment_neutro = pd.DataFrame(0.0, index=prices.index, columns=config.TICKERS)
    stress_zero = pd.Series(0.0, index=prices.index)

    configs = {
        "A0_baseline": (sentiment_neutro, stress_zero),
        "A2_so_sentimento": (sentiment_real, stress_zero),
        "A3_so_circuit_breaker": (sentiment_neutro, stress_real),
        "A4_completo": (sentiment_real, stress_real),
    }

    linhas = []
    for nome, (sent, stress) in configs.items():
        resultado = backtest.run_backtest(
            prices, cdi, sent, stress,
            benchmark="60_40", mu_method=args.mu_method, transaction_cost_bps=args.cost_bps,
        )
        m = resultado["metrics"]["Momentum Punch"]
        linhas.append({
            "config": nome,
            "retorno_total": m["Retorno total"],
            "cagr": m["CAGR"],
            "sharpe": m["Sharpe"],
            "max_drawdown": m["Max drawdown"],
            "custo_acumulado": m.get("Custo acumulado (giro)", "n/a"),
        })
        print(f"[run_ablations] {nome}: Sharpe={m['Sharpe']}, CAGR={m['CAGR']}, MaxDD={m['Max drawdown']}")

    df = pd.DataFrame(linhas)
    print("\n=== Matriz de ablação (reduzida: A0/A2/A3/A4) ===")
    print(df.to_string(index=False))

    df.to_csv("data/processed/ablation_results.csv", index=False)
    print("\nSalvo em data/processed/ablation_results.csv")

    # critério de aceitação da seção 6.1: A4 supera A0?
    sharpe_a0 = float(df[df["config"] == "A0_baseline"]["sharpe"].iloc[0])
    sharpe_a4 = float(df[df["config"] == "A4_completo"]["sharpe"].iloc[0])
    print(f"\n--- Critério de aceitação (seção 6.1) ---")
    print(f"Sharpe A0 (baseline): {sharpe_a0:.2f}")
    print(f"Sharpe A4 (completo): {sharpe_a4:.2f}")
    if sharpe_a4 > sharpe_a0:
        print(f"A4 SUPEROU A0 (+{sharpe_a4 - sharpe_a0:.2f} de Sharpe) — critério atendido nesse teste.")
    else:
        print(f"A4 NÃO superou A0 ({sharpe_a4 - sharpe_a0:+.2f} de Sharpe) — critério NÃO atendido, resultado negativo a reportar (não esconder).")


if __name__ == "__main__":
    main()
