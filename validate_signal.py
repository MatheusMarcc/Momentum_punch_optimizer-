"""
Validação ESTATÍSTICA do sinal (sentiment score / stress index), separada da
performance do portfólio. Responde: "esse sinal realmente prevê retorno
futuro?" — não "o portfólio foi bom?" (isso é o backtest, ver run_real_backtest.py).

Metodologia:
  1. Split treino/teste TEMPORAL (não aleatório — é série temporal, embaralhar
     vazaria informação do futuro pro passado).
  2. Information Coefficient (IC): correlação (Spearman, robusta a outlier)
     entre sinal(t) e retorno futuro(t+h), calculada separadamente em treino
     e teste. IC > 0 e estável entre treino/teste = sinal com poder preditivo
     genuíno, não overfitting.
  3. Regressão com erro-padrão Newey-West (HAC): corrige a autocorrelação
     natural de séries financeiras, que infla a significância se você usar
     OLS padrão. Reporta t-stat e p-valor de verdade.
  4. Hit rate: % dos dias em que o SINAL do score bateu com o SINAL do
     retorno futuro (ex: sentiment positivo -> retorno positivo). 50% = sinal
     não tem poder preditivo nenhum (é uma moeda).

IMPORTANTE sobre tamanho de amostra: com poucos dias de sentiment real (RSS
ainda raso), os resultados desta seção não vão ser estatisticamente
significativos — isso não é erro do código, é conclusão honesta: sinal com
n pequeno não permite afirmar nada com confiança. O stress_index via Focus
(2 anos) tem amostra grande o suficiente pra um teste de verdade.

Uso:
    python validate_signal.py --signal stress_index --horizon 5
    python validate_signal.py --signal sentiment --ticker BOVA11 --horizon 1
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


def compute_forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    """Retorno futuro acumulado de t até t+horizon (o que o sinal em t deveria prever)."""
    return prices.shift(-horizon) / prices - 1


def train_test_split_temporal(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split por TEMPO, não aleatório — os primeiros train_frac% viram treino,
    o resto (mais recente) vira teste. Embaralhar aleatoriamente aqui seria
    vazamento: o modelo "veria" padrões do futuro durante o treino."""
    corte = int(len(df) * train_frac)
    return df.iloc[:corte], df.iloc[corte:]


def information_coefficient(sinal: pd.Series, retorno_futuro: pd.Series) -> tuple[float, float, int]:
    """IC = correlação de Spearman entre sinal e retorno futuro. Retorna (IC, p-valor, n)."""
    df = pd.DataFrame({"sinal": sinal, "retorno": retorno_futuro}).dropna()
    if len(df) < 5:
        return float("nan"), float("nan"), len(df)
    ic, p = spearmanr(df["sinal"], df["retorno"])
    return ic, p, len(df)


def newey_west_regression(sinal: pd.Series, retorno_futuro: pd.Series, lags: int = 5):
    """Regressão retorno_futuro ~ sinal com erro-padrão HAC (Newey-West), que
    corrige a autocorrelação natural de retornos financeiros sobrepostos
    (horizontes >1 dia geram observações não-independentes — OLS comum
    subestima o erro-padrão e infla a significância)."""
    df = pd.DataFrame({"sinal": sinal, "retorno": retorno_futuro}).dropna()
    if len(df) < 10:
        return None
    X = sm.add_constant(df["sinal"])
    modelo = sm.OLS(df["retorno"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return modelo


def hit_rate(sinal: pd.Series, retorno_futuro: pd.Series) -> tuple[float, int]:
    """% de vezes que sign(sinal) == sign(retorno futuro). 50% = sem poder preditivo."""
    df = pd.DataFrame({"sinal": sinal, "retorno": retorno_futuro}).dropna()
    df = df[df["sinal"] != 0]  # sinal neutro não conta como "acerto" nem "erro"
    if len(df) == 0:
        return float("nan"), 0
    acertos = (np.sign(df["sinal"]) == np.sign(df["retorno"])).sum()
    return acertos / len(df), len(df)


def validate(sinal: pd.Series, retornos_futuros: pd.Series, nome_sinal: str, horizon: int):
    sinal, retornos_futuros = sinal.align(retornos_futuros, join="inner")
    train_sinal, test_sinal = train_test_split_temporal(sinal.to_frame("s"))
    train_ret, test_ret = train_test_split_temporal(retornos_futuros.to_frame("r"))

    print(f"\n=== Validação estatística: {nome_sinal} -> retorno futuro de {horizon} dia(s) ===")
    print(f"Amostra total: {len(sinal.dropna())} dias | Treino: {len(train_sinal)} | Teste: {len(test_sinal)}\n")

    for label, s, r in [("TREINO", train_sinal["s"], train_ret["r"]), ("TESTE", test_sinal["s"], test_ret["r"])]:
        ic, p_ic, n = information_coefficient(s, r)
        hr, n_hr = hit_rate(s, r)
        print(f"[{label}] n={n}")
        if np.isnan(ic):
            print("  Amostra pequena demais pra qualquer conclusão estatística.")
            continue
        print(f"  Information Coefficient (Spearman): {ic:+.3f} (p={p_ic:.3f}){' *** significativo (p<0.05)' if p_ic < 0.05 else ' — não significativo'}")
        print(f"  Hit rate: {hr:.1%} (n={n_hr}, 50% = sem poder preditivo)")

        modelo = newey_west_regression(s, r)
        if modelo is not None:
            coef = modelo.params.get("sinal", float("nan"))
            tstat = modelo.tvalues.get("sinal", float("nan"))
            pval = modelo.pvalues.get("sinal", float("nan"))
            print(f"  Regressão (Newey-West): coef={coef:+.4f}, t-stat={tstat:+.2f}, p={pval:.3f}{' *** significativo' if pval < 0.05 else ' — não significativo'}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=["stress_index", "sentiment"], required=True)
    parser.add_argument("--ticker", default="BOVA11", help="usado quando --signal=sentiment (ignorado pra stress_index, que é agregado)")
    parser.add_argument("--price-target", default="BOVA11", help="qual ETF usar como alvo do retorno futuro")
    parser.add_argument("--horizon", type=int, default=5, help="horizonte do retorno futuro em dias úteis")
    parser.add_argument("--prices-csv", default="data/raw/etf_prices.csv")
    args = parser.parse_args()

    prices_long = pd.read_csv(args.prices_csv, parse_dates=["data"])
    prices_wide = prices_long.pivot(index="data", columns="ticker", values="close")
    if args.price_target not in prices_wide.columns:
        raise ValueError(f"{args.price_target} não encontrado em {args.prices_csv}")

    retornos_futuros = compute_forward_returns(prices_wide[args.price_target], args.horizon)

    if args.signal == "stress_index":
        sinal = pd.read_csv("data/processed/stress_index.csv", index_col=0, parse_dates=True).iloc[:, 0]
        # estresse ALTO deveria PREVER retorno NEGATIVO — inverto o sinal aqui
        # só pra hit_rate/IC lerem na direção intuitiva (estresse-> queda = "acerto")
        sinal = -sinal
        nome = "Stress Index (Focus, invertido: estresse alto -> espera queda)"
    else:
        scores = pd.read_csv("data/processed/sentiment_scores.csv", index_col=0, parse_dates=True)
        if args.ticker not in scores.columns:
            raise ValueError(f"{args.ticker} não encontrado em sentiment_scores.csv")
        sinal = scores[args.ticker]
        nome = f"Sentiment Score ({args.ticker})"

    validate(sinal, retornos_futuros, nome, args.horizon)


if __name__ == "__main__":
    main()
