"""
Roda o backtest com dado REAL (preço de ETF via yfinance, CDI via Bacen SGS,
sentiment via Ollama), no lugar dos dados sintéticos do main.py.

Pré-requisitos (rode antes):
    python fetch_prices_yfinance.py
    python collect_data.py --only bacen_sgs
    python collect_data.py --only rss_news
    python build_sentiment_dataset.py

Uso:
    python run_real_backtest.py
"""
from __future__ import annotations

import pandas as pd

from momentum_punch import backtest, config


def load_prices(path: str = "data/raw/etf_prices.csv") -> pd.DataFrame:
    """Lê o CSV longo (data, ticker, close) do fetch_prices_yfinance.py e pivota
    pra formato wide (index=data, colunas=tickers), que é o que backtest.py espera."""
    df = pd.read_csv(path, parse_dates=["data"])
    wide = df.pivot(index="data", columns="ticker", values="close")
    faltando = set(config.TICKERS) - set(wide.columns)
    if faltando:
        print(f"[run_real_backtest] AVISO: sem preço pra {faltando} — não vai dar pra otimizar esses ativos.")
    return wide[[t for t in config.TICKERS if t in wide.columns]].dropna(how="all")


def load_cdi(path: str = "data/raw/bacen_sgs.csv") -> pd.Series:
    """Converte a taxa diária do CDI (% ao dia, coluna cdi_diario) em retorno decimal."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "cdi_diario" not in df.columns:
        raise ValueError(
            "Coluna 'cdi_diario' não encontrada em bacen_sgs.csv — rode de novo "
            "python collect_data.py --only bacen_sgs (precisa da versão atualizada do coletor)."
        )
    return (df["cdi_diario"] / 100).rename(config.RISK_FREE)


def load_sentiment(path: str, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Carrega o sentiment_scores.csv (ou stress_index.csv) e alinha com o índice
    de datas dos preços via forward-fill: o último score conhecido continua
    valendo até chegar um novo (é assim que, na prática, "o mercado carrega a
    última leitura de sentimento" entre um dia de coleta e o outro).
    Dias antes do primeiro score disponível ficam em 0.0 (neutro) — não dá pra
    inventar sentimento retroativo.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.reindex(price_index).ffill().fillna(0.0)
    return df


def main():
    prices = load_prices()
    print(f"[run_real_backtest] Preços: {len(prices)} dias, tickers: {list(prices.columns)}")

    cdi = load_cdi().reindex(prices.index).ffill().fillna(0.0)

    try:
        sentiment_scores = load_sentiment("data/processed/sentiment_scores.csv", prices.index)
    except FileNotFoundError:
        print("[run_real_backtest] sentiment_scores.csv não encontrado — rode build_sentiment_dataset.py antes. Usando sentimento neutro (0.0) por enquanto.")
        sentiment_scores = pd.DataFrame(0.0, index=prices.index, columns=config.TICKERS)

    try:
        stress_index = load_sentiment("data/processed/stress_index.csv", prices.index).iloc[:, 0]
    except FileNotFoundError:
        print("[run_real_backtest] stress_index.csv não encontrado — circuit breaker fica sempre em Risk-On por enquanto (stress=0.0).")
        stress_index = pd.Series(0.0, index=prices.index)

    result = backtest.run_backtest(
        prices=prices,
        cdi_daily_return=cdi,
        sentiment_scores=sentiment_scores,
        stress_index=stress_index,
        benchmark="60_40",
    )

    print("\n=== Métricas de performance (DADO REAL) ===")
    for strategy, metrics in result["metrics"].items():
        print(f"\n{strategy}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    n_riskoff = sum(1 for w in result["weights_history"].values() if w.get("_mode") == "RISK-OFF")
    print(f"\nRebalanceamentos em modo RISK-OFF: {n_riskoff} / {len(result['weights_history'])}")

    last_date = max(result["weights_history"])
    print(f"\nÚltima alocação ({last_date.date()}):")
    for k, v in result["weights_history"][last_date].items():
        if not k.startswith("_"):
            print(f"  {k}: {v:.1%}")


if __name__ == "__main__":
    main()
