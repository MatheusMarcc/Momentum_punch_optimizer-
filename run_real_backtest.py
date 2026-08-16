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

import argparse

import pandas as pd

from momentum_punch import backtest, config, manifesto


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


def load_sentiment(path: str, price_index: pd.DatetimeIndex, max_decay_days: int = 21) -> pd.DataFrame:
    """
    Carrega o sentiment_scores.csv e alinha com o índice de preços via
    forward-fill, MAS com decaimento por dia de calendário e corte —
    diferente de um ffill puro, que seguraria o último score pra sempre.

    Motivo: o EMA em sentiment.py decai por POSIÇÃO na tabela, não por
    dia de calendário — com fonte esparsa (ex: CVM, meses entre eventos),
    isso faz uma notícia antiga "durar" indevidamente. Aqui, cada dia sem
    observação nova decai o score linearmente até zerar em max_decay_days,
    representando que sentimento velho perde relevância com o tempo real.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.reindex(df.index.union(price_index)).sort_index()

    decaido = df.copy()
    for col in df.columns:
        dias_desde_obs = 0
        ultimo_valor = 0.0
        valores = []
        obs_real = df[col].notna()
        for data in df.index:
            if obs_real.get(data, False):
                ultimo_valor = df.loc[data, col]
                dias_desde_obs = 0
            else:
                dias_desde_obs += 1
            fator_decaimento = max(0.0, 1 - dias_desde_obs / max_decay_days)
            valores.append(ultimo_valor * fator_decaimento)
        decaido[col] = valores

    return decaido.reindex(price_index).fillna(0.0)


def load_benchmark_prices(path: str = "data/raw/benchmark_prices.csv") -> pd.DataFrame | None:
    """Preços de ativos usados SÓ como benchmark (IVVB11), nunca no universo
    investível. Ausência não é erro — o backtest roda sem o comparador 40/40/20."""
    import os
    if not os.path.exists(path):
        print(f"[run_real_backtest] {path} não existe — benchmark 40/40/20 sai achatado no CDI")
        return None
    df = pd.read_csv(path, parse_dates=["data"])
    return df.pivot(index="data", columns="ticker", values="close")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu-method", default="tilt_linear", choices=["tilt_linear", "black_litterman"])
    parser.add_argument("--cost-bps", type=float, default=10.0, help="custo de transação por giro, em bps (ex: 10 = 0.10%%). 0 = sem custo")
    parser.add_argument("--scores", default="data/processed/sentiment_scores.csv", help="CSV de sentiment a usar")
    parser.add_argument("--periodo", default="tudo", choices=["tudo", "treino", "teste"],
                        help="recorte de avaliação; o backtest sempre roda contínuo, só as métricas são recortadas")
    args = parser.parse_args()

    prices = load_prices()
    print(f"[run_real_backtest] Preços: {len(prices)} dias ({prices.index.min().date()} a {prices.index.max().date()}), tickers: {list(prices.columns)}")
    print(f"[run_real_backtest] Método: {args.mu_method} | custo: {args.cost_bps:.0f} bps | sinal: {args.scores}")

    cdi = load_cdi().reindex(prices.index).ffill().fillna(0.0)

    try:
        sentiment_scores = load_sentiment(args.scores, prices.index)
    except FileNotFoundError:
        print(f"[run_real_backtest] {args.scores} não encontrado — rode build_sentiment_dataset.py antes. Usando sentimento neutro (0.0).")
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
        mu_method=args.mu_method,
        transaction_cost_bps=args.cost_bps,
        benchmark_prices=load_benchmark_prices(),
    )

    corte = config.DATA_CORTE_TREINO_TESTE
    inicio, fim = {"tudo": (None, None), "treino": (None, corte), "teste": (corte, None)}[args.periodo]
    metrics = backtest.metricas_no_periodo(result, cdi, inicio, fim)

    print(f"\n=== Métricas (DADO REAL) — período: {args.periodo}" + (f", corte {corte}" if args.periodo != "tudo" else "") + " ===")
    for strategy, met in metrics.items():
        print(f"\n{strategy}:")
        for k, v in met.items():
            print(f"  {k}: {v}")

    # salva pro dashboard (Streamlit) consumir — antes só ficava impresso no terminal
    import os
    os.makedirs("data/processed", exist_ok=True)
    curvas = pd.DataFrame({"Momentum Punch": result["equity_curve"], "Benchmark": result["benchmark_curve"]})
    for nome, curva in result["benchmarks"].items():
        curvas[backtest.BENCHMARKS[nome]] = curva
    curvas.to_csv("data/processed/equity_curves.csv")
    print(f"\n[run_real_backtest] Equity curves salvas em data/processed/equity_curves.csv")

    manifesto.salvar(
        manifesto.gerar(
            experimento=f"backtest_real_{args.periodo}",
            categoria="VERIFICADO",
            entradas={
                "precos": "data/raw/etf_prices.csv",
                "cdi": "data/raw/bacen_sgs.csv",
                "sentiment": args.scores,
                "stress": "data/processed/stress_index.csv",
                "precos_benchmark": "data/raw/benchmark_prices.csv",
            },
            parametros_execucao={
                "periodo": args.periodo,
                "mu_method": args.mu_method,
                "custo_bps": args.cost_bps,
                "janela_precos": f"{prices.index.min().date()} a {prices.index.max().date()}",
                "dias_avaliados": len(result["equity_curve"].loc[inicio:fim]),
            },
            metricas=metrics,
            saidas=["data/processed/equity_curves.csv"],
        ),
        f"data/processed/manifesto_backtest_{args.periodo}.json",
    )

    last_date = max(result["weights_history"])
    print(f"\nÚltima alocação ({last_date.date()}):")
    for k, v in result["weights_history"][last_date].items():
        if not k.startswith("_"):
            print(f"  {k}: {v:.1%}")


if __name__ == "__main__":
    main()
