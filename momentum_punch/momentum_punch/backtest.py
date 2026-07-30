"""
Bloco de simulação/comparação (slides 09-10 do deck).

Walk-forward: em cada data de rebalanceamento, usa apenas dados até aquele dia
(sentiment + covariância trailing) para decidir os pesos, depois acumula o
retorno realizado até o próximo rebalanceamento. Isso evita look-ahead bias.

Benchmark: 100% BOVA11, ou o mix estático 60% BOVA11 / 40% CDI (escolha via
parâmetro `benchmark`).
"""
from __future__ import annotations

import pandas as pd

from . import config, optimizer, risk_overlay


def _rebalance_dates(index: pd.DatetimeIndex, freq: str = config.REBALANCE_FREQ) -> pd.DatetimeIndex:
    grouped = pd.Series(index=index, data=index).resample(freq).first()
    return pd.DatetimeIndex(grouped.dropna().values)


def run_backtest(
    prices: pd.DataFrame,
    cdi_daily_return: pd.Series,
    sentiment_scores: pd.DataFrame,
    stress_index: pd.Series,
    benchmark: str = "bova11",  # "bova11" ou "60_40"
) -> dict:
    """
    prices: DataFrame (index=data, colunas=config.TICKERS) com preços dos ETFs.
    cdi_daily_return: Series com retorno diário do CDI.
    sentiment_scores: DataFrame (index=data, colunas=config.TICKERS), já em EMA.
    stress_index: Series (index=data) com o índice de estresse (0 a 1).

    Retorna um dict com: equity_curve, benchmark_curve, weights_history, metrics.
    """
    returns = prices.pct_change().dropna()
    # warm-up: precisa de COV_LOOKBACK_DAYS de retornos antes do primeiro rebalanceamento
    tradeable_index = returns.index[config.COV_LOOKBACK_DAYS:]
    rebal_dates = _rebalance_dates(tradeable_index)

    portfolio_value = 1.0
    benchmark_value = 1.0
    equity_curve = {}
    benchmark_curve = {}
    weights_history = {}

    current_weights = None
    for i, date in enumerate(tradeable_index):
        if date in rebal_dates or current_weights is None:
            mu, sigma = optimizer.historical_mu_sigma(returns.loc[:date])
            today_sentiment = sentiment_scores.loc[:date].iloc[-1] if not sentiment_scores.loc[:date].empty else pd.Series(0.0, index=mu.index)
            mu_adj = optimizer.tilt_mu_by_sentiment(mu, today_sentiment)
            etf_weights = optimizer.optimize_weights(mu_adj, sigma)

            today_stress = stress_index.loc[:date].iloc[-1] if not stress_index.loc[:date].empty else 0.0
            final_weights = risk_overlay.apply_circuit_breaker(etf_weights, today_stress)
            current_weights = final_weights
            weights_history[date] = final_weights

        # retorno do dia com os pesos vigentes
        etf_ret = sum(
            current_weights.get(t, 0.0) * returns.loc[date, t] for t in config.TICKERS
        )
        cdi_ret = current_weights.get(config.RISK_FREE, 0.0) * cdi_daily_return.get(date, 0.0)
        day_return = etf_ret + cdi_ret
        portfolio_value *= (1 + day_return)
        equity_curve[date] = portfolio_value

        # benchmark
        if benchmark == "bova11":
            bench_ret = returns.loc[date, "BOVA11"]
        elif benchmark == "60_40":
            bench_ret = 0.6 * returns.loc[date, "BOVA11"] + 0.4 * cdi_daily_return.get(date, 0.0)
        else:
            raise ValueError("benchmark deve ser 'bova11' ou '60_40'")
        benchmark_value *= (1 + bench_ret)
        benchmark_curve[date] = benchmark_value

    equity_curve = pd.Series(equity_curve, name="Momentum Punch")
    benchmark_curve = pd.Series(benchmark_curve, name="Benchmark")

    metrics = {
        "Momentum Punch": _performance_metrics(equity_curve),
        "Benchmark": _performance_metrics(benchmark_curve),
    }

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "weights_history": weights_history,
        "metrics": metrics,
    }


def _performance_metrics(equity_curve: pd.Series) -> dict:
    daily_ret = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else float("nan")
    vol_annual = daily_ret.std() * (252 ** 0.5)
    sharpe = (daily_ret.mean() * 252) / vol_annual if vol_annual > 0 else float("nan")
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "Retorno total": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Volatilidade anual": f"{vol_annual:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max drawdown": f"{max_drawdown:.2%}",
    }
