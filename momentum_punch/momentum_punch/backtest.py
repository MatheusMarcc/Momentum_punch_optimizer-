"""Leakage-resistant walk-forward backtest with costs and audit trails."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from . import config, optimizer, risk_overlay


@dataclass(frozen=True)
class BacktestAssumptions:
    rebalance_freq: str = config.REBALANCE_FREQ
    transaction_cost_bps: float = config.TRANSACTION_COST_BPS
    benchmark: str = "60_40"
    sentiment_mode: str = config.SENTIMENT_ALPHA_MODE
    covariance_method: str = "sample"


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if index.empty:
        return pd.DatetimeIndex([])
    grouped = pd.Series(index=index, data=index).resample(freq).last()
    return pd.DatetimeIndex(grouped.dropna().values)


def _investable(weights: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in weights.items() if not k.startswith("_")}


def _turnover(old: dict[str, float], new: dict[str, float]) -> float:
    assets = set(old) | set(new)
    return 0.5 * sum(abs(new.get(a, 0.0) - old.get(a, 0.0)) for a in assets)


def _latest_row_asof(frame: pd.DataFrame, date: pd.Timestamp, columns: pd.Index) -> pd.Series:
    available = frame.loc[:date]
    if available.empty:
        return pd.Series(0.0, index=columns, dtype=float)
    return available.iloc[-1].reindex(columns).fillna(0.0).astype(float)


def _latest_value_asof(series: pd.Series, date: pd.Timestamp, default: float = 0.0) -> float:
    available = series.loc[:date]
    return default if available.empty else float(available.iloc[-1])


def _align_inputs(
    prices: pd.DataFrame,
    cdi_daily_return: pd.Series,
    sentiment_scores: pd.DataFrame,
    stress_index: pd.Series,
):
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    missing = [t for t in config.TICKERS if t not in prices.columns]
    if missing:
        raise ValueError(f"missing price columns: {missing}")

    prices = prices.sort_index()[config.TICKERS].astype(float)
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.dropna(subset=config.TICKERS, how="any")

    cdi = cdi_daily_return.sort_index().reindex(returns.index).fillna(0.0).astype(float)
    sentiment = sentiment_scores.sort_index().reindex(columns=config.TICKERS)
    stress = stress_index.sort_index().astype(float)
    return returns, cdi, sentiment, stress


def run_backtest(
    prices: pd.DataFrame,
    cdi_daily_return: pd.Series,
    sentiment_scores: pd.DataFrame,
    stress_index: pd.Series,
    benchmark: str = "60_40",
    transaction_cost_bps: float = config.TRANSACTION_COST_BPS,
    sentiment_mode: str = config.SENTIMENT_ALPHA_MODE,
    covariance_method: str = "sample",
    rebalance_freq: str = config.REBALANCE_FREQ,
) -> dict:
    """Run a close-to-next-session backtest.

    A signal computed with information available through date *t* is executed at
    the next trading date. Therefore, the return on *t* can never be earned by a
    portfolio decided on *t*.
    """
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    returns, cdi, sentiment, stress = _align_inputs(
        prices, cdi_daily_return, sentiment_scores, stress_index
    )
    if len(returns) <= config.COV_LOOKBACK_DAYS + 2:
        raise ValueError("not enough observations for the configured lookback")

    tradeable = returns.index[config.COV_LOOKBACK_DAYS:]
    signal_dates = set(_rebalance_dates(tradeable, rebalance_freq))

    current = {t: 0.0 for t in config.TICKERS}
    current[config.RISK_FREE] = 1.0
    pending: dict[str, float] | None = None
    pending_signal_date: pd.Timestamp | None = None

    portfolio_value = 1.0
    benchmark_value = 1.0
    portfolio_returns: dict[pd.Timestamp, float] = {}
    benchmark_returns: dict[pd.Timestamp, float] = {}
    equity_curve: dict[pd.Timestamp, float] = {}
    benchmark_curve: dict[pd.Timestamp, float] = {}
    weights_history: dict[pd.Timestamp, dict[str, float]] = {}
    signal_history: dict[pd.Timestamp, dict] = {}
    turnover_history: dict[pd.Timestamp, float] = {}
    cost_history: dict[pd.Timestamp, float] = {}

    for date in tradeable:
        # Execute yesterday's target before earning today's return.
        execution_cost = 0.0
        if pending is not None:
            new_investable = _investable(pending)
            old_investable = _investable(current)
            turnover = _turnover(old_investable, new_investable)
            execution_cost = turnover * transaction_cost_bps / 10_000.0
            portfolio_value *= 1.0 - execution_cost
            current = pending
            weights_history[date] = {
                **pending,
                "_signal_date": pending_signal_date.isoformat() if pending_signal_date is not None else None,
                "_execution_date": date.isoformat(),
            }
            turnover_history[date] = turnover
            cost_history[date] = execution_cost
            pending = None
            pending_signal_date = None

        etf_return = sum(current.get(t, 0.0) * float(returns.loc[date, t]) for t in config.TICKERS)
        cdi_return = current.get(config.RISK_FREE, 0.0) * float(cdi.loc[date])
        gross_day_return = etf_return + cdi_return
        portfolio_value *= 1.0 + gross_day_return
        portfolio_returns[date] = (1.0 - execution_cost) * (1.0 + gross_day_return) - 1.0
        equity_curve[date] = portfolio_value

        if benchmark == "bova11":
            bench_ret = float(returns.loc[date, "BOVA11"])
        elif benchmark == "60_40":
            bench_ret = 0.6 * float(returns.loc[date, "BOVA11"]) + 0.4 * float(cdi.loc[date])
        elif benchmark == "equal_weight":
            bench_ret = float(returns.loc[date, config.TICKERS].mean())
        else:
            raise ValueError("benchmark must be 'bova11', '60_40', or 'equal_weight'")
        benchmark_value *= 1.0 + bench_ret
        benchmark_returns[date] = bench_ret
        benchmark_curve[date] = benchmark_value

        # Compute a target at the close of the signal date for next-session execution.
        if date in signal_dates:
            mu, sigma = optimizer.historical_mu_sigma(
                returns.loc[:date],
                covariance_method=covariance_method,
            )
            today_sentiment = _latest_row_asof(sentiment, date, mu.index)
            mu_adjusted = optimizer.adjust_mu_by_sentiment(
                mu, today_sentiment, mode=sentiment_mode
            )
            etf_weights, diagnostics = optimizer.optimize_weights(
                mu_adjusted, sigma, return_diagnostics=True
            )
            today_stress = _latest_value_asof(stress, date, default=0.0)
            target = risk_overlay.apply_circuit_breaker(etf_weights, today_stress)
            pending = target
            pending_signal_date = date
            signal_history[date] = {
                "target_weights": target,
                "sentiment": today_sentiment.to_dict(),
                "stress_index": today_stress,
                "optimizer_status": diagnostics.status,
                "optimizer_fallback": diagnostics.used_fallback,
                "sentiment_mode": sentiment_mode,
                "covariance_method": covariance_method,
            }

    equity = pd.Series(equity_curve, name="Momentum Punch", dtype=float)
    benchmark_equity = pd.Series(benchmark_curve, name="Benchmark", dtype=float)
    port_ret = pd.Series(portfolio_returns, name="portfolio_return", dtype=float)
    bench_ret = pd.Series(benchmark_returns, name="benchmark_return", dtype=float)
    turnover = pd.Series(turnover_history, name="turnover", dtype=float)
    costs = pd.Series(cost_history, name="transaction_cost", dtype=float)

    metrics = {
        "Momentum Punch": performance_metrics(
            port_ret, risk_free_daily=cdi.reindex(port_ret.index).fillna(0.0),
            turnover=turnover, costs=costs
        ),
        "Benchmark": performance_metrics(
            bench_ret, risk_free_daily=cdi.reindex(bench_ret.index).fillna(0.0)
        ),
    }

    return {
        "equity_curve": equity,
        "benchmark_curve": benchmark_equity,
        "portfolio_returns": port_ret,
        "benchmark_returns": bench_ret,
        "weights_history": weights_history,
        "signal_history": signal_history,
        "turnover_history": turnover,
        "cost_history": costs,
        "metrics": metrics,
        "assumptions": BacktestAssumptions(
            rebalance_freq=rebalance_freq,
            transaction_cost_bps=transaction_cost_bps,
            benchmark=benchmark,
            sentiment_mode=sentiment_mode,
            covariance_method=covariance_method,
        ).__dict__,
    }


def performance_metrics(
    daily_returns: pd.Series,
    risk_free_daily: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> dict[str, float]:
    r = daily_returns.dropna().astype(float)
    if r.empty:
        raise ValueError("daily_returns is empty")

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    n_years = len(r) / config.TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else math.nan
    volatility = float(r.std(ddof=1) * math.sqrt(config.TRADING_DAYS))

    if risk_free_daily is None:
        excess = r
    else:
        excess = r - risk_free_daily.reindex(r.index).fillna(0.0)
    sharpe = (
        float(excess.mean() * config.TRADING_DAYS / volatility)
        if volatility > 0 else math.nan
    )

    downside = r[r < 0].std(ddof=1) * math.sqrt(config.TRADING_DAYS)
    sortino = float(r.mean() * config.TRADING_DAYS / downside) if downside and downside > 0 else math.nan

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else math.nan

    monthly = (1.0 + r).resample("ME").prod() - 1.0
    underwater_days = int((drawdown < 0).sum())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe_excess_over_cdi": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "best_month": float(monthly.max()) if not monthly.empty else math.nan,
        "worst_month": float(monthly.min()) if not monthly.empty else math.nan,
        "underwater_days": float(underwater_days),
        "turnover_total": float(turnover.sum()) if turnover is not None and not turnover.empty else 0.0,
        "transaction_cost_total": float(costs.sum()) if costs is not None and not costs.empty else 0.0,
    }


def format_metrics(metrics: dict[str, float]) -> dict[str, str]:
    percentage_keys = {
        "total_return", "cagr", "annual_volatility", "max_drawdown",
        "best_month", "worst_month", "transaction_cost_total"
    }
    formatted = {}
    for key, value in metrics.items():
        if key in percentage_keys:
            formatted[key] = f"{value:.2%}"
        elif key == "underwater_days":
            formatted[key] = f"{int(value)}"
        else:
            formatted[key] = f"{value:.3f}"
    return formatted
