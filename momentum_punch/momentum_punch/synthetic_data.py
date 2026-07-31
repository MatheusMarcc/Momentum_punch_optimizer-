"""Deterministic synthetic data for engineering tests only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_prices(
    n_days: int = 750,
    tickers: list[str] = config.TICKERS,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-07-29"), periods=n_days)
    annual_drift = {"ISUS11": 0.08, "GOVE11": 0.07, "REVE11": 0.09, "BOVA11": 0.08}
    annual_vol = {"ISUS11": 0.22, "GOVE11": 0.20, "REVE11": 0.30, "BOVA11": 0.24}
    data = {}
    for ticker in tickers:
        drift = annual_drift.get(ticker, 0.08) / config.TRADING_DAYS
        vol = annual_vol.get(ticker, 0.22) / np.sqrt(config.TRADING_DAYS)
        shocks = rng.normal(drift, vol, n_days)
        data[ticker] = 100.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(data, index=dates)


def generate_cdi_daily_return(
    n_days: int = 750,
    annual_rate: float = 0.11,
    dates: pd.DatetimeIndex | None = None,
) -> pd.Series:
    index = dates if dates is not None else pd.bdate_range(
        end=pd.Timestamp("2026-07-29"), periods=n_days
    )
    daily = (1.0 + annual_rate) ** (1.0 / config.TRADING_DAYS) - 1.0
    return pd.Series(daily, index=index, name=config.RISK_FREE)


def generate_sentiment_scores(
    dates: pd.DatetimeIndex,
    tickers: list[str] = config.TICKERS,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for ticker in tickers:
        values = np.zeros(len(dates))
        for i in range(1, len(dates)):
            values[i] = 0.85 * values[i - 1] + rng.normal(0.0, 0.20)
        data[ticker] = np.clip(values, -1.0, 1.0)
    return pd.DataFrame(data, index=dates).ewm(
        span=config.EMA_SPAN, adjust=False
    ).mean()


def generate_stress_index(dates: pd.DatetimeIndex, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = np.clip(rng.beta(2, 6, len(dates)), 0.0, 1.0)
    for start in rng.choice(len(dates), size=max(1, len(dates) // 180), replace=False):
        base[start : min(start + 8, len(dates))] = np.clip(
            base[start : min(start + 8, len(dates))] + 0.65, 0.0, 1.0
        )
    return pd.Series(base, index=dates, name="stress_index")
