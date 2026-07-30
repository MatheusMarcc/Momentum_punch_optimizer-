"""
Gera preços e sentiment scores sintéticos para testar o pipeline ponta a ponta
sem depender de dados reais de mercado / API de LLM. Troque por dados reais
(cotações da B3, scores gerados por sentiment.score_history) quando for rodar
o backtest de verdade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_prices(
    n_days: int = 750,
    tickers: list[str] = config.TICKERS,
    seed: int = 42,
) -> pd.DataFrame:
    """Movimento Browniano Geométrico simples por ativo, com drifts/vols distintos."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)

    annual_drift = {"ISUS11": 0.12, "GOVE11": 0.10, "REVE11": 0.15, "BOVA11": 0.11}
    annual_vol = {"ISUS11": 0.22, "GOVE11": 0.20, "REVE11": 0.30, "BOVA11": 0.24}

    prices = {}
    for t in tickers:
        mu_d = annual_drift.get(t, 0.10) / 252
        sigma_d = annual_vol.get(t, 0.22) / np.sqrt(252)
        rets = rng.normal(mu_d, sigma_d, n_days)
        prices[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(prices, index=dates)


def generate_cdi_daily_return(n_days: int = 750, annual_rate: float = 0.11) -> pd.Series:
    """CDI ~constante (simplificado): taxa anual convertida em retorno diário fixo."""
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    daily = (1 + annual_rate) ** (1 / 252) - 1
    return pd.Series(daily, index=dates, name=config.RISK_FREE)


def generate_sentiment_scores(
    dates: pd.DatetimeIndex,
    tickers: list[str] = config.TICKERS,
    seed: int = 7,
) -> pd.DataFrame:
    """Sentiment score sintético: ruído + autocorrelação (AR1), já em escala [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    data = {}
    for t in tickers:
        raw = np.zeros(n)
        raw[0] = rng.normal(0, 0.3)
        for i in range(1, n):
            raw[i] = 0.85 * raw[i - 1] + rng.normal(0, 0.25)
        data[t] = np.clip(raw, config.SENTIMENT_MIN, config.SENTIMENT_MAX)
    df = pd.DataFrame(data, index=dates)
    return df.ewm(span=config.EMA_SPAN, adjust=False).mean()


def generate_stress_index(dates: pd.DatetimeIndex, seed: int = 99) -> pd.Series:
    """Índice de estresse sintético com alguns picos de crise aleatórios."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    base = np.clip(rng.beta(2, 6, n), 0, 1)  # normalmente baixo
    n_shocks = max(1, n // 150)
    shock_idx = rng.choice(n, size=n_shocks, replace=False)
    for idx in shock_idx:
        span = slice(idx, min(idx + 10, n))
        base[span] = np.clip(base[span] + rng.uniform(0.5, 0.9), 0, 1)
    return pd.Series(base, index=dates, name="stress_index")
