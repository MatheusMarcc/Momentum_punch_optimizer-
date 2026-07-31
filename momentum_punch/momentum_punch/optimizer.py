"""Expected-return adjustment and constrained mean-variance optimization."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config


@dataclass(frozen=True)
class OptimizationDiagnostics:
    status: str
    solver: str
    used_fallback: bool
    objective_value: float | None


def _validate_returns(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if returns.empty:
        raise ValueError("returns is empty")
    window = returns.sort_index().tail(lookback).replace([np.inf, -np.inf], np.nan)
    window = window.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if len(window) < config.MIN_OBSERVATIONS:
        raise ValueError(
            f"insufficient history: {len(window)} observations; "
            f"minimum is {config.MIN_OBSERVATIONS}"
        )
    return window


def historical_mu_sigma(
    returns: pd.DataFrame,
    lookback: int = config.COV_LOOKBACK_DAYS,
    covariance_method: str = "sample",
) -> tuple[pd.Series, pd.DataFrame]:
    window = _validate_returns(returns, lookback)
    mu = window.mean() * config.TRADING_DAYS

    if covariance_method == "sample":
        sigma = window.cov() * config.TRADING_DAYS
    elif covariance_method == "ewma":
        ew_cov = window.ewm(span=lookback, adjust=False).cov()
        sigma = ew_cov.loc[window.index[-1]] * config.TRADING_DAYS
    else:
        raise ValueError("covariance_method must be 'sample' or 'ewma'")

    sigma = sigma.reindex(index=mu.index, columns=mu.index)
    if not np.isfinite(mu.to_numpy()).all() or not np.isfinite(sigma.to_numpy()).all():
        raise ValueError("mu or covariance contains non-finite values")
    return mu, sigma


def adjust_mu_by_sentiment(
    mu: pd.Series,
    sentiment_scores: pd.Series,
    mode: str = config.SENTIMENT_ALPHA_MODE,
    additive_alpha_annual: float = config.SENTIMENT_ALPHA_ANNUAL,
    multiplicative_strength: float = config.SENTIMENT_TILT_STRENGTH,
) -> pd.Series:
    scores = (
        sentiment_scores.reindex(mu.index)
        .fillna(0.0)
        .clip(config.SENTIMENT_MIN, config.SENTIMENT_MAX)
    )
    if mode == "none":
        adjusted = mu.copy()
    elif mode == "additive":
        adjusted = mu + additive_alpha_annual * scores
    elif mode == "multiplicative":
        adjusted = mu * (1.0 + multiplicative_strength * scores)
    else:
        raise ValueError("mode must be 'none', 'additive', or 'multiplicative'")
    return adjusted.astype(float)


def tilt_mu_by_sentiment(mu: pd.Series, sentiment_scores: pd.Series) -> pd.Series:
    return adjust_mu_by_sentiment(mu, sentiment_scores)


def _fallback_weights(index: pd.Index, max_weight: float) -> pd.Series:
    n = len(index)
    if n == 0:
        raise ValueError("cannot allocate an empty universe")
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError("constraints are infeasible: max_weight * n < 1")
    return pd.Series(1.0 / n, index=index, dtype=float)


def optimize_weights(
    mu_adjusted: pd.Series,
    sigma: pd.DataFrame,
    risk_aversion: float = config.RISK_AVERSION,
    max_weight: float = config.MAX_WEIGHT_PER_ETF,
    min_weight: float = config.MIN_WEIGHT_PER_ETF,
    return_diagnostics: bool = False,
):
    if mu_adjusted.empty:
        raise ValueError("mu_adjusted is empty")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if min_weight < 0 or max_weight <= 0 or min_weight > max_weight:
        raise ValueError("invalid weight bounds")

    assets = mu_adjusted.index
    sigma = sigma.reindex(index=assets, columns=assets)
    mu_vec = mu_adjusted.astype(float).to_numpy()
    sigma_mat = sigma.astype(float).to_numpy()

    if not np.isfinite(mu_vec).all() or not np.isfinite(sigma_mat).all():
        raise ValueError("optimizer inputs contain NaN or infinity")

    n = len(assets)
    if min_weight * n > 1.0 + 1e-12 or max_weight * n < 1.0 - 1e-12:
        raise ValueError("weight constraints are infeasible")

    sigma_mat = (sigma_mat + sigma_mat.T) / 2
    sigma_mat = sigma_mat + np.eye(n) * config.COVARIANCE_RIDGE

    def objective(weights: np.ndarray) -> float:
        expected = float(mu_vec @ weights)
        variance = float(weights @ sigma_mat @ weights)
        return -(expected - risk_aversion * variance)

    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    x0 = np.repeat(1.0 / n, n)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    used_fallback = not bool(result.success)
    if used_fallback:
        warnings.warn(
            f"SLSQP failed ({result.message}); using deterministic fallback",
            RuntimeWarning,
        )
        weights = _fallback_weights(assets, max_weight)
    else:
        raw = np.asarray(result.x, dtype=float)
        if not np.isfinite(raw).all() or raw.sum() <= 0:
            weights = _fallback_weights(assets, max_weight)
            used_fallback = True
        else:
            raw = np.clip(raw, min_weight, max_weight)
            raw = raw / raw.sum()
            weights = pd.Series(raw, index=assets, dtype=float)

    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise RuntimeError("optimizer returned weights that do not sum to one")
    if (weights < min_weight - 1e-8).any() or (weights > max_weight + 1e-8).any():
        raise RuntimeError("optimizer returned weights outside constraints")

    diagnostics = OptimizationDiagnostics(
        status="optimal" if result.success else f"fallback:{result.message}",
        solver="scipy-SLSQP",
        used_fallback=used_fallback,
        objective_value=None if not result.success else float(-result.fun),
    )
    weights.name = "weight"
    return (weights, diagnostics) if return_diagnostics else weights
