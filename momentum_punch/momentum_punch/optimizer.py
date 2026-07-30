"""
Bloco 4 e 5 do pipeline: Vetor de retorno ajustado + Otimização Markowitz.

Função objetivo (conforme o deck):
    Max( Retorno Esperado Ajustado por Sentimento ) - ( Penalidade de Risco * Variância )

mu_ajustado[i] = mu_historico[i] * (1 + SENTIMENT_TILT_STRENGTH * sentiment_score[i])

O CDI entra como ativo livre de risco: seu "retorno esperado" é o próprio CDI vigente
e sua variância/covariância com os demais é ~0 (tratado à parte, fora do otimizador
de risco, e alocado via o overlay do circuit breaker em risk_overlay.py).
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from . import config


def historical_mu_sigma(returns: pd.DataFrame, lookback: int = config.COV_LOOKBACK_DAYS):
    """
    Estima retorno esperado (mu, anualizado) e matriz de covariância (Sigma, anualizada)
    a partir de uma janela trailing de retornos diários dos ETFs (colunas = tickers).
    """
    window = returns.tail(lookback)
    mu = window.mean() * 252
    sigma = window.cov() * 252
    return mu, sigma


def tilt_mu_by_sentiment(mu: pd.Series, sentiment_scores: pd.Series) -> pd.Series:
    """Aplica o tilt do Sentiment Alpha Score sobre o retorno esperado histórico."""
    sentiment_scores = sentiment_scores.reindex(mu.index).fillna(0.0)
    tilt_factor = 1 + config.SENTIMENT_TILT_STRENGTH * sentiment_scores
    return mu * tilt_factor


def optimize_weights(
    mu_adjusted: pd.Series,
    sigma: pd.DataFrame,
    risk_aversion: float = config.RISK_AVERSION,
    max_weight: float = config.MAX_WEIGHT_PER_ETF,
    min_weight: float = config.MIN_WEIGHT_PER_ETF,
) -> pd.Series:
    """
    Resolve o Markowitz modificado via cvxpy:
        max  mu_adjusted' w  -  risk_aversion * w' Sigma w
        s.a. sum(w) == 1, min_weight <= w <= max_weight  (long-only, sem alavancagem)

    Retorna os pesos ótimos entre os ETFs de risco (soma = 1). A fração alocada a
    CDI é decidida depois, no overlay do circuit breaker (risk_overlay.py), que pode
    reduzir a exposição total a renda variável e mandar o excedente pro CDI.
    """
    n = len(mu_adjusted)
    w = cp.Variable(n)
    mu_vec = mu_adjusted.values
    sigma_mat = sigma.loc[mu_adjusted.index, mu_adjusted.index].values

    # simetriza (evita ruído de ponto flutuante) e regulariza pra garantir PSD numérica
    sigma_mat = (sigma_mat + sigma_mat.T) / 2 + np.eye(n) * 1e-8

    objective = cp.Maximize(mu_vec @ w - risk_aversion * cp.quad_form(w, sigma_mat))
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if w.value is None:
        # fallback: equal-weight se o solver falhar (ex. dados degenerados)
        weights = np.repeat(1 / n, n)
    else:
        weights = np.clip(w.value, 0, None)
        weights = weights / weights.sum()

    return pd.Series(weights, index=mu_adjusted.index, name="weight")
