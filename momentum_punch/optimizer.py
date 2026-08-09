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


def black_litterman_posterior(
    mu_prior: pd.Series,
    sigma: pd.DataFrame,
    sentiment_scores: pd.Series,
    view_confidence: dict[str, float] = None,
    tilt_strength: dict[str, float] = None,
    tau: float = 0.05,
) -> pd.Series:
    """
    Alternativa ao tilt linear simples (tilt_mu_by_sentiment): usa Black-Litterman
    de verdade pra combinar o retorno histórico (prior) com "views" de sentimento,
    ponderadas pela CONFIANÇA de cada view — não um multiplicador arbitrário.

    Por que trocar: testamos o tilt linear (mesmo dobrado pro ISUS11, que tem
    sinal estatisticamente validado) e o efeito na alocação final foi quase nulo
    — o mecanismo de transmissão é fraco demais pra competir com a instabilidade
    do Markowitz. Black-Litterman resolve isso formalmente: quanto maior a
    confiança na view (Ω menor), mais o posterior se afasta do prior histórico
    e se aproxima da view — de um jeito matematicamente principiado, não um
    "* 2" arbitrário.

    view_confidence: 0 a 1 por ticker (0 = ignora a view, usa só o histórico;
      1 = confia total na view). Default: config.SENTIMENT_TILT_STRENGTH_POR_TICKER
      normalizado — reaproveita a mesma calibração da validação estatística.
    tilt_strength: mesma função de antes (define O QUE a view diz, não quanto
      pesa — isso é o papel do view_confidence agora).
    tau: incerteza do prior histórico (valor pequeno padrão em BL, 0.01-0.05).

    Tickers com confiança 0 (ex: GOVE11) são excluídos da matriz de views —
    pra eles o posterior é literalmente igual ao prior histórico puro.
    """
    if view_confidence is None:
        base = config.SENTIMENT_TILT_STRENGTH_POR_TICKER
        maior = max(base.values()) if base else 1.0
        view_confidence = {t: (v / maior if maior > 0 else 0.0) for t, v in base.items()}
    if tilt_strength is None:
        tilt_strength = config.SENTIMENT_TILT_STRENGTH_POR_TICKER

    tickers_com_view = [t for t in mu_prior.index if view_confidence.get(t, 0.0) > 0]

    if not tickers_com_view:
        return mu_prior  # ninguém tem confiança > 0 -> posterior = prior, sem views

    n = len(mu_prior)
    k = len(tickers_com_view)
    sigma_mat = sigma.loc[mu_prior.index, mu_prior.index].values

    # P: matriz de views absolutas (1 linha por ticker com view, 1 na posição do ticker)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)
    sentiment_scores = sentiment_scores.reindex(mu_prior.index).fillna(0.0)

    for i, ticker in enumerate(tickers_com_view):
        idx = mu_prior.index.get_loc(ticker)
        P[i, idx] = 1.0
        forca = tilt_strength.get(ticker, config.SENTIMENT_TILT_STRENGTH_BASE)
        Q[i] = mu_prior[ticker] * (1 + forca * sentiment_scores[ticker])
        confianca = min(max(view_confidence[ticker], 1e-4), 1.0)  # evita divisão por zero
        p_row = P[i : i + 1, :]
        var_view = (p_row @ sigma_mat @ p_row.T).item()
        omega_diag[i] = (1 / confianca - 1) * var_view

    Omega = np.diag(omega_diag)
    tau_sigma = tau * sigma_mat
    tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(n) * 1e-10)
    omega_inv = np.linalg.inv(Omega + np.eye(k) * 1e-10)

    M_inv = tau_sigma_inv + P.T @ omega_inv @ P
    M = np.linalg.inv(M_inv + np.eye(n) * 1e-10)
    posterior = M @ (tau_sigma_inv @ mu_prior.values + P.T @ omega_inv @ Q)

    return pd.Series(posterior, index=mu_prior.index)


def tilt_mu_by_sentiment(mu: pd.Series, sentiment_scores: pd.Series) -> pd.Series:
    """
    Aplica o tilt do Sentiment Alpha Score sobre o retorno esperado histórico,
    de forma ADITIVA (mu + kappa*sentiment), não multiplicativa. O motivo:
    tilt multiplicativo (mu * (1+kappa*sentiment)) inverte o efeito quando o
    mu histórico é negativo — sentimento positivo, nesse caso, deixaria o
    retorno esperado AINDA MAIS negativo, o oposto do que deveria acontecer.
    Aditivo não tem esse problema: o deslocamento não depende do sinal de mu.
    """
    sentiment_scores = sentiment_scores.reindex(mu.index).fillna(0.0)
    kappas = pd.Series(
        [config.SENTIMENT_TILT_STRENGTH_POR_TICKER.get(t, config.SENTIMENT_TILT_STRENGTH_BASE) for t in mu.index],
        index=mu.index,
    )
    return mu + kappas * sentiment_scores


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
