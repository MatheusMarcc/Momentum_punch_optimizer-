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
    mu_method: str = "tilt_linear",  # "tilt_linear" ou "black_litterman"
    transaction_cost_bps: float = 0.0,  # custo por unidade de giro, em bps (ex: 10 = 0.10%)
) -> dict:
    """
    prices: DataFrame (index=data, colunas=config.TICKERS) com preços dos ETFs.
    cdi_daily_return: Series com retorno diário do CDI.
    sentiment_scores: DataFrame (index=data, colunas=config.TICKERS), já em EMA.
    stress_index: Series (index=data) com o índice de estresse (0 a 1).
    mu_method: como combinar sentimento com o retorno histórico —
        "tilt_linear" (mu + kappa*sentiment, ADITIVO — ver optimizer.tilt_mu_by_sentiment
        pro motivo de ser aditivo e não multiplicativo) ou
        "black_litterman" (combinação bayesiana ponderada por confiança —
        ver optimizer.black_litterman_posterior).
    transaction_cost_bps: custo proporcional ao GIRO (turnover) a cada
        rebalanceamento, em pontos-base (100 bps = 1%). Giro = soma dos
        valores absolutos da mudança de peso por ativo. Custo = giro * taxa,
        debitado do patrimônio no dia do rebalanceamento. 0 = sem custo
        (comportamento original).

    Retorna um dict com: equity_curve, benchmark_curve, weights_history, metrics.
    """
    returns = prices.pct_change().dropna()
    # warm-up: precisa de COV_LOOKBACK_DAYS de retornos antes do primeiro rebalanceamento
    tradeable_index = returns.index[config.COV_LOOKBACK_DAYS:]
    rebal_dates = _rebalance_dates(tradeable_index)
    taxa = transaction_cost_bps / 10000.0

    portfolio_value = 1.0
    benchmark_value = 1.0
    equity_curve = {}
    benchmark_curve = {}
    weights_history = {}
    turnover_history = {}
    custo_acumulado = 0.0

    # Ponto-in-time corrigido: pesos decididos usando informação até `date`
    # (inclusive) só valem a partir do PRÓXIMO dia, não do próprio `date` —
    # antes, o retorno do mesmo dia usado pra estimar mu/sigma também era
    # capturado pelos pesos decididos com aquela informação (vazamento).
    current_weights = None  # pesos em vigor HOJE (decididos em dia(s) anterior(es))
    for i, date in enumerate(tradeable_index):
        # 1. aplica os pesos já vigentes (decididos antes) ao retorno de HOJE
        if current_weights is not None:
            etf_ret = sum(
                current_weights.get(t, 0.0) * returns.loc[date, t] for t in config.TICKERS
            )
            cdi_ret = current_weights.get(config.RISK_FREE, 0.0) * cdi_daily_return.get(date, 0.0)
            day_return = etf_ret + cdi_ret
        else:
            day_return = 0.0  # ainda não há decisão prévia (primeiro dia da janela)
        portfolio_value *= (1 + day_return)

        # 2. decide os pesos de amanhã em diante, usando informação até HOJE
        if date in rebal_dates or current_weights is None:
            mu, sigma = optimizer.historical_mu_sigma(returns.loc[:date])
            today_sentiment = sentiment_scores.loc[:date].iloc[-1] if not sentiment_scores.loc[:date].empty else pd.Series(0.0, index=mu.index)
            if mu_method == "black_litterman":
                mu_adj = optimizer.black_litterman_posterior(mu, sigma, today_sentiment)
            else:
                mu_adj = optimizer.tilt_mu_by_sentiment(mu, today_sentiment)
            etf_weights = optimizer.optimize_weights(mu_adj, sigma)

            today_stress = stress_index.loc[:date].iloc[-1] if not stress_index.loc[:date].empty else 0.0
            final_weights = risk_overlay.apply_circuit_breaker(etf_weights, today_stress)

            # giro = soma |peso novo - peso antigo| por ativo (inclui CDI);
            # primeira decisão da série não tem "antigo" pra comparar (giro=0,
            # convenção comum: a montagem inicial da carteira não é "custo de
            # rebalanceamento", é o próprio início da estratégia)
            if current_weights is not None:
                todos_ativos = set(final_weights) | set(current_weights)
                todos_ativos = {a for a in todos_ativos if not a.startswith("_")}
                giro = sum(abs(final_weights.get(a, 0.0) - current_weights.get(a, 0.0)) for a in todos_ativos)
            else:
                giro = 0.0

            custo = giro * taxa
            custo_acumulado += custo
            portfolio_value *= (1 - custo)  # custo debitado no dia da decisão, antes de valer os novos pesos

            turnover_history[date] = giro
            current_weights = final_weights  # só passa a valer no próximo `date` do loop
            weights_history[date] = final_weights

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
    metrics["Momentum Punch"]["Custo acumulado (giro)"] = f"{custo_acumulado:.2%}"
    metrics["Momentum Punch"]["Giro médio por rebal."] = f"{pd.Series(turnover_history).mean():.1%}" if turnover_history else "n/a"

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "weights_history": weights_history,
        "turnover_history": turnover_history,
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
