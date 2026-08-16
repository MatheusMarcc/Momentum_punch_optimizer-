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


# Os cinco benchmarks da seção 5.2 do pré-relatório. "markowitz_sem_texto" não
# está aqui de propósito: ele não é uma carteira de pesos fixos, é o próprio
# motor rodando com sentimento neutro — ou seja, a configuração A0 da matriz de
# ablação. Duplicar a lógica aqui seria uma segunda implementação da mesma
# coisa, passível de divergir; run_ablations.py já o produz.
BENCHMARKS = {
    "bova11": "100% BOVA11",
    "60_40": "60% BOVA11 / 40% CDI",
    "equal_weight": "Pesos iguais entre ETFs",
    "esg_estatica": "ESG estática (ISUS11/GOVE11/REVE11)",
    "40_40_20": "40% BOVA11 / 40% IVVB11 / 20% CDI",
}

# Carteira ESG estática: os três ETFs temáticos em peso igual, sem otimização e
# sem overlay. É o "e se eu só comprasse a temática e segurasse?" — a pergunta
# que um avaliador faz primeiro quando vê uma estratégia ESG sofisticada.
ESG_ESTATICA = ["ISUS11", "GOVE11", "REVE11"]


def _retorno_benchmark(nome: str, date, returns: pd.DataFrame, cdi_hoje: float,
                       retornos_extra: pd.DataFrame | None) -> float:
    """Retorno de um dia do benchmark `nome`."""
    if nome == "bova11":
        return returns.loc[date, "BOVA11"]
    if nome == "60_40":
        return 0.6 * returns.loc[date, "BOVA11"] + 0.4 * cdi_hoje
    if nome == "equal_weight":
        disponiveis = [t for t in config.TICKERS if t in returns.columns]
        return sum(returns.loc[date, t] for t in disponiveis) / len(disponiveis)
    if nome == "esg_estatica":
        disponiveis = [t for t in ESG_ESTATICA if t in returns.columns]
        return sum(returns.loc[date, t] for t in disponiveis) / len(disponiveis)
    if nome == "40_40_20":
        # sem preço do IVVB11 esse benchmark não existe — devolver algo
        # "parecido" seria inventar um comparador, então ele fica achatado em
        # CDI e o run que o pedir precisa passar benchmark_prices
        if retornos_extra is None or "IVVB11" not in retornos_extra.columns:
            return cdi_hoje
        ivvb = retornos_extra["IVVB11"].get(date, 0.0)
        return 0.4 * returns.loc[date, "BOVA11"] + 0.4 * ivvb + 0.2 * cdi_hoje
    raise ValueError(f"benchmark desconhecido: {nome}")


def run_backtest(
    prices: pd.DataFrame,
    cdi_daily_return: pd.Series,
    sentiment_scores: pd.DataFrame,
    stress_index: pd.Series,
    benchmark: str = "bova11",  # qualquer chave de BENCHMARKS
    mu_method: str = "tilt_linear",  # "tilt_linear" ou "black_litterman"
    transaction_cost_bps: float = 0.0,  # custo por unidade de giro, em bps (ex: 10 = 0.10%)
    benchmark_prices: pd.DataFrame | None = None,
    tilt_mode: str = "aditivo",  # "aditivo" (produção) ou "multiplicativo" (braço A7)
    cov_estimator: str = "amostral",  # "amostral" (A11), "ledoit_wolf" ou "ewma" (A12)
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
    benchmark_prices: preços de ativos que entram SÓ em benchmark, nunca no
        universo investível (hoje: IVVB11, pro comparador 40/40/20). Manter
        isso separado de `prices` é o que impede um ativo de comparação de
        vazar pro otimizador.

    Retorna um dict com: equity_curve, benchmark_curve (o escolhido em
    `benchmark`), benchmarks (todos), weights_history, metrics.
    """
    returns = prices.pct_change().dropna()
    # warm-up: precisa de COV_LOOKBACK_DAYS de retornos antes do primeiro rebalanceamento
    tradeable_index = returns.index[config.COV_LOOKBACK_DAYS:]
    rebal_dates = _rebalance_dates(tradeable_index)
    taxa = transaction_cost_bps / 10000.0

    retornos_extra = benchmark_prices.pct_change().dropna() if benchmark_prices is not None else None

    portfolio_value = 1.0
    equity_curve = {}
    valores_benchmark = {n: 1.0 for n in BENCHMARKS}
    curvas_benchmark = {n: {} for n in BENCHMARKS}
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
            mu, sigma = optimizer.historical_mu_sigma(returns.loc[:date], cov_estimator=cov_estimator)
            today_sentiment = sentiment_scores.loc[:date].iloc[-1] if not sentiment_scores.loc[:date].empty else pd.Series(0.0, index=mu.index)
            if mu_method == "black_litterman":
                mu_adj = optimizer.black_litterman_posterior(mu, sigma, today_sentiment)
            else:
                mu_adj = optimizer.tilt_mu_by_sentiment(mu, today_sentiment, modo=tilt_mode)
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

        # todos os benchmarks são acumulados na mesma passada — o parâmetro
        # `benchmark` só escolhe qual sai em `benchmark_curve` (compatibilidade
        # com quem já consumia isso); a tabela completa vem em `benchmarks`
        cdi_hoje = cdi_daily_return.get(date, 0.0)
        for nome in BENCHMARKS:
            valores_benchmark[nome] *= (1 + _retorno_benchmark(nome, date, returns, cdi_hoje, retornos_extra))
            curvas_benchmark[nome][date] = valores_benchmark[nome]

    equity_curve = pd.Series(equity_curve, name="Momentum Punch")
    curvas_benchmark = {n: pd.Series(c, name=BENCHMARKS[n]) for n, c in curvas_benchmark.items()}
    if benchmark not in curvas_benchmark:
        raise ValueError(f"benchmark deve ser um de {list(BENCHMARKS)}")
    benchmark_curve = curvas_benchmark[benchmark].rename("Benchmark")

    metrics = {"Momentum Punch": _performance_metrics(equity_curve, cdi_daily_return)}
    metrics["Momentum Punch"]["Custo acumulado (giro)"] = f"{custo_acumulado:.2%}"
    metrics["Momentum Punch"].update(_metricas_de_exposicao(weights_history, turnover_history))
    for nome, curva in curvas_benchmark.items():
        metrics[BENCHMARKS[nome]] = _performance_metrics(curva, cdi_daily_return)
    # alias mantido pra quem consumia o formato antigo
    metrics["Benchmark"] = metrics[BENCHMARKS[benchmark]]

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "benchmarks": curvas_benchmark,
        "weights_history": weights_history,
        "turnover_history": turnover_history,
        "metrics": metrics,
    }


def metricas_no_periodo(
    result: dict,
    cdi_daily_return: pd.Series,
    inicio: str | pd.Timestamp | None = None,
    fim: str | pd.Timestamp | None = None,
) -> dict:
    """
    Recalcula as métricas num recorte da janela, renormalizando cada curva pra
    começar em 1,0 no início do recorte.

    Existe pro protocolo de treino/teste: o backtest roda CONTÍNUO (senão o
    período de teste perderia os 60 dias de warm-up da covariância e começaria
    sem pesos vigentes), mas a avaliação final tem que sair só do período de
    teste. Recortar a curva depois é diferente de rodar o backtest só no
    recorte — e é a versão correta, porque preserva o estado que a estratégia
    realmente carregava ao entrar no teste.
    """
    def _recorta(curva: pd.Series) -> pd.Series:
        c = curva.loc[inicio:fim] if (inicio or fim) else curva
        return c / c.iloc[0] if len(c) else c

    equity = _recorta(result["equity_curve"])
    if equity.empty:
        return {}

    metrics = {"Momentum Punch": _performance_metrics(equity, cdi_daily_return)}
    pesos_no_periodo = {
        d: w for d, w in result["weights_history"].items()
        if (inicio is None or d >= pd.Timestamp(inicio)) and (fim is None or d <= pd.Timestamp(fim))
    }
    giro_no_periodo = {d: g for d, g in result["turnover_history"].items() if d in pesos_no_periodo}
    metrics["Momentum Punch"].update(_metricas_de_exposicao(pesos_no_periodo, giro_no_periodo))
    for nome, curva in result["benchmarks"].items():
        metrics[BENCHMARKS[nome]] = _performance_metrics(_recorta(curva), cdi_daily_return)
    return metrics


def _metricas_de_exposicao(weights_history: dict, turnover_history: dict) -> dict:
    """Métricas de CARTEIRA (não da curva): quanto ficou em risco, quão
    concentrado, com que frequência o circuit breaker disparou. A seção 5.4 do
    pré-relatório pede as três — sem elas dá pra ter Sharpe bonito escondendo
    uma carteira que passou o tempo todo em caixa ou tudo num ativo só."""
    if not weights_history:
        return {}

    exposicoes, concentracoes, n_risk_off = [], [], 0
    for pesos in weights_history.values():
        ativos = {k: v for k, v in pesos.items() if not k.startswith("_") and k != config.RISK_FREE}
        exposicoes.append(sum(ativos.values()))
        concentracoes.append(max(ativos.values()) if ativos else 0.0)
        if pesos.get("_mode") == "RISK-OFF":
            n_risk_off += 1

    total = len(weights_history)
    return {
        "Exposição média a risco": f"{sum(exposicoes) / total:.1%}",
        "Concentração máxima": f"{max(concentracoes):.1%}",
        "Rebalanceamentos em Risk-Off": f"{n_risk_off}/{total} ({n_risk_off / total:.0%})",
        "Giro médio por rebal.": f"{pd.Series(turnover_history).mean():.1%}" if turnover_history else "n/a",
    }


def _performance_metrics(equity_curve: pd.Series, rf_daily: pd.Series | None = None) -> dict:
    """
    Métricas de performance da curva de patrimônio.

    Sobre o Sharpe: o índice de Sharpe é definido sobre o retorno EXCEDENTE ao
    ativo livre de risco — (r - rf), não r puro. Num mercado com CDI a ~13% a.a.
    isso não é detalhe de casa decimal: a razão retorno/volatilidade crua fica
    2x a 3x maior que o Sharpe de verdade, e a diferença entre estratégia e
    benchmark muda de tamanho. Por isso as duas medidas são reportadas
    SEPARADAS e rotuladas pelo que cada uma é — a razão crua fica visível
    porque é comparável com material que a reporta como "Sharpe", mas o nome
    dela aqui deixa explícito que não é Sharpe.

    rf_daily: retorno diário do ativo livre de risco (CDI), em decimal. Se None
    (ex: dado sintético sem CDI), o Sharpe excedente sai como "n/a" — em vez de
    silenciosamente virar a razão crua e ser lido como Sharpe depois.
    """
    daily_ret = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = len(equity_curve) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else float("nan")
    vol_annual = daily_ret.std() * (252 ** 0.5)
    retorno_sobre_vol = (daily_ret.mean() * 252) / vol_annual if vol_annual > 0 else float("nan")
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    if rf_daily is not None:
        rf = rf_daily.reindex(daily_ret.index).fillna(0.0)
        excesso = daily_ret - rf
        vol_excesso = excesso.std() * (252 ** 0.5)
        sharpe = (excesso.mean() * 252) / vol_excesso if vol_excesso > 0 else float("nan")
        sharpe_str = f"{sharpe:.2f}"
        cdi_str = f"{(1 + rf).prod() - 1:.2%}"

        # Sortino: mesma ideia do Sharpe, mas penalizando só o desvio ABAIXO do
        # CDI. Volatilidade pra cima não é risco pro cotista — o Sharpe trata
        # as duas igual, o Sortino não.
        downside = excesso[excesso < 0]
        dd_vol = downside.std() * (252 ** 0.5) if len(downside) > 1 else float("nan")
        sortino_str = f"{(excesso.mean() * 252) / dd_vol:.2f}" if dd_vol and dd_vol > 0 else "n/a"
    else:
        sharpe_str = sortino_str = "n/a (sem série de CDI)"
        cdi_str = "n/a"

    # Calmar: CAGR por unidade de dor máxima. Complementa o Sharpe porque
    # captura o pior caminho percorrido, não a dispersão média.
    calmar_str = f"{cagr / abs(max_drawdown):.2f}" if max_drawdown < 0 else "n/a"

    # Tempo submerso: fração dos dias abaixo do pico anterior, e a maior
    # sequência ininterrupta. Duas carteiras com o mesmo drawdown máximo podem
    # ter experiências opostas — uma recupera em um mês, a outra fica 2 anos no
    # vermelho. O relatório pede essa distinção.
    submerso = drawdown < -1e-9
    frac_submerso = submerso.mean()
    maior_seq, seq = 0, 0
    for v in submerso:
        seq = seq + 1 if v else 0
        maior_seq = max(maior_seq, seq)

    mensal = equity_curve.resample("ME").last().pct_change().dropna()

    return {
        "Retorno total": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Volatilidade anual": f"{vol_annual:.2%}",
        "Sharpe (excedente ao CDI)": sharpe_str,
        "Sortino (excedente ao CDI)": sortino_str,
        "Calmar": calmar_str,
        "Retorno/Vol (NÃO é Sharpe)": f"{retorno_sobre_vol:.2f}",
        "Max drawdown": f"{max_drawdown:.2%}",
        "Tempo submerso": f"{frac_submerso:.0%} dos dias (maior sequência: {maior_seq} dias)",
        "Melhor mês": f"{mensal.max():.2%}" if len(mensal) else "n/a",
        "Pior mês": f"{mensal.min():.2%}" if len(mensal) else "n/a",
        "CDI acumulado no período": cdi_str,
    }
