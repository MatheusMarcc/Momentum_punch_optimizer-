"""
Testa cada peça do projeto isoladamente e imprime um resumo claro no final —
em vez de descobrir erro por erro rodando os scripts principais um a um.

Uso:
    python run_diagnostics.py
    python run_diagnostics.py --include-gdelt   # inclui o teste do GDELT (lento, rate limit)
    python run_diagnostics.py --include-cvm     # inclui o teste da CVM (download grande)

Cada checagem usa um recorte pequeno/rápido (poucos dias, poucos itens) — não
é pra popular os CSVs de verdade, é só pra confirmar que cada peça responde.
Pra coleta de verdade, use collect_data.py / build_sentiment_dataset.py normalmente.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback

RESULTADOS: list[tuple[str, str, str]] = []  # (nome, status, detalhe)


def check(nome: str, fn):
    try:
        detalhe = fn()
        RESULTADOS.append((nome, "OK", detalhe or ""))
        print(f"[OK]     {nome}: {detalhe or ''}")
    except Exception as exc:
        RESULTADOS.append((nome, "FALHOU", str(exc)))
        print(f"[FALHOU] {nome}: {exc}")


def skip(nome: str, motivo: str):
    RESULTADOS.append((nome, "PULADO", motivo))
    print(f"[PULADO] {nome}: {motivo}")


# ---------------------------------------------------------------------------

def check_dependencias():
    import numpy, pandas, cvxpy, feedparser, requests, bs4  # noqa
    return "numpy/pandas/cvxpy/feedparser/requests/bs4 importam ok"


def check_ollama_online():
    import requests
    from momentum_punch import config

    resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
    resp.raise_for_status()
    modelos = [m["name"] for m in resp.json().get("models", [])]
    if config.OLLAMA_MODEL not in modelos and not any(config.OLLAMA_MODEL.split(":")[0] in m for m in modelos):
        raise RuntimeError(
            f"Ollama tá rodando mas modelo '{config.OLLAMA_MODEL}' não encontrado. "
            f"Modelos instalados: {modelos}. Rode: ollama pull {config.OLLAMA_MODEL}"
        )
    return f"Ollama online, modelos instalados: {modelos}"


def check_sentiment_scoring():
    from momentum_punch import sentiment

    resultado = sentiment.score_texts_for_ticker(
        "BOVA11", ["Ibovespa fecha em alta com otimismo do mercado"], provider="ollama"
    )
    if resultado.rationale.startswith("[fallback"):
        raise RuntimeError(f"LLM respondeu mas caiu no fallback: {resultado.rationale}")
    return f"score={resultado.score}, rationale='{resultado.rationale[:50]}'"


def check_bacen_sgs():
    from momentum_punch.collectors import bacen_sgs

    hoje = dt.date.today()
    inicio = (hoje - dt.timedelta(days=15)).strftime("%d/%m/%Y")
    fim = hoje.strftime("%d/%m/%Y")
    df = bacen_sgs.fetch_all_series(inicio, fim, series={"cdi_diario": 12})
    if df.empty:
        raise RuntimeError("retornou vazio")
    return f"{len(df)} registros de CDI nos últimos 15 dias"


def check_bacen_focus():
    from momentum_punch.collectors import bacen_focus

    hoje = dt.date.today()
    inicio = (hoje - dt.timedelta(days=60)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")
    df = bacen_focus.fetch_expectativas_selic(inicio, fim)
    if df.empty:
        raise RuntimeError("retornou vazio — endpoint ExpectativasMercadoSelic pode ter mudado, confira no navegador")
    return f"{len(df)} registros de expectativa Selic nos últimos 60 dias"


def check_rss_news():
    from momentum_punch.collectors import rss_news

    df = rss_news.fetch_feed("infomoney", rss_news.FEEDS["infomoney"])
    if df.empty:
        raise RuntimeError("0 itens retornados (feed pode estar fora do ar ou filtro derrubou tudo)")
    return f"{len(df)} itens do InfoMoney (já filtrados)"


def check_yfinance():
    import yfinance as yf

    data = yf.download("BOVA11.SA", period="5d", progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError("retornou vazio")
    return f"{len(data)} dias de preço BOVA11.SA (últimos 5 dias úteis)"


def check_optimizer():
    import pandas as pd
    from momentum_punch import optimizer

    mu = pd.Series([0.10, 0.12, 0.08, 0.11], index=["A", "B", "C", "D"])
    sigma = pd.DataFrame(
        [[0.04, 0.01, 0.00, 0.01],
         [0.01, 0.05, 0.01, 0.00],
         [0.00, 0.01, 0.03, 0.01],
         [0.01, 0.00, 0.01, 0.04]],
        index=mu.index, columns=mu.index,
    )
    pesos = optimizer.optimize_weights(mu, sigma)
    if abs(pesos.sum() - 1.0) > 1e-4:
        raise RuntimeError(f"pesos não somam 1: {pesos.sum()}")
    return f"otimizou ok, pesos somam {pesos.sum():.4f}"


def check_synthetic_backtest():
    from momentum_punch import backtest, synthetic_data

    prices = synthetic_data.generate_prices(n_days=200)
    cdi = synthetic_data.generate_cdi_daily_return(n_days=200)
    sentiment_scores = synthetic_data.generate_sentiment_scores(prices.index)
    stress = synthetic_data.generate_stress_index(prices.index)
    result = backtest.run_backtest(prices, cdi, sentiment_scores, stress, benchmark="60_40")
    if result["equity_curve"].empty:
        raise RuntimeError("equity_curve veio vazia")
    return f"backtest sintético rodou, {len(result['equity_curve'])} dias, Sharpe={result['metrics']['Momentum Punch']['Sharpe']}"


def check_reddit_credenciais():
    import os
    if not os.environ.get("REDDIT_CLIENT_ID") or not os.environ.get("REDDIT_CLIENT_SECRET"):
        raise RuntimeError("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET não definidos")
    return "credenciais presentes (não testei a chamada de verdade)"


def check_gdelt():
    from momentum_punch.collectors import gdelt

    hoje = dt.date.today()
    inicio = hoje - dt.timedelta(days=7)
    df = gdelt.fetch_articles("Brasil economia", dt.datetime.combine(inicio, dt.time.min), dt.datetime.combine(hoje, dt.time.max), max_records=10)
    return f"{len(df)} artigos (últimos 7 dias, teste rápido)"


def check_cvm():
    from momentum_punch.collectors import cvm_fatos_relevantes

    df = cvm_fatos_relevantes.fetch_ipe_ano(dt.date.today().year)
    if df.empty:
        raise RuntimeError("retornou vazio")
    return f"{len(df)} registros IPE no ano corrente"


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-gdelt", action="store_true")
    parser.add_argument("--include-cvm", action="store_true")
    parser.add_argument("--include-reddit", action="store_true")
    args = parser.parse_args()

    print("=== Diagnóstico Momentum Punch ===\n")

    check("Dependências Python", check_dependencias)
    check("Ollama online + modelo instalado", check_ollama_online)
    check("Sentiment scoring (Ollama de verdade)", check_sentiment_scoring)
    check("Bacen SGS (CDI)", check_bacen_sgs)
    check("Bacen Focus (Expectativas)", check_bacen_focus)
    check("RSS notícias (InfoMoney)", check_rss_news)
    check("Preço real de ETF (yfinance)", check_yfinance)
    check("Otimizador (Markowitz/cvxpy)", check_optimizer)
    check("Backtest sintético ponta a ponta", check_synthetic_backtest)

    if args.include_reddit:
        check("Reddit (credenciais)", check_reddit_credenciais)
    else:
        skip("Reddit", "use --include-reddit pra testar")

    if args.include_gdelt:
        check("GDELT", check_gdelt)
    else:
        skip("GDELT", "desativado por decisão do projeto (rate limit não confiável) — use --include-gdelt pra forçar")

    if args.include_cvm:
        check("CVM (fatos relevantes)", check_cvm)
    else:
        skip("CVM", "download grande, não testado por padrão — use --include-cvm")

    print("\n=== Resumo ===")
    ok = sum(1 for _, s, _ in RESULTADOS if s == "OK")
    falhou = sum(1 for _, s, _ in RESULTADOS if s == "FALHOU")
    pulado = sum(1 for _, s, _ in RESULTADOS if s == "PULADO")
    for nome, status, detalhe in RESULTADOS:
        marcador = {"OK": "✓", "FALHOU": "✗", "PULADO": "-"}[status]
        print(f"  {marcador} {nome}: {status}")
    print(f"\n{ok} OK | {falhou} falharam | {pulado} pulados")

    if falhou > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
