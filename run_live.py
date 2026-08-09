"""
Execução AO VIVO — não é backtest. Puxa notícia de agora, escora via FinBERT
na hora, e devolve a alocação recomendada PRA HOJE. É a implementação real do
fluxo original do deck: "08:00 da manhã, o robô acorda, lê notícia, decide".

Uso:
    python run_live.py                # roda uma vez, imprime a decisão de agora
    python run_live.py --loop 3600    # roda em loop, a cada 3600s (1h), contínuo
    python run_live.py --json out.json  # também salva a decisão em JSON
        (formato pensado pro "Dashboard Interativo" do deck original: um
        arquivo que um front-end simples consumiria sem recalcular nada)

Pré-requisitos: já ter rodado fetch_prices_yfinance.py e collect_data.py
--only bacen_sgs pelo menos uma vez (usa o histórico de preço/CDI já salvo
pra estimar mu/sigma — não recalcula 3 anos de preço a cada execução, só
o SENTIMENTO é ao vivo).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import pandas as pd

from build_sentiment_dataset import _filtra_por_ticker
from momentum_punch import config, optimizer, risk_overlay, sentiment
from momentum_punch.collectors import rss_news


def _log(msg: str):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")


def fetch_live_sentiment() -> dict[str, float]:
    """Puxa notícia AGORA (RSS, sem espera) e escora via FinBERT na hora —
    latência de segundos, não minutos, já que FinBERT é local e determinístico."""
    _log("Buscando notícias agora (RSS)...")
    df = rss_news.fetch_all()
    if df.empty:
        _log("Nenhuma notícia nova encontrada nos feeds agora.")
        return {t: 0.0 for t in config.TICKERS}

    textos = (df["titulo"].fillna("") + ". " + df["resumo"].fillna("")).tolist()
    _log(f"{len(textos)} notícia(s) coletada(s), escorando por ticker...")

    scores = {}
    for ticker in config.TICKERS:
        textos_relevantes = _filtra_por_ticker(textos, ticker)
        if not textos_relevantes:
            scores[ticker] = 0.0
            continue
        resultado = sentiment.score_texts_for_ticker(ticker, textos_relevantes)
        scores[ticker] = resultado.score
        _log(f"  {ticker}: score={resultado.score:+.3f} ({len(textos_relevantes)} notícia(s) relevante(s))")
    return scores


def load_historical_mu_sigma() -> tuple[pd.Series, pd.DataFrame]:
    """Usa o histórico de preço já salvo (fetch_prices_yfinance.py) pra estimar
    mu/sigma — isso não muda minuto a minuto, só o sentimento muda ao vivo."""
    prices_long = pd.read_csv("data/raw/etf_prices.csv", parse_dates=["data"])
    prices_wide = prices_long.pivot(index="data", columns="ticker", values="close")
    returns = prices_wide.pct_change().dropna()
    return optimizer.historical_mu_sigma(returns)


def load_latest_stress() -> float:
    """Usa o último valor conhecido do stress_index (Focus) — não recalcula
    2 anos de expectativas a cada execução ao vivo, só lê o mais recente."""
    try:
        stress = pd.read_csv("data/processed/stress_index.csv", index_col=0, parse_dates=True).iloc[:, 0]
        return float(stress.iloc[-1])
    except (FileNotFoundError, IndexError):
        _log("stress_index.csv não encontrado/vazio — assumindo Risk-On (stress=0.0)")
        return 0.0


def run_once(mu_method: str = "tilt_linear") -> dict:
    _log("=== Execução ao vivo iniciada ===")

    mu, sigma = load_historical_mu_sigma()
    sentiment_scores = pd.Series(fetch_live_sentiment())
    stress_now = load_latest_stress()

    if mu_method == "black_litterman":
        mu_adj = optimizer.black_litterman_posterior(mu, sigma, sentiment_scores)
    else:
        mu_adj = optimizer.tilt_mu_by_sentiment(mu, sentiment_scores)

    etf_weights = optimizer.optimize_weights(mu_adj, sigma)
    pesos_finais = risk_overlay.apply_circuit_breaker(etf_weights, stress_now)

    decisao = {
        "timestamp": dt.datetime.now().isoformat(),
        "sentiment_scores": sentiment_scores.to_dict(),
        "stress_index": stress_now,
        "modo": pesos_finais["_mode"],
        "pesos": {k: round(v, 4) for k, v in pesos_finais.items() if not k.startswith("_")},
        "mu_method": mu_method,
    }

    _log(f"Modo: {decisao['modo']} (stress={stress_now:.2f})")
    _log("Alocação recomendada agora:")
    for ticker, peso in decisao["pesos"].items():
        _log(f"  {ticker}: {peso:.1%}")

    return decisao


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu-method", default="tilt_linear", choices=["tilt_linear", "black_litterman"])
    parser.add_argument("--loop", type=int, default=0, help="segundos entre execuções (0 = roda só uma vez)")
    parser.add_argument("--json", default=None, help="caminho pra salvar a decisão em JSON (sobrescreve a cada execução)")
    args = parser.parse_args()

    while True:
        decisao = run_once(mu_method=args.mu_method)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(decisao, f, indent=2, ensure_ascii=False)
            _log(f"Salvo em {args.json}")

        if args.loop <= 0:
            break
        _log(f"Próxima execução em {args.loop}s... (Ctrl+C pra parar)\n")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
