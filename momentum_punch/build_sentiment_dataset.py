"""
Pega o texto bruto já coletado (data/raw/rss_news.csv) e manda pro Ollama via
sentiment.py, gerando o Sentiment Alpha Score por data/ticker.

Fluxo:
  1. python collect_data.py --only rss_news      (gera data/raw/rss_news.csv)
  2. ollama pull qwen2.5:3b-instruct-q4_K_M       (uma vez só)
  3. python build_sentiment_dataset.py            (este script)

Saída: data/processed/sentiment_scores.csv (index=data, colunas=tickers, já em EMA)

Observação sobre RSS: como não tem backfill, na prática isso vai te dar só
UM dia de score de cada vez (o dia em que você rodou o collect_data.py).
Pra construir histórico, rode collect_data.py + build_sentiment_dataset.py
periodicamente (ex: cron diário) e deixe acumular — ou use GDELT como fonte
de texto histórico pro backtest (ver momentum_punch/collectors/gdelt.py).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from momentum_punch import config, sentiment


def load_daily_texts(rss_csv: str = "data/raw/rss_news.csv") -> dict[str, dict[str, list[str]]]:
    """
    Lê o CSV de notícias e agrupa por data de publicação. Como ainda não temos
    filtro por ticker (a LLM recebe o pool inteiro de manchetes do dia + o tema
    do ticker, e julga relevância pelo prompt), cada ticker recebe o mesmo
    conjunto de textos daquele dia.
    """
    if not os.path.exists(rss_csv):
        raise FileNotFoundError(
            f"{rss_csv} não existe. Rode antes: python collect_data.py --only rss_news"
        )

    df = pd.read_csv(rss_csv)
    df["publicado_em"] = pd.to_datetime(df["publicado_em"], errors="coerce", utc=True)
    df = df.dropna(subset=["publicado_em"])
    df["data"] = df["publicado_em"].dt.date.astype(str)

    texts_by_date_ticker: dict[str, dict[str, list[str]]] = {}
    for date, group in df.groupby("data"):
        textos_do_dia = (group["titulo"].fillna("") + ". " + group["resumo"].fillna("")).tolist()
        texts_by_date_ticker[date] = {ticker: textos_do_dia for ticker in config.TICKERS}

    return texts_by_date_ticker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=config.SENTIMENT_PROVIDER, choices=["ollama", "groq"])
    parser.add_argument("--rss-csv", default="data/raw/rss_news.csv")
    parser.add_argument("--out", default="data/processed/sentiment_scores.csv")
    args = parser.parse_args()

    texts_by_date_ticker = load_daily_texts(args.rss_csv)
    n_dates = len(texts_by_date_ticker)
    print(f"[build_sentiment_dataset] {n_dates} data(s) encontrada(s) no CSV, usando provider={args.provider}")

    if n_dates == 0:
        print("[build_sentiment_dataset] Nenhuma data válida encontrada, encerrando.")
        return

    scores_df = sentiment.score_history(texts_by_date_ticker, provider=args.provider)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scores_df.to_csv(args.out)
    print(f"\n[build_sentiment_dataset] Salvo em {args.out}")
    print(scores_df.tail())


if __name__ == "__main__":
    main()
