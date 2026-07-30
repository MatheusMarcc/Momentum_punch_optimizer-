"""
Bloco 1 do pipeline: Coleta.

Duas fontes previstas no deck:
  1) Notícias de portais financeiros -> feeds RSS (grátis, sem necessidade de chave).
  2) Tweets de contas financeiras influentes -> API do X/Twitter (paga, precisa de
     Bearer Token). Implementado como função opcional que degrada graciosamente
     se não houver credencial.

Ambas retornam listas de textos curtos (headline/tweet), que depois vão para
sentiment.py ser pontuados pela LLM.
"""
from __future__ import annotations

import datetime as dt
import os

import feedparser

# Feeds RSS de notícias financeiras BR (ajuste/complete com os que você quiser cobrir)
NEWS_FEEDS = [
    "https://www.infomoney.com.br/feed/",
    "https://valor.globo.com/rss/",
    "https://www.moneytimes.com.br/feed/",
]

# Contas financeiras influentes a monitorar no X/Twitter (handles, sem @)
TWITTER_ACCOUNTS = [
    "b3", "ibovespa", "infomoney", "valoreconomico",
]


def fetch_news_headlines(feeds: list[str] = NEWS_FEEDS, max_per_feed: int = 15) -> list[str]:
    """Baixa manchetes recentes dos feeds RSS configurados."""
    headlines: list[str] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:  # rede instável, feed fora do ar, etc.
            print(f"[data_collection] Falha ao ler feed {url}: {exc}")
            continue
        for entry in parsed.entries[:max_per_feed]:
            title = getattr(entry, "title", None)
            if title:
                headlines.append(title.strip())
    return headlines


def fetch_tweets(accounts: list[str] = TWITTER_ACCOUNTS, max_per_account: int = 10) -> list[str]:
    """
    Busca tweets recentes das contas monitoradas via API v2 do X.
    Requer a variável de ambiente X_BEARER_TOKEN. Se ausente, retorna lista vazia
    (o pipeline segue funcionando só com notícias).
    """
    bearer_token = os.environ.get("X_BEARER_TOKEN")
    if not bearer_token:
        print("[data_collection] X_BEARER_TOKEN não definido — pulando coleta de tweets.")
        return []

    import requests  # import local: só precisa se formos realmente chamar a API

    tweets: list[str] = []
    headers = {"Authorization": f"Bearer {bearer_token}"}
    for handle in accounts:
        try:
            user_resp = requests.get(
                f"https://api.twitter.com/2/users/by/username/{handle}",
                headers=headers, timeout=10,
            )
            user_resp.raise_for_status()
            user_id = user_resp.json()["data"]["id"]

            tl_resp = requests.get(
                f"https://api.twitter.com/2/users/{user_id}/tweets",
                headers=headers,
                params={"max_results": max_per_account, "exclude": "retweets,replies"},
                timeout=10,
            )
            tl_resp.raise_for_status()
            for tweet in tl_resp.json().get("data", []):
                tweets.append(tweet["text"])
        except Exception as exc:
            print(f"[data_collection] Falha ao buscar tweets de @{handle}: {exc}")
    return tweets


def collect_daily_texts(date: dt.date | None = None) -> dict[str, list[str]]:
    """
    Ponto de entrada usado pelo job diário (08:00 no deck): junta notícias + tweets
    do dia. A separação por ticker acontece depois, em sentiment.py, mandando o
    mesmo pool de texto para a LLM com o tema de cada ETF — ou, se você quiser
    filtrar por keyword antes de chamar a LLM (economiza tokens), adicione um
    filtro simples aqui usando TICKER_THEMES do config.
    """
    date = date or dt.date.today()
    texts = fetch_news_headlines() + fetch_tweets()
    print(f"[data_collection] {date}: {len(texts)} textos coletados.")
    return {"all": texts}
