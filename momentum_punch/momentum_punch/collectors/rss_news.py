"""
Coletor de notícias via RSS — grátis, sem chave, mas SEM histórico profundo:
cada execução só pega o que está no feed no momento (geralmente últimas ~20-50
notícias por veículo). Pra ter histórico, rode isso como job diário/periódico e
vá acumulando no CSV (a função save_incremental cuida de deduplicar por link).
"""
from __future__ import annotations

import datetime as dt

import feedparser
import pandas as pd

FEEDS = {
    "infomoney": "https://www.infomoney.com.br/feed/",
    "moneytimes": "https://www.moneytimes.com.br/feed/",
    "valor_economico": "https://valor.globo.com/rss/",
}


def fetch_feed(name: str, url: str) -> pd.DataFrame:
    parsed = feedparser.parse(url)
    rows = []
    for entry in parsed.entries:
        rows.append(
            {
                "fonte": name,
                "titulo": getattr(entry, "title", ""),
                "resumo": getattr(entry, "summary", ""),
                "link": getattr(entry, "link", ""),
                "publicado_em": getattr(entry, "published", None),
                "coletado_em": dt.datetime.now().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def fetch_all(feeds: dict[str, str] = FEEDS) -> pd.DataFrame:
    frames = []
    for name, url in feeds.items():
        try:
            df = fetch_feed(name, url)
            frames.append(df)
            print(f"[rss_news] {name}: {len(df)} itens")
        except Exception as exc:
            print(f"[rss_news] Falha ao ler {name} ({url}): {exc}")
    if not frames:
        return pd.DataFrame(columns=["fonte", "titulo", "resumo", "link", "publicado_em", "coletado_em"])
    return pd.concat(frames, ignore_index=True)


def save_incremental(new_df: pd.DataFrame, out_path: str = "data/raw/rss_news.csv") -> pd.DataFrame:
    """Acumula no CSV existente, deduplicando por link. Rode isso periodicamente
    (ex: cron diário) pra ir construindo histórico, já que RSS não tem backfill."""
    import os

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="link", keep="first")
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)
    return combined


if __name__ == "__main__":
    df = fetch_all()
    combined = save_incremental(df)
    print(f"[rss_news] Total acumulado: {len(combined)} notícias em data/raw/rss_news.csv")
