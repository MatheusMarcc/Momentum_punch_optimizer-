"""
Coletor Reddit via PRAW, free tier (uso não-comercial, 100 QPM).

Setup (grátis, ~2 minutos):
  1. Crie um app em https://www.reddit.com/prefs/apps (tipo "script")
  2. Anote client_id e client_secret
  3. Defina as variáveis de ambiente:
       export REDDIT_CLIENT_ID="..."
       export REDDIT_CLIENT_SECRET="..."
       export REDDIT_USER_AGENT="momentum_punch/0.1 by u/seu_usuario"

pip install praw
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd

try:
    import praw
except ImportError:
    praw = None

SUBREDDITS = ["investimentos", "farialimabets", "BrasilInvest"]


def _get_client():
    if praw is None:
        raise RuntimeError("Pacote 'praw' não instalado. Rode: pip install praw")
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "momentum_punch/0.1")
    if not client_id or not client_secret:
        raise RuntimeError("Defina REDDIT_CLIENT_ID e REDDIT_CLIENT_SECRET.")
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def fetch_subreddit_posts(
    subreddit_name: str,
    limit: int = 100,
    client=None,
) -> pd.DataFrame:
    client = client or _get_client()
    subreddit = client.subreddit(subreddit_name)
    rows = []
    for post in subreddit.new(limit=limit):
        rows.append(
            {
                "subreddit": subreddit_name,
                "titulo": post.title,
                "texto": post.selftext,
                "score": post.score,
                "num_comentarios": post.num_comments,
                "criado_em": dt.datetime.fromtimestamp(post.created_utc).isoformat(),
                "url": post.url,
            }
        )
    return pd.DataFrame(rows)


def fetch_all(subreddits: list[str] = SUBREDDITS, limit_per_sub: int = 100) -> pd.DataFrame:
    client = _get_client()
    frames = []
    for name in subreddits:
        try:
            df = fetch_subreddit_posts(name, limit=limit_per_sub, client=client)
            frames.append(df)
            print(f"[reddit] r/{name}: {len(df)} posts")
        except Exception as exc:
            print(f"[reddit] Falha em r/{name}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["subreddit", "titulo", "texto", "score", "num_comentarios", "criado_em", "url"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = fetch_all()
    out_path = "data/raw/reddit_posts.csv"
    df.to_csv(out_path, index=False)
    print(f"[reddit] Salvo em {out_path} ({len(df)} posts)")
