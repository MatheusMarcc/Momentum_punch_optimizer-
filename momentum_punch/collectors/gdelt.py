"""
Coletor GDELT 2.0 DOC API — grátis, sem chave, com histórico (o que o RSS não tem).
Documentação: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Bom principalmente pros temas de ESTRESSE MACRO/GEOPOLÍTICO (eleições, conflitos,
choques globais) porque a cobertura de veículo BR específico é mais fraca que a de
imprensa internacional — pro sentiment por ticker, prefira RSS+CVM.

Limite prático: a API cobre só ~últimos 3 meses em detalhe fino de free-text;
pra período mais longo, use a busca por blocos mensais (loop de queries).
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from ._http import get_with_retry

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_articles(
    query: str,
    start: dt.datetime,
    end: dt.datetime,
    max_records: int = 250,
    language: str = "portuguese",
) -> pd.DataFrame:
    """
    query: termos de busca, ex: 'Brasil eleição OR "crise cambial"'.
    start/end: janela de tempo (a API aceita no máximo alguns meses por chamada).
    """
    params = {
        "query": f"{query} sourcelang:{language}",
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    # a API do GDELT tem rate limit bem agressivo (429 fácil) — backoff mais longo aqui
    resp = get_with_retry(BASE_URL, params=params, max_retries=5, backoff_seconds=15.0, timeout=30)
    data = resp.json()
    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame(columns=["titulo", "url", "fonte", "data", "tom"])
    df = pd.DataFrame(articles)
    df = df.rename(columns={"title": "titulo", "url": "url", "domain": "fonte", "seendate": "data", "tone": "tom"})
    df["data"] = pd.to_datetime(df["data"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    return df[["titulo", "url", "fonte", "data", "tom"]]


def fetch_range_in_monthly_chunks(
    query: str,
    data_inicial: dt.date,
    data_final: dt.date,
    max_records_per_chunk: int = 250,
    sleep_seconds: float = 20.0,
) -> pd.DataFrame:
    """Quebra o intervalo em blocos mensais pra contornar o limite prático da API
    e evita rate limit com um sleep entre chamadas."""
    frames = []
    current = data_inicial
    while current < data_final:
        chunk_end = min(current + dt.timedelta(days=30), data_final)
        try:
            df = fetch_articles(
                query,
                dt.datetime.combine(current, dt.time.min),
                dt.datetime.combine(chunk_end, dt.time.max),
                max_records=max_records_per_chunk,
            )
            frames.append(df)
            print(f"[gdelt] {current} a {chunk_end}: {len(df)} artigos")
        except Exception as exc:
            print(f"[gdelt] Falha no bloco {current} a {chunk_end}: {exc}")
        current = chunk_end
        time.sleep(sleep_seconds)
    if not frames:
        return pd.DataFrame(columns=["titulo", "url", "fonte", "data", "tom"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="url")


if __name__ == "__main__":
    hoje = dt.date.today()
    inicio = hoje - dt.timedelta(days=90)

    # exemplo: cobrir os temas de estresse macro/geopolítico do config.STRESS_THEMES
    query = 'Brasil economia'
    df = fetch_range_in_monthly_chunks(query, inicio, hoje)
    out_path = "data/raw/gdelt_stress.csv"
    df.to_csv(out_path, index=False)
    print(f"[gdelt] Salvo em {out_path} ({len(df)} artigos)")
