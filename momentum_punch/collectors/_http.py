"""Helper de retry com backoff exponencial pras APIs públicas que usamos
(Bacen, GDELT) — todas têm erro transitório ocasional (502, 429) que não
significa que o request está errado, só que o servidor engasgou."""
from __future__ import annotations

import time

import requests


def get_with_retry(
    url: str,
    params: dict | None = None,
    max_retries: int = 4,
    backoff_seconds: float = 5.0,
    timeout: int = 30,
) -> requests.Response:
    """
    Tenta o GET várias vezes com backoff exponencial em erros transitórios
    (429 rate limit, 500/502/503/504 erro de servidor). Erros 4xx que não sejam
    429 (ex: 400 Bad Request, 403 Forbidden — query malformada ou bloqueio
    permanente) propagam IMEDIATAMENTE na primeira tentativa, sem retry — tentar
    de novo não muda um erro de sintaxe ou bloqueio, só desperdiça tempo.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            # erro de rede/conexão (não chegou a ter resposta HTTP) — esse sim vale retry
            last_exc = exc
            wait = backoff_seconds * (2 ** attempt)
            print(f"[http_retry] erro de conexão em {url} — tentativa {attempt + 1}/{max_retries}, esperando {wait:.0f}s: {exc}")
            time.sleep(wait)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = backoff_seconds * (2 ** attempt)
            print(f"[http_retry] {resp.status_code} em {url} — tentativa {attempt + 1}/{max_retries}, esperando {wait:.0f}s")
            time.sleep(wait)
            last_exc = requests.HTTPError(f"{resp.status_code} após {attempt + 1} tentativas")
            continue

        # qualquer outro status (incluindo erro 4xx permanente) — não insiste, propaga já
        resp.raise_for_status()
        return resp

    raise last_exc or RuntimeError(f"Falha desconhecida ao buscar {url}")
