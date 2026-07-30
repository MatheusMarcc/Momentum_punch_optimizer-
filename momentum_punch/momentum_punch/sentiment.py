"""
Bloco 2 e 3 do pipeline: Score LLM + Suavização EMA.

Providers suportados (nenhum pago):
  - "ollama": modelo local via Ollama (http://localhost:11434), zero custo,
    sem rate limit — ideal pro backtest histórico com muitas chamadas.
  - "groq": API gratuita (console.groq.com), sem cartão, ~30 req/min —
    ideal pra validação pontual/demo com modelo maior sem depender da sua GPU.

Troque o provider em config.SENTIMENT_PROVIDER.

- score_texts_for_ticker(): manda o texto bruto (notícias/tweets) do dia pra LLM
  e recebe um Sentiment Score de -1 (pânico) a +1 (euforia) por ticker.
- score_stress_index(): mesma ideia, mas para o índice de estresse geopolítico
  usado no circuit breaker (0 = calmo, 1 = crise).
- apply_ema(): suaviza a série diária de scores em Sentiment Alpha Score (EMA).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd
import requests

from . import config


@dataclass
class SentimentResult:
    ticker: str
    score: float
    rationale: str


_SENTIMENT_SYSTEM_PROMPT = """\
Você é um analista quantitativo de sentimento de mercado. Dado um conjunto de \
manchetes de notícias e/ou tweets referentes a um ativo, retorne SOMENTE um JSON \
válido (sem markdown, sem texto adicional) no formato:

{"score": <float entre -1.0 e 1.0>, "rationale": "<justificativa em até 20 palavras>"}

Onde -1.0 = pânico/pessimismo extremo, 0.0 = neutro, +1.0 = euforia/otimismo extremo.
Baseie-se apenas no conteúdo fornecido, sem inventar fatos.
"""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # fallback: alguns modelos locais menores adicionam texto antes/depois do JSON
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return json.loads(raw)


def _call_ollama(system: str, user: str, host: str = config.OLLAMA_HOST, model: str = config.OLLAMA_MODEL) -> str:
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",  # força o Ollama a devolver JSON válido
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_groq(system: str, user: str, model: str = config.GROQ_MODEL) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Defina a variável de ambiente GROQ_API_KEY (grátis em console.groq.com).")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_llm(system: str, user: str, provider: str | None = None) -> str:
    provider = provider or config.SENTIMENT_PROVIDER
    if provider == "ollama":
        return _call_ollama(system, user)
    elif provider == "groq":
        return _call_groq(system, user)
    else:
        raise ValueError(f"Provider desconhecido: {provider}. Use 'ollama' ou 'groq'.")


def score_texts_for_ticker(ticker: str, texts: list[str], provider: str | None = None) -> SentimentResult:
    """Envia os textos do dia para a LLM e retorna o Sentiment Score do ticker."""
    theme = config.TICKER_THEMES.get(ticker, ticker)
    joined = "\n".join(f"- {t}" for t in texts) if texts else "(nenhum texto coletado hoje)"

    user_prompt = (
        f"Ativo: {ticker}\n"
        f"Temas relevantes: {theme}\n\n"
        f"Textos do dia:\n{joined}\n\n"
        "Retorne o JSON de sentimento para este ativo."
    )

    raw = _call_llm(_SENTIMENT_SYSTEM_PROMPT, user_prompt, provider=provider)
    parsed = _extract_json(raw)
    score = float(parsed["score"])
    score = max(config.SENTIMENT_MIN, min(config.SENTIMENT_MAX, score))
    return SentimentResult(ticker=ticker, score=score, rationale=parsed.get("rationale", ""))


def score_stress_index(texts: list[str], provider: str | None = None) -> float:
    """Retorna o Índice de Estresse/Incerteza (0 a 1) usado no circuit breaker."""
    joined = "\n".join(f"- {t}" for t in texts) if texts else "(nenhum texto coletado hoje)"

    system = (
        "Você mede o nível de estresse/incerteza macro-geopolítica em textos "
        f"sobre {config.STRESS_THEMES}. Retorne SOMENTE um JSON: "
        '{"stress": <float entre 0.0 e 1.0>} onde 0.0 = calmo e 1.0 = crise severa.'
    )
    raw = _call_llm(system, f"Textos do dia:\n{joined}", provider=provider)
    parsed = _extract_json(raw)
    return max(0.0, min(1.0, float(parsed["stress"])))


def apply_ema(daily_scores: pd.Series, span: int = config.EMA_SPAN) -> pd.Series:
    """Converte scores diários brutos em Sentiment Alpha Score (EMA)."""
    return daily_scores.ewm(span=span, adjust=False).mean()


def score_history(
    texts_by_date_ticker: dict[str, dict[str, list[str]]],
    tickers: list[str] = config.TICKERS,
    provider: str | None = None,
) -> pd.DataFrame:
    """
    Roda o scoring da LLM sobre um histórico {data: {ticker: [textos]}} e devolve
    um DataFrame (index=data, colunas=tickers) já suavizado por EMA.
    Útil para o backtest com dados históricos reais.

    Com provider="ollama" (default): sem rate limit, mas mais lento por chamada
    numa 4GB — pra um backtest grande, rode em background e vá salvando o
    resultado incremental (ver score_history_incremental abaixo).
    """
    rows = {}
    for date, per_ticker_texts in sorted(texts_by_date_ticker.items()):
        row = {}
        for ticker in tickers:
            texts = per_ticker_texts.get(ticker, [])
            row[ticker] = score_texts_for_ticker(ticker, texts, provider=provider).score
        rows[date] = row
        print(f"[sentiment] {date}: {row}")
    raw_df = pd.DataFrame.from_dict(rows, orient="index")
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()
    return raw_df.apply(apply_ema)
