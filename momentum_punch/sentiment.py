"""
Bloco 2 e 3 do pipeline: Score de Sentimento + Suavização EMA.

Sentiment POR TICKER (alimenta o tilt do Markowitz): via FinBERT-PT-BR
(lucas-leme/FinBERT-PT-BR), classificador BERT especializado em sentimento
financeiro em português, treinado sobre BERTimbau com 1.4M notícias
financeiras PT-BR + paper publicado. Determinístico, roda local (CPU/GPU).

Índice de estresse geopolítico (score_stress_index, usado com GDELT): continua
via LLM (Ollama/Groq) porque é uma tarefa de julgamento sobre contexto
macro/geopolítico, não classificação de tom positivo/negativo — o FinBERT-PT-BR
não serve pra isso. Na prática esse caminho está DORMENTE no backtest atual:
o circuit breaker em produção usa o stress index objetivo do Bacen Focus
(build_stress_index_focus.py), não este aqui — GDELT foi abandonado por rate
limit não confiável. Mantido só porque run_diagnostics.py e
build_sentiment_dataset.py --stress ainda referenciam essa função.
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


# ---------------------------------------------------------------------------
# Sentiment por ticker (o que alimenta o Markowitz) — FinBERT-PT-BR
# ---------------------------------------------------------------------------

_finbert_tokenizer = None
_finbert_model = None
_finbert_id2label = None


def _load_finbert_ptbr():
    """Carrega o FinBERT-PT-BR uma única vez (singleton em memória do processo)."""
    global _finbert_tokenizer, _finbert_model, _finbert_id2label
    if _finbert_model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = config.FINBERT_PTBR_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[sentiment] carregando {model_name} em {device} (só na primeira chamada)...")
    _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    _finbert_model.eval()
    _finbert_id2label = _finbert_model.config.id2label
    print(f"[sentiment] FinBERT-PT-BR carregado. labels: {_finbert_id2label}")


def score_texts_for_ticker(ticker: str, texts: list[str], **_ignored) -> SentimentResult:
    """
    Classifica os textos do dia (positivo/negativo/neutro) via FinBERT-PT-BR e
    agrega em um único Sentiment Score -1 (pessimismo) a +1 (otimismo) para o
    ticker, via média de (P(positivo) - P(negativo)) por texto.

    Sem texto no dia -> score neutro (0.0).
    **_ignored existe só por compatibilidade de assinatura com chamadas antigas
    que passavam provider=/max_retries= — não fazem mais sentido aqui.
    """
    if not texts:
        return SentimentResult(ticker=ticker, score=0.0, rationale="sem texto coletado no dia")

    import torch
    import torch.nn.functional as F

    _load_finbert_ptbr()

    device = next(_finbert_model.parameters()).device
    inputs = _finbert_tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        logits = _finbert_model(**inputs).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    per_text_scores = []
    for row in probs:
        label_probs = {_finbert_id2label[i]: float(p) for i, p in enumerate(row)}
        pos = label_probs.get("POSITIVE", label_probs.get("Positive", label_probs.get("positive", 0.0)))
        neg = label_probs.get("NEGATIVE", label_probs.get("Negative", label_probs.get("negative", 0.0)))
        per_text_scores.append(pos - neg)

    score = sum(per_text_scores) / len(per_text_scores)
    score = max(config.SENTIMENT_MIN, min(config.SENTIMENT_MAX, score))
    rationale = f"FinBERT-PT-BR: média de {len(texts)} texto(s), score bruto {score:+.3f}"
    return SentimentResult(ticker=ticker, score=score, rationale=rationale)


# ---------------------------------------------------------------------------
# Índice de estresse geopolítico (GDELT) — DORMENTE, mantido só por
# compatibilidade com build_sentiment_dataset.py --stress e run_diagnostics.py
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
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
            "format": "json",
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


def score_stress_index(texts: list[str], provider: str = "ollama", max_retries: int = 2) -> float:
    """Retorna o Índice de Estresse/Incerteza (0 a 1) usado no circuit breaker via GDELT.
    DORMENTE no backtest de produção — ver nota no topo do arquivo."""
    joined = "\n".join(f"- {t}" for t in texts) if texts else "(nenhum texto coletado hoje)"

    system = (
        "Você mede o nível de estresse/incerteza macro-geopolítica em textos "
        f"sobre {config.STRESS_THEMES}. Retorne SOMENTE um JSON: "
        '{"stress": <float entre 0.0 e 1.0>} onde 0.0 = calmo e 1.0 = crise severa.'
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = _call_ollama(system, f"Textos do dia:\n{joined}") if provider == "ollama" \
                else _call_groq(system, f"Textos do dia:\n{joined}")
            parsed = _extract_json(raw)
            raw_stress = parsed.get("stress", parsed.get("score", parsed.get("valor")))
            if raw_stress is None:
                raise KeyError(f"resposta sem campo de stress reconhecível: {parsed}")
            return max(0.0, min(1.0, float(raw_stress)))
        except Exception as exc:
            last_error = exc
            print(f"[sentiment] stress_index: tentativa {attempt + 1}/{max_retries + 1} falhou ({exc})")

    print(f"[sentiment] stress_index: todas as tentativas falharam, usando 0.0")
    return 0.0


# ---------------------------------------------------------------------------
# Suavização e histórico
# ---------------------------------------------------------------------------

def apply_ema(daily_scores: pd.Series, span: int = config.EMA_SPAN) -> pd.Series:
    """Converte scores diários brutos em Sentiment Alpha Score (EMA)."""
    return daily_scores.ewm(span=span, adjust=False).mean()


def score_history(
    texts_by_date_ticker: dict[str, dict[str, list[str]]],
    tickers: list[str] = config.TICKERS,
) -> pd.DataFrame:
    """Roda o scoring (FinBERT-PT-BR) sobre um histórico {data: {ticker: [textos]}}
    e devolve um DataFrame (index=data, colunas=tickers) já suavizado por EMA."""
    rows = {}
    for date, per_ticker_texts in sorted(texts_by_date_ticker.items()):
        row = {}
        for ticker in tickers:
            texts = per_ticker_texts.get(ticker, [])
            row[ticker] = score_texts_for_ticker(ticker, texts).score
        rows[date] = row
        print(f"[sentiment] {date}: {row}")
    raw_df = pd.DataFrame.from_dict(rows, orient="index")
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()
    return raw_df.apply(apply_ema)
