"""
Manifesto de experimento — seção 8.2 do pré-relatório.

O documento se compromete com uma coisa forte: "cada execução deverá registrar
ID, período, fonte e hash dos dados, cutoff, universo, seed, commit, prompt,
modelo, temperatura, parâmetros, custos, status do solver, fallback, métricas,
arquivos de saída e categoria da evidência". Sem isso, nenhum número do
relatório é reproduzível — e um relatório que se apresenta como auditável e não
registra o que gerou os próprios números é pior que um que não promete nada.

O que este módulo garante na prática:

  * HASH DOS INSUMOS. Cada CSV de entrada entra com sha256. Se o preço for
    re-baixado e mudar (yfinance revisa histórico com mais frequência do que se
    imagina), o hash muda e a divergência aparece, em vez de virar um número
    diferente sem explicação.

  * COMMIT. O estado do código no momento da execução, incluindo se havia
    alteração não commitada — "rodei com a árvore suja" é informação, não
    detalhe.

  * PROMPT E MODELO. O texto exato do prompt entra por hash, junto com modelo e
    temperatura. Mudar uma palavra do prompt muda o score; sem o hash não há
    como saber depois qual versão produziu qual resultado.

  * CATEGORIA DA EVIDÊNCIA. O campo `categoria` obriga quem roda a declarar se
    aquilo é VERIFICADO, SINTÉTICO ou PENDENTE (Tabela 4), pra número de dado
    sintético não vazar pro relatório como se fosse resultado real.

O manifesto é gravado JUNTO com o resultado, não num log à parte, porque
resultado sem procedência não deveria circular sozinho.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess

from . import config

CATEGORIAS = {"VERIFICADO", "DERIVADO", "SINTETICO", "PENDENTE", "CONTRADITORIO"}


def hash_arquivo(caminho: str) -> str | None:
    """sha256 do arquivo, ou None se não existe (ausência é registrada como
    ausência, não como string vazia que passaria por hash válido)."""
    if not os.path.exists(caminho):
        return None
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except Exception:
        return None


def estado_do_codigo() -> dict:
    """Commit e limpeza da árvore. `arvore_suja=True` significa que o código em
    disco não corresponde exatamente ao commit registrado — o resultado não é
    reproduzível só pelo hash do commit."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "arvore_suja": bool(status) if status is not None else None,
        "arquivos_modificados": [l[3:] for l in status.splitlines()] if status else [],
    }


def parametros_do_modelo() -> dict:
    """Snapshot dos parâmetros que definem a decisão. Vem do config, não digitado
    à mão — número digitado no relatório é exatamente o que a seção 8.2 proíbe."""
    return {
        "universo": list(config.TICKERS),
        "ativo_livre_de_risco": config.RISK_FREE,
        "rebalanceamento": config.REBALANCE_FREQ,
        "janela_covariancia_dias": config.COV_LOOKBACK_DAYS,
        "aversao_ao_risco": config.RISK_AVERSION,
        "peso_maximo_por_etf": config.MAX_WEIGHT_PER_ETF,
        "peso_minimo_por_etf": config.MIN_WEIGHT_PER_ETF,
        "ema_halflife_dias": config.EMA_HALFLIFE_DAYS,
        "limiar_estresse": config.STRESS_THRESHOLD_Z,
        "exposicao_maxima_risk_off": config.RISK_OFF_MAX_EQUITY,
        "tilt_por_ticker": dict(config.SENTIMENT_TILT_STRENGTH_POR_TICKER),
        "tilt_base": config.SENTIMENT_TILT_STRENGTH_BASE,
        "corte_treino_teste": config.DATA_CORTE_TREINO_TESTE,
    }


def configuracao_do_llm() -> dict:
    """Modelo, temperatura e HASH do prompt. O prompt entra por hash e não por
    texto inteiro pra não inchar o manifesto — o texto vive versionado no git,
    e o hash liga um ao outro."""
    from . import sentiment

    return {
        "provider_estruturado": "ollama",
        "modelo": config.OLLAMA_MODEL,
        "temperatura": 0.1,  # fixada em sentiment._call_ollama
        "prompt_template_sha256": hash_texto(sentiment._STRUCTURED_SYSTEM_PROMPT_TEMPLATE),
        "modelo_finbert": config.FINBERT_PTBR_MODEL,
    }


def gerar(
    experimento: str,
    categoria: str,
    entradas: dict[str, str],
    parametros_execucao: dict,
    metricas: dict | None = None,
    saidas: list[str] | None = None,
    observacoes: str | None = None,
) -> dict:
    """
    experimento: identificador legível ("backtest_final_teste", "ablacao_A0_A12").
    categoria: uma de CATEGORIAS — obriga a declarar o tipo de evidência.
    entradas: {rótulo: caminho} dos arquivos consumidos; viram hash.
    parametros_execucao: o que variou nesta execução (período, custo, sinal...).
    """
    if categoria not in CATEGORIAS:
        raise ValueError(f"categoria deve ser uma de {sorted(CATEGORIAS)}, recebi {categoria!r}")

    return {
        "experimento": experimento,
        "categoria_da_evidencia": categoria,
        "quando": dt.datetime.now().isoformat(timespec="seconds"),
        "codigo": estado_do_codigo(),
        "ambiente": {"python": platform.python_version(), "so": platform.platform()},
        "entradas": {rotulo: {"caminho": c, "sha256": hash_arquivo(c)} for rotulo, c in entradas.items()},
        "parametros_do_modelo": parametros_do_modelo(),
        "parametros_da_execucao": parametros_execucao,
        "llm": configuracao_do_llm(),
        "metricas": metricas or {},
        "saidas": saidas or [],
        "observacoes": observacoes,
    }


def salvar(manifesto: dict, caminho: str) -> str:
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, indent=2, ensure_ascii=False)
    print(f"[manifesto] {manifesto['experimento']} ({manifesto['categoria_da_evidencia']}) -> {caminho}")
    return caminho
