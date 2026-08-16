"""
Puxa histórico do GDELT de forma PACIENTE e RESUMÍVEL — pensado pra rodar em
background por um tempo longo (o rate limit deles é agressivo, não dá pra
apressar). Se cair no meio (rede, Ctrl+C, PC desligou), roda de novo e ele
retoma de onde parou, sem re-baixar o que já tem.

Diferença do gdelt.py original: aqui eu busco POR TEMA (estresse macro E os 4
temas por ticker — governança, ESG, energia, mercado amplo), não só estresse.
Isso desbloqueia sentiment POR TICKER com profundidade histórica de verdade,
não só o índice de estresse do circuit breaker.

O que foi corrigido depois de testar a API de verdade (o "0 artigos sem erro"
que a versão anterior produzia tinha DUAS causas, ambas na query):

  1. Termos justapostos ("Brasil sustentabilidade ESG") são AND implícito no
     GDELT — exige os três na mesma matéria, e quase nada bate. Agora cada
     tema é um grupo OR explícito.
  2. Os termos precisam vir ACENTUADOS ("governança", não "governanca"):
     testando o mesmo bloco de jun/2022, a versão sem acento devolveu 0 e a
     acentuada devolveu resultado.

Também confirmado no teste: `sourcelang:portuguese` funciona (o código de 3
letras não), o histórico alcança pelo menos 2022, e cada chamada é limitada a
250 registros — então um bloco mensal de tema amplo vem TRUNCADO em 250. Isso
é uma amostra do mês, não o censo do mês; está registrado no manifesto.

Uso:
    python gdelt_historical_pull.py --validar             # 1 bloco por tema, confere volume (~2 min)
    python gdelt_historical_pull.py --desde 2021-07-15    # coleta o histórico todo
    python gdelt_historical_pull.py --desde 2021-07-15    # rodar de novo continua de onde parou
    python gdelt_historical_pull.py --desde 2021-07-15 --resume-only   # só mostra o que falta

Saída: data/raw/gdelt_by_topic/{topico}.csv (um arquivo por tema)
       data/raw/gdelt_by_topic/_manifest.json (controle do que já foi baixado)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

from momentum_punch.collectors import gdelt

OUT_DIR = "data/raw/gdelt_by_topic"
MANIFEST_PATH = os.path.join(OUT_DIR, "_manifest.json")

# Uma query por tema, cada uma um grupo OR de termos ACENTUADOS. Ver o cabeçalho
# do arquivo pro motivo — justaposição vira AND e derruba o recall a zero.
TOPIC_QUERIES = {
    "stress": '(crise OR eleições OR conflito OR guerra OR "risco fiscal") sourcelang:portuguese',
    "ISUS11": '(sustentabilidade OR ESG OR "mudanças climáticas" OR emissões OR carbono) sourcelang:portuguese',
    "GOVE11": '(governança OR acionistas OR "conselho de administração" OR escândalo OR fraude) sourcelang:portuguese',
    "REVE11": '("energia renovável" OR eólica OR solar OR "transição energética") sourcelang:portuguese',
    "BOVA11": '(Ibovespa OR Bolsa OR Selic OR inflação OR economia) sourcelang:portuguese',
}


def _log(msg: str):
    """print com flush — sem isso a saída fica presa no buffer quando o script
    roda redirecionado pra arquivo (que é o caso normal, já que isso roda horas
    em background) e não dá pra acompanhar o progresso."""
    print(msg, flush=True)


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _month_chunks(desde: dt.date, ate: dt.date | None = None) -> list[tuple[dt.date, dt.date]]:
    ate = ate or dt.date.today()
    chunks = []
    current = desde
    while current < ate:
        chunk_end = min(current + dt.timedelta(days=30), ate)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def _baixa_bloco(topico: str, query: str, inicio: dt.date, fim: dt.date) -> int:
    """Baixa um bloco e mescla no CSV do tema. Devolve quantos artigos NOVOS
    entraram (não o acumulado — a versão anterior registrava o acumulado no
    manifesto, o que fazia todo bloco parecer produtivo)."""
    df = gdelt.fetch_articles(
        query,
        dt.datetime.combine(inicio, dt.time.min),
        dt.datetime.combine(fim, dt.time.max),
        max_records=250,
    )
    if df.empty:
        return 0

    out_csv = os.path.join(OUT_DIR, f"{topico}.csv")
    antes = 0
    if os.path.exists(out_csv):
        existente = pd.read_csv(out_csv)
        antes = len(existente)
        df = pd.concat([existente, df], ignore_index=True).drop_duplicates(subset="url")
    df.to_csv(out_csv, index=False)
    return len(df) - antes


def validar(sleep_seconds: float):
    """Puxa UM bloco recente por tema e reporta o volume. Serve pra não
    descobrir só depois de 2h de coleta que uma query está devolvendo zero."""
    fim = dt.date.today()
    inicio = fim - dt.timedelta(days=30)
    _log(f"[gdelt_historical] Validando as {len(TOPIC_QUERIES)} queries no bloco {inicio} a {fim}\n")

    problemas = []
    for topico, query in TOPIC_QUERIES.items():
        try:
            df = gdelt.fetch_articles(
                query,
                dt.datetime.combine(inicio, dt.time.min),
                dt.datetime.combine(fim, dt.time.max),
                max_records=250,
            )
            n = len(df)
            marca = "OK" if n >= 50 else ("MAGRO" if n > 0 else "ZERO")
            exemplo = f" | ex: {str(df.iloc[0]['titulo'])[:60]}" if n else ""
            _log(f"  {topico:8s} {marca:6s} {n:4d} artigos{exemplo}")
            if n < 50:
                problemas.append(topico)
        except Exception as exc:
            _log(f"  {topico:8s} FALHOU  {type(exc).__name__}: {str(exc)[:70]}")
            problemas.append(topico)
        time.sleep(sleep_seconds)

    if problemas:
        _log(f"\n[gdelt_historical] Revise a query destes temas antes da coleta longa: {problemas}")
    else:
        _log("\n[gdelt_historical] Todas as queries produzem volume. Pode rodar a coleta completa.")
    return not problemas


def run(desde: dt.date, sleep_seconds: float, resume_only: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = _load_manifest()
    chunks = _month_chunks(desde)

    plano = []
    for topico, query in TOPIC_QUERIES.items():
        for inicio, fim in chunks:
            chave = f"{topico}|{inicio.isoformat()}|{fim.isoformat()}"
            if chave not in manifest:
                plano.append((topico, query, inicio, fim, chave))

    _log(f"[gdelt_historical] janela: {desde} a {dt.date.today()} ({len(chunks)} blocos x {len(TOPIC_QUERIES)} temas)")
    _log(f"[gdelt_historical] {len(manifest)} bloco(s) já baixado(s) antes, {len(plano)} pendente(s)")

    if resume_only:
        for topico, _, inicio, fim, _ in plano:
            _log(f"  pendente: {topico} {inicio} a {fim}")
        return

    if not plano:
        _log("[gdelt_historical] Nada pendente, já tem tudo dessa janela.")
        return

    _log(f"[gdelt_historical] Tempo estimado: ~{len(plano) * sleep_seconds / 60:.0f} min (mais, se tomar 429 e precisar de backoff)")
    _log("[gdelt_historical] Pode interromper a qualquer momento e retomar rodando de novo.\n")

    t0 = time.time()
    for i, (topico, query, inicio, fim, chave) in enumerate(plano, 1):
        try:
            novos = _baixa_bloco(topico, query, inicio, fim)
            manifest[chave] = {
                "artigos_novos": novos,
                "quando": dt.datetime.now().isoformat(),
                # 250 é o teto da API: bloco que bate no teto veio TRUNCADO,
                # é amostra do período e não o total — fica registrado pra não
                # ser lido como cobertura completa depois
                "truncado_no_teto": novos >= 250,
            }
            _save_manifest(manifest)  # salva a cada bloco, resiliente a interrupção
            decorrido = time.time() - t0
            resta = (len(plano) - i) * decorrido / i / 60
            _log(f"[{i}/{len(plano)}] {topico} {inicio} a {fim}: +{novos} artigos (faltam ~{resta:.0f} min)")
        except Exception as exc:
            _log(f"[{i}/{len(plano)}] {topico} {inicio} a {fim}: FALHOU ({type(exc).__name__}: {str(exc)[:60]}) — não marcado, tenta de novo na próxima execução")

        time.sleep(sleep_seconds)

    _log("\n[gdelt_historical] Concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", default="2021-07-15",
                        help="data inicial (default: 2021-07-15, início de negociação do REVE11)")
    parser.add_argument("--sleep", type=float, default=15.0,
                        help="segundos entre chamadas. O GDELT pede 1 req/5s e devolve 429 mesmo com 6s — 15 é o que se mostrou estável")
    parser.add_argument("--resume-only", action="store_true", help="só mostra o que falta, não baixa nada")
    parser.add_argument("--validar", action="store_true", help="puxa 1 bloco por tema e confere o volume, sem coletar o histórico")
    args = parser.parse_args()

    if args.validar:
        sys.exit(0 if validar(args.sleep) else 1)

    run(dt.date.fromisoformat(args.desde), args.sleep, args.resume_only)
