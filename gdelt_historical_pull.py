"""
Puxa histórico do GDELT de forma PACIENTE e RESUMÍVEL — pensado pra rodar em
background por um tempo longo (o rate limit deles é agressivo, não dá pra
apressar). Se cair no meio (rede, Ctrl+C, PC desligou), roda de novo e ele
retoma de onde parou, sem re-baixar o que já tem.

Diferença do gdelt.py original: aqui eu busco POR TEMA (estresse macro E os 4
temas por ticker — governança, ESG, energia, mercado amplo), não só estresse.
Isso desbloqueia sentiment POR TICKER com profundidade histórica de verdade,
não só o índice de estresse do circuit breaker.

Uso:
    python gdelt_historical_pull.py --months 6      # últimos 6 meses (recomendado pra começar)
    python gdelt_historical_pull.py --months 6       # rodar de novo continua de onde parou
    python gdelt_historical_pull.py --months 6 --resume-only   # só mostra o que falta, não baixa

Saída: data/raw/gdelt_by_topic/{topico}.csv (um arquivo por tema)
       data/raw/gdelt_by_topic/_manifest.json (controle do que já foi baixado)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

import pandas as pd

from momentum_punch import config
from momentum_punch.collectors import gdelt

OUT_DIR = "data/raw/gdelt_by_topic"
MANIFEST_PATH = os.path.join(OUT_DIR, "_manifest.json")

# Query por tema — termos mais simples que a versão original (menos operadores
# OR encadeados), porque suspeito que a query complexa demais tenha contribuído
# pros "0 artigos sem erro" que apareceram antes. Query simples, uma por tema.
TOPIC_QUERIES = {
    "stress": "Brasil crise OR eleição OR conflito",
    "ISUS11": "Brasil sustentabilidade ESG",
    "GOVE11": "Brasil governança corporativa escândalo",
    "REVE11": "Brasil energia renovável",
    "BOVA11": "Brasil Ibovespa economia",
}


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _month_chunks(months_back: int) -> list[tuple[dt.date, dt.date]]:
    hoje = dt.date.today()
    chunks = []
    current = hoje - dt.timedelta(days=30 * months_back)
    while current < hoje:
        chunk_end = min(current + dt.timedelta(days=30), hoje)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def run(months_back: int, sleep_seconds: float, resume_only: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = _load_manifest()
    chunks = _month_chunks(months_back)

    total_pendente = 0
    plano = []
    for topico, query in TOPIC_QUERIES.items():
        for inicio, fim in chunks:
            chave = f"{topico}|{inicio.isoformat()}|{fim.isoformat()}"
            if chave not in manifest:
                plano.append((topico, query, inicio, fim, chave))
                total_pendente += 1

    print(f"[gdelt_historical] {len(manifest)} bloco(s) já baixado(s) antes, {total_pendente} pendente(s)")
    if resume_only:
        for topico, _, inicio, fim, _ in plano:
            print(f"  pendente: {topico} {inicio} a {fim}")
        return

    if total_pendente == 0:
        print("[gdelt_historical] Nada pendente, já tem tudo dessa janela.")
        return

    tempo_estimado_min = total_pendente * sleep_seconds / 60
    print(f"[gdelt_historical] Tempo estimado: ~{tempo_estimado_min:.0f} min (pode ser mais, com retries)")
    print("[gdelt_historical] Pode interromper com Ctrl+C a qualquer momento e retomar depois rodando de novo.\n")

    for i, (topico, query, inicio, fim, chave) in enumerate(plano, 1):
        print(f"[{i}/{total_pendente}] {topico}: {inicio} a {fim}")
        try:
            df = gdelt.fetch_articles(
                query,
                dt.datetime.combine(inicio, dt.time.min),
                dt.datetime.combine(fim, dt.time.max),
                max_records=250,
            )
            out_csv = os.path.join(OUT_DIR, f"{topico}.csv")
            if os.path.exists(out_csv) and len(df) > 0:
                existente = pd.read_csv(out_csv)
                df = pd.concat([existente, df], ignore_index=True).drop_duplicates(subset="url")
            if len(df) > 0:
                df.to_csv(out_csv, index=False)
            print(f"    -> {len(df)} artigos acumulados em {topico}.csv")

            manifest[chave] = {"artigos": len(df), "quando": dt.datetime.now().isoformat()}
            _save_manifest(manifest)  # salva a cada bloco, não só no fim — resiliente a interrupção
        except Exception as exc:
            print(f"    FALHOU: {exc} (não marcado como feito, vai tentar de novo na próxima execução)")

        time.sleep(sleep_seconds)

    print("\n[gdelt_historical] Concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6, help="quantos meses pra trás cobrir")
    parser.add_argument("--sleep", type=float, default=25.0, help="segundos entre chamadas (aumente se continuar tomando 429)")
    parser.add_argument("--resume-only", action="store_true", help="só mostra o que falta, não baixa nada")
    args = parser.parse_args()

    run(args.months, args.sleep, args.resume_only)
