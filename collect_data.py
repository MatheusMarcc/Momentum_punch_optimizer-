"""
Roda todos os coletores e salva CSVs em data/raw/. Cada coletor é independente —
se um falhar (rede, credencial faltando, endpoint mudou), os outros continuam.

Uso:
    python collect_data.py                  # roda todos
    python collect_data.py --only bacen_sgs,rss_news   # roda só alguns

Setup de credenciais (nenhuma é obrigatória — sem elas, o coletor correspondente
é pulado):
    export REDDIT_CLIENT_ID="..."
    export REDDIT_CLIENT_SECRET="..."
    export REDDIT_USER_AGENT="momentum_punch/0.1 by u/seu_usuario"

Bacen (SGS e Focus), RSS e GDELT não precisam de credencial.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import traceback

os.makedirs("data/raw", exist_ok=True)


def run_bacen_sgs():
    from momentum_punch.collectors import bacen_sgs

    hoje = dt.date.today()
    inicio = (hoje - dt.timedelta(days=365 * 3)).strftime("%d/%m/%Y")
    fim = hoje.strftime("%d/%m/%Y")
    df = bacen_sgs.fetch_all_series(inicio, fim)
    df.to_csv("data/raw/bacen_sgs.csv")
    print(f"[collect_data] bacen_sgs -> data/raw/bacen_sgs.csv ({len(df)} linhas)")


def run_bacen_focus():
    from momentum_punch.collectors import bacen_focus

    hoje = dt.date.today()
    inicio = (hoje - dt.timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")
    df = bacen_focus.fetch_focus_stress_inputs(inicio, fim)
    df.to_csv("data/raw/bacen_focus.csv")
    print(f"[collect_data] bacen_focus -> data/raw/bacen_focus.csv ({len(df)} linhas)")


def run_rss_news():
    from momentum_punch.collectors import rss_news

    df = rss_news.fetch_all()
    combined = rss_news.save_incremental(df)
    print(f"[collect_data] rss_news -> data/raw/rss_news.csv ({len(combined)} linhas acumuladas)")


def run_gdelt():
    from momentum_punch.collectors import gdelt

    hoje = dt.date.today()
    inicio = hoje - dt.timedelta(days=90)
    query = 'Brasil (eleição OR "conflito armado" OR "crise cambial" OR "risco fiscal")'
    df = gdelt.fetch_range_in_monthly_chunks(query, inicio, hoje)
    df.to_csv("data/raw/gdelt_stress.csv", index=False)
    print(f"[collect_data] gdelt -> data/raw/gdelt_stress.csv ({len(df)} linhas)")


def run_reddit():
    from momentum_punch.collectors import reddit_collector

    df = reddit_collector.fetch_all()
    df.to_csv("data/raw/reddit_posts.csv", index=False)
    print(f"[collect_data] reddit -> data/raw/reddit_posts.csv ({len(df)} linhas)")


def run_cvm():
    from momentum_punch.collectors import cvm_fatos_relevantes

    ano_atual = dt.date.today().year
    df = cvm_fatos_relevantes.fetch_fatos_relevantes(anos=[ano_atual - 1, ano_atual])
    df.to_csv("data/raw/cvm_fatos_relevantes.csv", index=False)
    print(f"[collect_data] cvm -> data/raw/cvm_fatos_relevantes.csv ({len(df)} linhas)")


COLLECTORS = {
    "bacen_sgs": run_bacen_sgs,
    "bacen_focus": run_bacen_focus,
    "rss_news": run_rss_news,
    "gdelt": run_gdelt,
    "reddit": run_reddit,
    "cvm": run_cvm,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Lista separada por vírgula dos coletores a rodar (padrão: todos). "
        f"Opções: {', '.join(COLLECTORS)}",
    )
    args = parser.parse_args()

    names = args.only.split(",") if args.only else list(COLLECTORS.keys())

    print(f"[collect_data] Rodando: {', '.join(names)}\n")
    resultados = {"ok": [], "falhou": []}
    for name in names:
        if name not in COLLECTORS:
            print(f"[collect_data] Coletor desconhecido: {name}, pulando.")
            continue
        print(f"\n--- {name} ---")
        try:
            COLLECTORS[name]()
            resultados["ok"].append(name)
        except Exception:
            print(f"[collect_data] FALHOU: {name}")
            traceback.print_exc()
            resultados["falhou"].append(name)

    print("\n=== Resumo ===")
    print(f"OK: {resultados['ok']}")
    print(f"Falhou: {resultados['falhou']}")


if __name__ == "__main__":
    main()
