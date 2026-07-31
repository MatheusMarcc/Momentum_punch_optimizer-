"""
Pega o texto bruto já coletado (data/raw/rss_news.csv) e manda pro Ollama via
sentiment.py, gerando o Sentiment Alpha Score por data/ticker.

Fluxo:
  1. python collect_data.py --only rss_news      (gera data/raw/rss_news.csv)
  2. ollama pull qwen2.5:3b-instruct-q4_K_M       (uma vez só)
  3. python build_sentiment_dataset.py            (este script)

Saída: data/processed/sentiment_scores.csv (index=data, colunas=tickers, já em EMA)

Observação sobre RSS: como não tem backfill, na prática isso vai te dar só
UM dia de score de cada vez (o dia em que você rodou o collect_data.py).
Pra construir histórico, rode collect_data.py + build_sentiment_dataset.py
periodicamente (ex: cron diário) e deixe acumular — ou use GDELT como fonte
de texto histórico pro backtest (ver momentum_punch/collectors/gdelt.py).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from momentum_punch import config, sentiment


def _filtra_por_ticker(textos: list[str], ticker: str) -> list[str]:
    """Mantém só os textos que citam alguma palavra-chave do ticker (config.TICKER_KEYWORDS).
    Se nenhum texto do dia bater pra um ticker específico (dia sem notícia
    relevante pra ele), a lista fica vazia e o LLM recebe isso explicitamente
    — não inventa sentimento do nada, o score.py já trata texto vazio como
    '(nenhum texto coletado hoje)'."""
    keywords = config.TICKER_KEYWORDS.get(ticker, [])
    if not keywords:
        return textos
    return [t for t in textos if any(kw in t.lower() for kw in keywords)]


def load_daily_texts(
    rss_csv: str = "data/raw/rss_news.csv",
    source: str = "rss",
    gdelt_csv: str = "data/raw/gdelt_stress.csv",
    gdelt_topics_dir: str = "data/raw/gdelt_by_topic",
) -> dict[str, dict[str, list[str]]]:
    """
    Lê o texto bruto e agrupa por data de publicação, filtrando por ticker via
    palavra-chave (config.TICKER_KEYWORDS) — sem isso, todo ticker recebia
    o mesmo pool de manchetes e o score acabava saindo quase idêntico entre eles.

    source="rss": só o que está no feed agora, sem backfill.
    source="gdelt": um CSV único de estresse geral, filtrado por keyword depois
        (cobertura fraca pra tema específico de ticker).
    source="gdelt_topics": CSVs SEPARADOS por tema (gdelt_historical_pull.py),
        já nasceram de uma query específica por ticker — melhor diferenciação,
        e com profundidade histórica real (meses, não 1-2 dias como o RSS).
    """
    if source == "rss":
        if not os.path.exists(rss_csv):
            raise FileNotFoundError(
                f"{rss_csv} não existe. Rode antes: python collect_data.py --only rss_news"
            )
        df = pd.read_csv(rss_csv)
        df["publicado_em"] = pd.to_datetime(df["publicado_em"], errors="coerce", utc=True)
        df = df.dropna(subset=["publicado_em"])
        df["data_str"] = df["publicado_em"].dt.date.astype(str)
        textos_por_data = {
            date: (group["titulo"].fillna("") + ". " + group["resumo"].fillna("")).tolist()
            for date, group in df.groupby("data_str")
        }
        texts_by_date_ticker: dict[str, dict[str, list[str]]] = {}
        for date, textos_do_dia in textos_por_data.items():
            texts_by_date_ticker[date] = {t: _filtra_por_ticker(textos_do_dia, t) for t in config.TICKERS}
        return texts_by_date_ticker

    elif source == "gdelt":
        if not os.path.exists(gdelt_csv):
            raise FileNotFoundError(
                f"{gdelt_csv} não existe. Rode antes: python collect_data.py --only gdelt"
            )
        df = pd.read_csv(gdelt_csv, parse_dates=["data"])
        df = df.dropna(subset=["data"])
        df["data_str"] = df["data"].dt.date.astype(str)
        textos_por_data = {
            date: group["titulo"].fillna("").tolist()
            for date, group in df.groupby("data_str")
        }
        texts_by_date_ticker = {}
        for date, textos_do_dia in textos_por_data.items():
            texts_by_date_ticker[date] = {t: _filtra_por_ticker(textos_do_dia, t) for t in config.TICKERS}
        return texts_by_date_ticker

    elif source == "gdelt_topics":
        texts_by_date_ticker = {}
        for ticker in config.TICKERS:
            csv_path = os.path.join(gdelt_topics_dir, f"{ticker}.csv")
            if not os.path.exists(csv_path):
                print(f"[build_sentiment_dataset] AVISO: {csv_path} não existe, {ticker} vai ficar vazio. Rode gdelt_historical_pull.py antes.")
                continue
            df = pd.read_csv(csv_path, parse_dates=["data"])
            df = df.dropna(subset=["data"])
            df["data_str"] = df["data"].dt.date.astype(str)
            # ainda aplico o filtro de keyword como segurança extra, mesmo a
            # query já sendo específica — reduz falso positivo de busca ampla
            for date, group in df.groupby("data_str"):
                textos = _filtra_por_ticker(group["titulo"].fillna("").tolist(), ticker)
                texts_by_date_ticker.setdefault(date, {t: [] for t in config.TICKERS})
                texts_by_date_ticker[date][ticker] = textos
        return texts_by_date_ticker

    elif source == "cvm":
        cvm_csv = "data/raw/cvm_fatos_relevantes.csv"
        if not os.path.exists(cvm_csv):
            raise FileNotFoundError(
                f"{cvm_csv} não existe. Rode antes: python pull_cvm_historical.py"
            )
        df = pd.read_csv(cvm_csv, low_memory=False)

        # Data_Entrega = data de protocolo/entrega do documento à CVM (divulgação
        # real ao mercado). Data_Referencia é a data do EVENTO referenciado no
        # documento (pode ser futura — foi essa a fonte do bug de datas em
        # 2027/2028 que a gente pegou no backtest). Confirmado nas colunas reais
        # do dataset IPE: CNPJ_Companhia, Nome_Companhia, Codigo_CVM,
        # Data_Referencia, Categoria, Tipo, Especie, Assunto, Data_Entrega,
        # Tipo_Apresentacao, Protocolo_Entrega, Versao, Link_Download.
        data_col = next((c for c in df.columns if c == "Data_Entrega"), None) \
            or next((c for c in df.columns if "entrega" in c.lower()), None) \
            or next((c for c in df.columns if "data" in c.lower() and "refer" not in c.lower()), None) \
            or next((c for c in df.columns if "data" in c.lower()), None)
        texto_col = next((c for c in df.columns if "assunto" in c.lower()), None) \
            or next((c for c in df.columns if "denom" in c.lower() or "nome" in c.lower()), None)

        if data_col is None or texto_col is None:
            raise ValueError(
                f"Não achei coluna de data/texto em {cvm_csv}. Colunas disponíveis: "
                f"{list(df.columns)} — ajuste a detecção manualmente."
            )

        df[data_col] = pd.to_datetime(df[data_col], errors="coerce", dayfirst=True)
        df = df.dropna(subset=[data_col])
        df["data_str"] = df[data_col].dt.date.astype(str)
        textos_por_data = {
            date: group[texto_col].fillna("").astype(str).tolist()
            for date, group in df.groupby("data_str")
        }
        texts_by_date_ticker = {}
        for date, textos_do_dia in textos_por_data.items():
            texts_by_date_ticker[date] = {t: _filtra_por_ticker(textos_do_dia, t) for t in config.TICKERS}
        return texts_by_date_ticker

    else:
        raise ValueError("source deve ser 'rss', 'gdelt', 'gdelt_topics' ou 'cvm'")


def build_stress_index_from_gdelt(
    gdelt_csv: str = "data/raw/gdelt_by_topic/stress.csv",
    out: str = "data/processed/stress_index.csv",
    provider: str | None = None,
):
    """
    Roda sentiment.score_stress_index() sobre cada dia do GDELT, salva série
    histórica. O GDELT foi configurado (collectors/gdelt.py) pra buscar os
    temas de estresse macro/geopolítico, então é a fonte certa pro circuit
    breaker — diferente do sentiment por ticker, que puxa de load_daily_texts.
    """
    if not os.path.exists(gdelt_csv):
        raise FileNotFoundError(f"{gdelt_csv} não existe. Rode antes: python collect_data.py --only gdelt")

    df = pd.read_csv(gdelt_csv, parse_dates=["data"])
    df = df.dropna(subset=["data"])
    df["data_str"] = df["data"].dt.date.astype(str)
    textos_por_data = {date: group["titulo"].fillna("").tolist() for date, group in df.groupby("data_str")}

    print(f"[build_sentiment_dataset] {len(textos_por_data)} dia(s) de estresse a processar via GDELT")

    rows = {}
    for date, texts in sorted(textos_por_data.items()):
        rows[date] = sentiment.score_stress_index(texts, provider=provider)
        print(f"[build_sentiment_dataset] {date}: stress={rows[date]:.2f}")

    stress_series = pd.Series(rows, name="stress_index")
    stress_series.index = pd.to_datetime(stress_series.index)
    stress_series = stress_series.sort_index()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    stress_series.to_csv(out)
    print(f"[build_sentiment_dataset] Salvo em {out} ({len(stress_series)} dias)")
    return stress_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama", choices=["ollama", "groq"],
                         help="Só usado com --stress (GDELT, dormente). O sentiment por ticker usa sempre FinBERT-PT-BR.")
    parser.add_argument("--source", default="rss", choices=["rss", "gdelt", "gdelt_topics", "cvm"], help="rss=sem histórico, gdelt/gdelt_topics=via GDELT, cvm=Fato Relevante (multi-ano, sem rate limit)")
    parser.add_argument("--rss-csv", default="data/raw/rss_news.csv")
    parser.add_argument("--gdelt-csv", default="data/raw/gdelt_stress.csv")
    parser.add_argument("--out", default="data/processed/sentiment_scores.csv")
    parser.add_argument("--force", action="store_true", help="reprocessa dias já escorados (ignora o CSV existente)")
    parser.add_argument("--stress", action="store_true", help="gera stress_index.csv a partir do GDELT em vez do sentiment por ticker")
    args = parser.parse_args()

    if args.stress:
        build_stress_index_from_gdelt(args.gdelt_csv, out="data/processed/stress_index.csv", provider=args.provider)
        return

    texts_by_date_ticker = load_daily_texts(args.rss_csv, source=args.source, gdelt_csv=args.gdelt_csv)

    # pula datas já escoradas em execuções anteriores — importante pra rodar isso
    # como job diário sem reprocessar o histórico inteiro toda vez (mais lento
    # à toa, mesmo sem custo de API já que é Ollama local)
    if not args.force and os.path.exists(args.out):
        already_scored = pd.read_csv(args.out, index_col=0, parse_dates=True).index
        already_scored_str = {d.strftime("%Y-%m-%d") for d in already_scored}
        antes = len(texts_by_date_ticker)
        texts_by_date_ticker = {d: v for d, v in texts_by_date_ticker.items() if d not in already_scored_str}
        pulados = antes - len(texts_by_date_ticker)
        if pulados:
            print(f"[build_sentiment_dataset] {pulados} data(s) já escorada(s) antes, pulando (use --force pra reprocessar)")

    n_dates = len(texts_by_date_ticker)
    print(f"[build_sentiment_dataset] {n_dates} data(s) nova(s) a processar (sentiment via FinBERT-PT-BR)")

    if n_dates == 0:
        print("[build_sentiment_dataset] Nada novo pra processar, encerrando.")
        return

    novos_scores = sentiment.score_history(texts_by_date_ticker)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out) and not args.force:
        existentes = pd.read_csv(args.out, index_col=0, parse_dates=True)
        scores_df = pd.concat([existentes, novos_scores]).sort_index()
        scores_df = scores_df[~scores_df.index.duplicated(keep="last")]
    else:
        scores_df = novos_scores

    scores_df.to_csv(args.out)
    print(f"\n[build_sentiment_dataset] Salvo em {args.out} ({len(scores_df)} dias no total)")
    print(scores_df.tail())


if __name__ == "__main__":
    main()