"""
Coletor Bacen SGS (Sistema Gerenciador de Séries Temporais) — API REST pública,
sem necessidade de chave. Documentação: https://dadosabertos.bcb.gov.br/

Séries usadas (códigos SGS):
  432  - Meta Selic definida pelo Copom (% a.a.)
  433  - IPCA - variação mensal (%)
  1    - Dólar comercial venda (fim de período)
  4390 - Selic acumulada no mês (% a.m.)

Endpoint: http://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ._http import get_with_retry

BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

SERIES = {
    "selic_meta": 432,
    "ipca_mensal": 433,
    "dolar_venda": 1,
    "cdi_diario": 12,  # % ao dia — usado pra montar o retorno diário do ativo livre de risco
}


def fetch_series(codigo: int, data_inicial: str, data_final: str) -> pd.DataFrame:
    """
    data_inicial/data_final no formato DD/MM/AAAA (é o formato que a API do Bacen exige).
    Retorna DataFrame com colunas [data, valor].
    """
    url = BASE_URL.format(codigo=codigo)
    params = {
        "formato": "json",
        "dataInicial": data_inicial,
        "dataFinal": data_final,
    }
    resp = get_with_retry(url, params=params)
    df = pd.DataFrame(resp.json())
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float)
    return df


def fetch_all_series(
    data_inicial: str,
    data_final: str,
    series: dict[str, int] = SERIES,
) -> pd.DataFrame:
    """Junta todas as séries num único DataFrame wide (index=data, colunas=nome da série)."""
    frames = {}
    for name, codigo in series.items():
        try:
            df = fetch_series(codigo, data_inicial, data_final)
            frames[name] = df.set_index("data")["valor"]
            print(f"[bacen_sgs] {name} (SGS {codigo}): {len(df)} registros")
        except Exception as exc:
            print(f"[bacen_sgs] Falha ao buscar série {name} (SGS {codigo}): {exc}")
    return pd.DataFrame(frames)


if __name__ == "__main__":
    hoje = dt.date.today()
    inicio = (hoje - dt.timedelta(days=365 * 3)).strftime("%d/%m/%Y")
    fim = hoje.strftime("%d/%m/%Y")

    df = fetch_all_series(inicio, fim)
    out_path = "data/raw/bacen_sgs.csv"
    df.to_csv(out_path)
    print(f"[bacen_sgs] Salvo em {out_path} ({len(df)} linhas)")
