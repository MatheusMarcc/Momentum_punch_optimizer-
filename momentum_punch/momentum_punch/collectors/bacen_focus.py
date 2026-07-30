"""
Coletor Bacen Focus (Expectativas de Mercado) — API Olinda (OData), pública, sem chave.
Documentação: https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/aplicacao

Usamos o endpoint de expectativas de mercado ANUAIS/MENSAIS (ex: Selic e câmbio pro
ano corrente e próximo) e o de IPCA, que são os mais usados como proxy de "consenso
de mercado" no Índice de Estresse.

Endpoint base: https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

BASE_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata"


def fetch_expectativas_mensais(
    indicador: str,
    data_inicial: str,
    data_final: str,
) -> pd.DataFrame:
    """
    indicador: "Selic", "Câmbio", "IPCA", etc (nomes conforme cadastrados no Focus).
    data_* no formato AAAA-MM-DD.
    Retorna a mediana, desvio-padrão e coeficiente de variação das expectativas —
    a DISPERSÃO (desvio-padrão) é o que mais interessa como sinal de incerteza/estresse.
    """
    endpoint = f"{BASE_URL}/ExpectativasMercadoMensais"
    filtro = (
        f"Indicador eq '{indicador}' and "
        f"Data ge '{data_inicial}' and Data le '{data_final}'"
    )
    params = {
        "$filter": filtro,
        "$format": "json",
        "$select": "Indicador,Data,Media,Mediana,DesvioPadrao,Minimo,Maximo,numeroRespondentes",
        "$orderby": "Data asc",
    }
    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["value"])
    if not df.empty:
        df["Data"] = pd.to_datetime(df["Data"])
    return df


def fetch_focus_stress_inputs(
    data_inicial: str,
    data_final: str,
    indicadores: list[str] = ("Selic", "Câmbio", "IPCA"),
) -> pd.DataFrame:
    """
    Junta a dispersão (DesvioPadrao) de cada indicador num único DataFrame wide
    (index=data, colunas=f"{indicador}_std", f"{indicador}_mediana").
    Dispersão crescente nas expectativas = sinal objetivo de incerteza macro,
    pra alimentar o Índice de Estresse do circuit breaker sem depender só da LLM.
    """
    series = {}
    for ind in indicadores:
        try:
            df = fetch_expectativas_mensais(ind, data_inicial, data_final)
            if df.empty:
                print(f"[bacen_focus] {ind}: nenhum registro retornado")
                continue
            grouped = df.groupby("Data").agg(
                {"DesvioPadrao": "mean", "Mediana": "mean"}
            )
            series[f"{ind}_std"] = grouped["DesvioPadrao"]
            series[f"{ind}_mediana"] = grouped["Mediana"]
            print(f"[bacen_focus] {ind}: {len(df)} registros")
        except Exception as exc:
            print(f"[bacen_focus] Falha ao buscar {ind}: {exc}")
    return pd.DataFrame(series)


if __name__ == "__main__":
    hoje = dt.date.today()
    inicio = (hoje - dt.timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")

    df = fetch_focus_stress_inputs(inicio, fim)
    out_path = "data/raw/bacen_focus.csv"
    df.to_csv(out_path)
    print(f"[bacen_focus] Salvo em {out_path} ({len(df)} linhas)")
