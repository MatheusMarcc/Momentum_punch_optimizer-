"""
Coletor Bacen Focus (Expectativas de Mercado) — via biblioteca python-bcb
(github.com/wilsonfreitas/python-bcb), em vez de eu montar a query OData na
mão. Motivo da troca: tentei duas vezes construir a URL manualmente e recebi
400 Bad Request mesmo em indicadores que deveriam ser válidos (IPCA incluso) —
sinal de que o bug tava na minha construção da query, não só nos nomes de
indicador. A python-bcb é mantida pela comunidade e usada amplamente, então
confio mais nela pra acertar a sintaxe OData do que em eu continuar chutando.

pip install python-bcb
"""
from __future__ import annotations

import pandas as pd
from bcb import Expectativas


def fetch_expectativas_selic(data_inicial: str, data_final: str) -> pd.DataFrame:
    """data_* no formato AAAA-MM-DD. Selic tem endpoint OData próprio."""
    em = Expectativas()
    ep = em.get_endpoint("ExpectativasMercadoSelic")
    df = ep.query().filter(ep.Data >= data_inicial, ep.Data <= data_final).collect()
    return df


def fetch_expectativas_mensais(indicador: str, data_inicial: str, data_final: str) -> pd.DataFrame:
    """
    indicador: nome exato cadastrado no Focus (ex: "IPCA", "Taxa de câmbio",
    "Produção industrial"). Não inclui Selic — usa fetch_expectativas_selic pra isso.
    data_* no formato AAAA-MM-DD.
    """
    em = Expectativas()
    ep = em.get_endpoint("ExpectativaMercadoMensais")
    df = (
        ep.query()
        .filter(ep.Indicador == indicador)
        .filter(ep.Data >= data_inicial, ep.Data <= data_final)
        .collect()
    )
    return df


def fetch_focus_stress_inputs(
    data_inicial: str,
    data_final: str,
    indicadores_mensais: list[str] = ("Taxa de câmbio", "IPCA"),
) -> pd.DataFrame:
    """
    Junta a dispersão (DesvioPadrao) de cada indicador num único DataFrame wide
    (index=data, colunas=f"{indicador}_std", f"{indicador}_mediana").
    Dispersão crescente nas expectativas = sinal objetivo de incerteza macro,
    pra alimentar o Índice de Estresse do circuit breaker sem depender só da LLM.
    """
    series = {}

    try:
        selic_df = fetch_expectativas_selic(data_inicial, data_final)
        if not selic_df.empty and "DesvioPadrao" in selic_df.columns:
            grouped = selic_df.groupby("Data").agg({"DesvioPadrao": "mean", "Mediana": "mean"})
            series["Selic_std"] = grouped["DesvioPadrao"]
            series["Selic_mediana"] = grouped["Mediana"]
            print(f"[bacen_focus] Selic: {len(selic_df)} registros")
        else:
            print("[bacen_focus] Selic: nenhum registro retornado")
    except Exception as exc:
        print(f"[bacen_focus] Erro em Selic: {exc}")

    for ind in indicadores_mensais:
        try:
            df = fetch_expectativas_mensais(ind, data_inicial, data_final)
            if df.empty or "DesvioPadrao" not in df.columns:
                print(f"[bacen_focus] {ind}: nenhum registro retornado")
                continue
            grouped = df.groupby("Data").agg({"DesvioPadrao": "mean", "Mediana": "mean"})
            series[f"{ind}_std"] = grouped["DesvioPadrao"]
            series[f"{ind}_mediana"] = grouped["Mediana"]
            print(f"[bacen_focus] {ind}: {len(df)} registros")
        except Exception as exc:
            print(f"[bacen_focus] Erro em {ind}: {exc}")

    return pd.DataFrame(series)


if __name__ == "__main__":
    import datetime as dt

    hoje = dt.date.today()
    inicio = (hoje - dt.timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")

    df = fetch_focus_stress_inputs(inicio, fim)
    out_path = "data/raw/bacen_focus.csv"
    df.to_csv(out_path)
    print(f"[bacen_focus] Salvo em {out_path} ({len(df)} linhas)")
