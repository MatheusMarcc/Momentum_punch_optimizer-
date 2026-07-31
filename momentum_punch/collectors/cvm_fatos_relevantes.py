"""
Coletor CVM — Fatos Relevantes, via portal de Dados Abertos (dados.cvm.gov.br).

CORREÇÃO: os arquivos são .zip (contendo um .csv dentro), não .csv direto —
confirmei isso no índice do diretório (dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/),
meu primeiro chute (.csv direto) dava 404.

Dataset IPE = Informes Periódicos e Eventuais. Fato Relevante é um dos valores
da coluna de categoria dentro desse dataset (junto com "Comunicado ao Mercado" etc).

Uso pensado no MVP: filtrar fato relevante das empresas que compõem o GOVE11
(Índice de Governança Corporativa Trade), não das 4 ETFs em geral — ver
discussão no README sobre por que isso não generaliza bem pros outros ETFs.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from ._http import get_with_retry

BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{ano}.zip"


def fetch_ipe_ano(ano: int) -> pd.DataFrame:
    url = BASE_URL.format(ano=ano)
    resp = get_with_retry(url, max_retries=2, timeout=60)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        # o zip tem 1 CSV dentro, nome geralmente igual ao do zip
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"Nenhum CSV encontrado dentro de {url}")
        with z.open(csv_names[0]) as f:
            # CSVs da CVM tipicamente usam ';' como separador e encoding latin-1
            df = pd.read_csv(f, sep=";", encoding="latin-1")
    return df


def fetch_fatos_relevantes(
    anos: list[int],
    empresas_cnpj_ou_nome: list[str] | None = None,
) -> pd.DataFrame:
    """
    anos: lista de anos a baixar (ex: [2024, 2025, 2026]).
    empresas_cnpj_ou_nome: filtro opcional por nome/razão social (contains, case-insensitive)
      pra restringir só às empresas que compõem o ETF de interesse (ex: GOVE11).
    """
    frames = []
    for ano in anos:
        try:
            df = fetch_ipe_ano(ano)
            frames.append(df)
            print(f"[cvm] ano {ano}: {len(df)} registros IPE")
        except Exception as exc:
            print(f"[cvm] Falha ao baixar ano {ano}: {exc}")

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)

    # filtra só a categoria "Fato Relevante" — nome da coluna pode variar
    # entre "Categoria" e "Categoria_Documento" dependendo do ano
    cat_col = next((c for c in full.columns if "categoria" in c.lower()), None)
    if cat_col:
        full = full[full[cat_col].astype(str).str.contains("Fato Relevante", case=False, na=False)]

    if empresas_cnpj_ou_nome:
        nome_col = next((c for c in full.columns if "denom" in c.lower() or "nome" in c.lower()), None)
        if nome_col:
            pattern = "|".join(empresas_cnpj_ou_nome)
            full = full[full[nome_col].astype(str).str.contains(pattern, case=False, na=False)]

    return full


if __name__ == "__main__":
    df = fetch_fatos_relevantes(anos=[2024, 2025, 2026])
    out_path = "data/raw/cvm_fatos_relevantes.csv"
    df.to_csv(out_path, index=False)
    print(f"[cvm] Salvo em {out_path} ({len(df)} registros)")
