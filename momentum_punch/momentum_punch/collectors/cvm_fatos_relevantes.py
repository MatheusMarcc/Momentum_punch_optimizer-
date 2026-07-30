"""
Coletor CVM — Fatos Relevantes, via portal de Dados Abertos (dados.cvm.gov.br).

ATENÇÃO — verifique antes de rodar em produção: a CVM disponibiliza fato relevante
dentro do dataset "IPE" (Informes Periódicos e Eventuais), como um dos valores da
coluna Categoria (junto com "Comunicado ao Mercado", "Aviso aos Acionistas" etc).
O layout exato (nome de coluna, encoding, separador) muda ocasionalmente — eu
escrevi isso com base no formato mais recente que conheço, mas se der erro de
parsing, abra a URL abaixo no navegador pra conferir o CSV mais atual:

  http://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/

Uso pensado no MVP: filtrar fato relevante das empresas que compõem o GOVE11
(Índice de Governança Corporativa Trade), não das 4 ETFs em geral — ver discussão
no README sobre por que isso não generaliza bem pros outros ETFs.
"""
from __future__ import annotations

import pandas as pd
import requests

BASE_URL = "http://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{ano}.csv"


def fetch_ipe_ano(ano: int) -> pd.DataFrame:
    url = BASE_URL.format(ano=ano)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    from io import StringIO

    # CSVs da CVM tipicamente usam ';' como separador e encoding latin-1
    df = pd.read_csv(StringIO(resp.content.decode("latin-1")), sep=";")
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
    # entre "Categoria" e "Categoria_Documento" dependendo do ano; ajuste se necessário
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
