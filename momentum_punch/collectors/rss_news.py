"""
Coletor de notícias via RSS — grátis, sem chave, mas SEM histórico profundo:
cada execução só pega o que está no feed no momento. Rode como job periódico e
deixe save_incremental acumular.

v2: tentei trocar pra feeds por categoria do InfoMoney, mas essas URLs não
existem (voltaram 0 itens) — voltei pro feed geral e deixei o filtro por
categoria (baseado no path da URL de cada matéria) fazer o trabalho, que já
testei e funciona: derruba loteria/esporte/política solta, mantém
mercados/economia/negócios/mundo.

ATENÇÃO — Valor Econômico: a URL abaixo não está confirmada (retornou 0 itens
no seu teste). Rode diagnose_feeds.py pra achar a URL certa, ou remova a fonte
do dict FEEDS se não achar — o pipeline funciona só com InfoMoney + Money Times.
"""
from __future__ import annotations

import datetime as dt

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

FEEDS = {
    "infomoney": "https://www.infomoney.com.br/feed/",
    "moneytimes": "https://www.moneytimes.com.br/feed/",
    "valor_economico": "https://valor.globo.com/rss/",  # não confirmado, ver aviso acima
}

# segunda camada de filtro (segurança, caso algum feed volte a misturar assunto):
# categorias sempre mantidas (aparecem na URL da matéria)
CATEGORIAS_PERMITIDAS = {"mercados", "economia", "negocios", "business", "mundo"}
# categorias sempre descartadas
CATEGORIAS_EXCLUIDAS = {"esportes", "consumo", "cultura", "diversao", "entretenimento"}
# "politica" só entra se tiver palavra-chave de risco macro/eleitoral no título
PALAVRAS_CHAVE_POLITICA = ["eleiç", "elei", "stf", "congresso", "reforma", "fiscal", "orçamento", "tarifa"]


def _clean_html(raw_html: str) -> str:
    """Remove tags/markup do resumo, deixando só o texto (RSS do InfoMoney vem com <img>/<p> cru)."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    # remove o boilerplate "The post ... appeared first on ..."
    if "appeared first on" in text:
        text = text.split("The post")[0].strip()
    return text


def _categoria_da_url(link: str) -> str | None:
    """Extrai o primeiro segmento de path da URL, que costuma ser a categoria (ex: /mercados/...)."""
    try:
        path = link.split("://", 1)[1].split("/", 1)[1]
        return path.split("/")[0].lower()
    except (IndexError, ValueError):
        return None


def _passa_no_filtro(titulo: str, link: str) -> bool:
    categoria = _categoria_da_url(link)
    if categoria in CATEGORIAS_EXCLUIDAS:
        return False
    if categoria == "politica":
        titulo_lower = titulo.lower()
        return any(kw in titulo_lower for kw in PALAVRAS_CHAVE_POLITICA)
    # categoria permitida, ou desconhecida (feed sem padrão de URL claro) -> deixa passar
    return True


def fetch_feed(name: str, url: str) -> pd.DataFrame:
    parsed = feedparser.parse(url)
    rows = []
    for entry in parsed.entries:
        titulo = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        if not _passa_no_filtro(titulo, link):
            continue
        rows.append(
            {
                "fonte": name,
                "titulo": titulo,
                "resumo": _clean_html(getattr(entry, "summary", "")),
                "link": link,
                "publicado_em": getattr(entry, "published", None),
                "coletado_em": dt.datetime.now().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def fetch_all(feeds: dict[str, str] = FEEDS) -> pd.DataFrame:
    frames = []
    for name, url in feeds.items():
        try:
            df = fetch_feed(name, url)
            frames.append(df)
            print(f"[rss_news] {name}: {len(df)} itens (após filtro)")
        except Exception as exc:
            print(f"[rss_news] Falha ao ler {name} ({url}): {exc}")
    if not frames:
        return pd.DataFrame(columns=["fonte", "titulo", "resumo", "link", "publicado_em", "coletado_em"])
    return pd.concat(frames, ignore_index=True)


def save_incremental(new_df: pd.DataFrame, out_path: str = "data/raw/rss_news.csv") -> pd.DataFrame:
    """Acumula no CSV existente, deduplicando por link."""
    import os

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="link", keep="first")
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)
    return combined


if __name__ == "__main__":
    df = fetch_all()
    combined = save_incremental(df)
    print(f"[rss_news] Total acumulado: {len(combined)} notícias em data/raw/rss_news.csv")
