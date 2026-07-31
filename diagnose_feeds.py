"""Diagnóstico rápido: testa cada feed RSS individualmente e mostra quantos
itens cada um devolveu de verdade, sem passar pelo resto do pipeline."""
import feedparser

FEEDS = {
    "infomoney": "https://www.infomoney.com.br/feed/",
    "moneytimes": "https://www.moneytimes.com.br/feed/",
    "valor_economico": "https://valor.globo.com/rss/",
}

for name, url in FEEDS.items():
    parsed = feedparser.parse(url)
    status = getattr(parsed, "status", "sem status HTTP (erro de conexão)")
    print(f"{name}: status={status}, entries={len(parsed.entries)}, bozo={parsed.bozo}")
    if parsed.bozo:
        print(f"  -> motivo do erro: {parsed.bozo_exception}")
