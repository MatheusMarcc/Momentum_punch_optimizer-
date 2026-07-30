# Momentum Punch — Sentiment Alpha Portfolio
Desafio Itaú Asset Quant AI 2026

Implementação dos 5 blocos do pipeline descrito no deck: Coleta → Score LLM →
Suavização EMA → Vetor de retorno ajustado → Otimização Markowitz, com overlay
de circuit breaker geopolítico.

## Setup do LLM de sentimento (grátis — Ollama local)

Sem custo de API. Rodando na sua RTX 3050 4GB:

```bash
# 1. instale o Ollama: https://ollama.com/download
# 2. baixe o modelo (uma vez só, ~2GB):
ollama pull qwen2.5:3b-instruct-q4_K_M
# 3. teste:
python test_sentiment_ollama.py
```

3B em vez de 7B por causa da VRAM — 7B em Q4 fica em ~4.5-5GB e no limite/lento
numa 4GB. Se quiser mais qualidade e tiver folga de VRAM, troque
`config.OLLAMA_MODEL` por `qwen2.5:7b-instruct-q4_K_M`.

Alternativa sem instalar nada localmente: `config.SENTIMENT_PROVIDER = "groq"`
(API gratuita, precisa de `GROQ_API_KEY` grátis em console.groq.com, ~30
req/min — bom pra validar num modelo maior sem depender da GPU, mas aperta
se você rodar o backtest histórico inteiro numa sentada só).

## Instalação

```bash
pip install -r requirements.txt
export X_BEARER_TOKEN="seu-token"         # opcional, só se for coletar tweets via API do X (paga)
export GROQ_API_KEY="sua-chave"           # opcional, só se usar provider="groq" no sentiment.py
```

## Rodando a demo (dados sintéticos)

```bash
python main.py
```

Isso roda o backtest inteiro com preços e sentiment **sintéticos** (não são dados
reais de mercado — servem só pra validar que a mecânica do pipeline funciona:
otimização, circuit breaker, cálculo de métricas). Troque pelos dados reais
antes de tirar qualquer conclusão pro desafio.

## Estrutura

```
momentum_punch/
  config.py          # tickers, thresholds, parâmetros do modelo — mexa aqui primeiro
  data_collection.py # coleta de notícias (RSS) e tweets (API do X, opcional)
  sentiment.py        # scoring via Claude + suavização EMA + índice de estresse
  optimizer.py         # Markowitz modificado (mu ajustado por sentimento) via cvxpy
  risk_overlay.py      # circuit breaker: Risk-On / Risk-Off
  backtest.py           # motor de backtest walk-forward + métricas + benchmark
  synthetic_data.py     # gerador de dados sintéticos pra teste local
main.py                 # orquestra o pipeline ponta a ponta
```

## Coleta de dados (Etapa 1 — só puxar pra CSV)

```bash
python collect_data.py                 # roda todos os coletores
python collect_data.py --only bacen_sgs,rss_news   # roda só alguns
```

Gera CSVs em `data/raw/`: `bacen_sgs.csv`, `bacen_focus.csv`, `rss_news.csv`,
`gdelt_stress.csv`, `reddit_posts.csv`, `cvm_fatos_relevantes.csv`.

**Importante — eu não consigo rodar isso daqui do sandbox**: minha rede só alcança
domínios de infra (pypi, github etc), não bcb.gov.br, cvm.gov.br, gdelt ou os
portais de notícia. Rodei o orquestrador aqui só pra confirmar que a lógica e o
tratamento de erro funcionam (cada coletor falha isolado, sem derrubar os outros);
os CSVs de verdade só saem rodando na sua máquina.

Cada coletor, o que precisa e o que não precisa de credencial:

| Coletor | Credencial | Histórico? |
|---|---|---|
| `bacen_sgs` | Nenhuma | Sim, profundo (SGS tem série completa) |
| `bacen_focus` | Nenhuma | Sim (Focus semanal, desde ~2001) |
| `rss_news` | Nenhuma | **Não** — só o que está no feed agora. Rode como job periódico e deixe `save_incremental` acumular. |
| `gdelt` | Nenhuma | Parcial — bom pra últimos meses, mais fraco pra histórico muito longo |
| `reddit` | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` (grátis, [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)) | Limitado (a API do Reddit não retorna posts antigos facilmente) |
| `cvm` | Nenhuma | Sim, por ano (dataset IPE anual) — **URL/layout não 100% verificado por mim, confira `dados.cvm.gov.br` se der erro de parsing** |

Pontos de atenção específicos:
- **RSS não tem backfill.** Se você precisa de histórico de notícia pra alimentar
  o backtest longo, GDELT é a fonte melhor pra isso; RSS serve pra ir daqui pra
  frente (produção/demo ao vivo).
- **CVM**: o coletor já filtra a categoria "Fato Relevante" dentro do dataset IPE,
  mas eu não tenho 100% de certeza do nome exato da coluna/URL no layout mais
  recente — testei a lógica, não o dado real. Se `fetch_ipe_ano` der erro de
  parsing, abra a URL no navegador e ajuste o `sep`/encoding conforme o CSV atual.
- **Reddit**: uso não-comercial (tier free, 100 QPM) — adequado pro escopo do
  desafio acadêmico, não pra um produto comercial depois.

## Plugando dados reais

1. **Preços dos ETFs (B3)**: `synthetic_data.generate_prices()` é só um mock.
   Eu não tenho acesso de rede a provedores de cotação B3 daqui do sandbox
   (Yahoo Finance também tem cobertura ruim de ETFs B3). Exporte histórico da
   sua corretora, MetaTrader5, ou um provedor tipo Comdinheiro/Cedro, e monte
   um DataFrame `prices` (index=data, colunas=`ISUS11, GOVE11, REVE11, BOVA11`)
   igual ao que `generate_prices()` retorna.

2. **CDI**: baixe a série histórica do CDI (ex: SGS do Bacen, série 12) e
   converta pra retorno diário — substitui `generate_cdi_daily_return()`.

3. **Notícias/tweets**: `data_collection.fetch_news_headlines()` já funciona
   com feeds RSS reais (ajuste a lista `NEWS_FEEDS` em `data_collection.py`).
   Pra tweets, defina `X_BEARER_TOKEN` (API paga do X/Twitter).

4. **Sentiment score real**: em vez de `synthetic_data.generate_sentiment_scores()`,
   monte um dict `{data: {ticker: [textos]}}` com o histórico coletado e rode
   `sentiment.score_history(...)` — isso chama a API da Claude de verdade e
   já devolve o DataFrame suavizado por EMA, no formato que `run_backtest()`
   espera.

5. **Índice de estresse real**: mesma lógica, usando `sentiment.score_stress_index()`
   sobre os textos do dia relacionados a `config.STRESS_THEMES`.

## Pontos que valem calibração no backtest (mencionados no deck)

- `config.REBALANCE_FREQ`: frequência de rebalanceamento (semanal por padrão,
  mas o deck já cita que isso deve ser calibrado).
- `config.SENTIMENT_TILT_STRENGTH`: quanto o sentiment score desloca o mu
  histórico — é o parâmetro mais sensível do modelo, vale rodar um grid search.
- `config.STRESS_THRESHOLD_Z` e `config.RISK_OFF_MAX_EQUITY`: sensibilidade do
  circuit breaker — thresholds muito baixos derrubam o Sharpe (trigger-happy),
  muito altos não protegem o drawdown.
- `config.RISK_AVERSION`: lambda da função objetivo de Markowitz.

## Métricas calculadas no backtest

Retorno total, CAGR, volatilidade anualizada, Sharpe, max drawdown — comparado
contra o benchmark (`100% BOVA11` ou `60% BOVA11 / 40% CDI`, configurável em
`run_backtest(..., benchmark=...)`).
