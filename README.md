# Momentum Punch — Sentiment Alpha Portfolio
Desafio Itaú Asset Quant AI 2026

Implementação dos 5 blocos do pipeline descrito no deck: Coleta → Score LLM →
Suavização EMA → Vetor de retorno ajustado → Otimização Markowitz, com overlay
de circuit breaker geopolítico.

## Reproduzindo os números do relatório

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python fetch_prices_yfinance.py --years 5      # preço dos 4 ETFs (REVE11 começa em 15/07/2021)
python collect_data.py --only bacen_sgs,bacen_focus
python build_stress_index_focus.py             # índice de estresse (dispersão do Focus)
python pull_cvm_historical.py                  # corpus textual (fatos relevantes)
python build_sentiment_dataset.py --structured --source cvm \
    --out data/processed/sentiment_scores_genai.csv   # scoring via LLM (~45 min, Ollama local)

pytest test_suite.py                           # invariantes do software
python run_ablations.py --periodo teste --scores data/processed/sentiment_scores_genai.csv
python gerar_tabelas_relatorio.py --scores data/processed/sentiment_scores_genai.csv
```

Cada execução grava um **manifesto** (`data/processed/manifesto_*.json`) com
sha256 de cada insumo, commit do código, modelo e hash do prompt — é o que
permite conferir depois qual versão produziu qual número.

As tabelas saem em `relatorio/tabelas/`, cada uma em dois formatos gerados do
mesmo arquivo-fonte: `.tex` pra `\input{}` no Overleaf e `.csv` pro dashboard
consumir. Nenhum número do relatório é digitado à mão, e painel e PDF não podem
divergir — se divergirem, é porque alguém editou um dos dois manualmente.

```bash
streamlit run dashboard.py    # aba "Resultados finais" lê relatorio/tabelas/*.csv
```

### Protocolo de treino/teste

`config.DATA_CORTE_TREINO_TESTE` (31/12/2024) separa calibração de avaliação.
Os parâmetros são escolhidos com `validate_signal_matrix.py`, que só olha o
treino, e o teste é avaliado **uma vez**. `--periodo {tudo,treino,teste}`
controla o recorte; o backtest sempre roda contínuo e só as métricas são
recortadas, pra o período de teste não perder o warm-up da covariância.

### Sobre o Sharpe

`Sharpe (excedente ao CDI)` é o índice de Sharpe de verdade, sobre `(r - CDI)`.
`Retorno/Vol` é a razão crua, reportada em separado e rotulada como não sendo
Sharpe. Com CDI a ~13% a.a., as duas diferem por um fator de 2 a 3 — confundir
uma com a outra infla o resultado de forma substancial.

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
  config.py          # tickers, thresholds, corte treino/teste — mexa aqui primeiro
  data_collection.py # coleta de notícias (RSS) e tweets (API do X, opcional)
  text_filter.py     # filtro de relevância por ticker (keywords + exclusões)
  sentiment.py       # scoring GenAI estruturado + FinBERT + EMA + índice de estresse
  optimizer.py       # Markowitz modificado, Black-Litterman, Ledoit-Wolf
  risk_overlay.py    # circuit breaker: Risk-On / Risk-Off
  backtest.py        # backtest walk-forward + métricas + os 5 benchmarks
  manifesto.py       # procedência: hash dos insumos, commit, prompt, categoria
  synthetic_data.py  # gerador de dados sintéticos pra teste local
main.py                       # pipeline ponta a ponta (dados sintéticos)
run_real_backtest.py          # backtest com dado real, por período
run_ablations.py              # matriz A0–A12
validate_signal_matrix.py     # validação estatística + calibração só no treino
gerar_tabelas_relatorio.py    # tabelas .tex do relatório
dashboard.py                  # painel Streamlit
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

4. **Sentiment score real**: monte um dict `{data: {ticker: [textos]}}` com o
   histórico coletado e rode `sentiment.score_history_structured(...)` — o
   motor GenAI, que devolve sentimento, relevância e confiança por ticker,
   aplica o gate (`score = sentimento × relevância`) e entrega o DataFrame já
   suavizado por EMA, no formato que `run_backtest()` espera. Roda local via
   Ollama, sem custo de API. `sentiment.score_history()` é a alternativa
   determinística via FinBERT-PT-BR, mais rápida mas sem relevância nem
   confiança — não é o caminho avaliado no relatório.

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

## Backtest com dado real (preço via yfinance)

```bash
python fetch_prices_yfinance.py        # preço real dos ETFs (Yahoo Finance, sufixo .SA)
python collect_data.py --only bacen_sgs   # inclui CDI (SGS 12) agora
python run_real_backtest.py
```

Cobertura confirmada no Yahoo Finance para os quatro ETFs. O REVE11 tem
histórico desde **15/07/2021** (data de início de negociação), então `--years 5`
cobre a série inteira dele — o default de 3 anos descartava metade da amostra
sem avisar, e a janela mais curta que ele produzia excluía justamente o período
de 2021–2023, em que a estratégia teve seu pior desempenho relativo.

`data/raw/benchmark_prices.csv` guarda o IVVB11, usado **só** como benchmark
(comparador 40/40/20). Ele nunca entra no universo investível — manter os
arquivos separados é o que impede um ativo de comparação de vazar pro
otimizador.

O `run_real_backtest.py` já lida com sentiment/stress index incompletos: se
`data/processed/sentiment_scores.csv` ou `stress_index.csv` não existirem
ainda (ou cobrirem só alguns dias), ele usa forward-fill a partir do último
score conhecido e neutro (0.0) antes disso — não trava o backtest, só avisa.

## Automação diária (Agendador de Tarefas do Windows)

O `daily_pipeline.py` encadeia coleta de RSS + scoring via Ollama, com log em
`data/daily_pipeline.log`. Pra rodar sozinho todo dia:

**Opção A — pela interface:**
1. Abra o Agendador de Tarefas (Win+R → `taskschd.msc`)
2. "Criar Tarefa Básica" → nome "Momentum Punch Diário"
3. Disparador: Diariamente, escolha um horário (ex: 08:00, batendo com o "job das 8h" do deck)
4. Ação: "Iniciar um programa" → aponte pro `run_daily_pipeline.bat` (confira se o caminho dentro dele bate com onde você descompactou a pasta)
5. Finalizar

**Opção B — por linha de comando (PowerShell como administrador):**
```powershell
schtasks /create /tn "MomentumPunchDiario" /tr "C:\Users\mathe\Desktop\ItauQuantAI\momentum_punch\run_daily_pipeline.bat" /sc daily /st 08:00
```

Pra testar sem esperar o agendamento: `schtasks /run /tn "MomentumPunchDiario"`,
depois confere `data\daily_pipeline.log`.

Bacen SGS/Focus e GDELT ficam de fora do job diário de propósito — SGS/Focus já
trazem o histórico completo numa chamada só, e GDELT tem rate limit agressivo.
Rode esses via `collect_data.py` manualmente de vez em quando.
