"""
Parâmetros centrais do Momentum Punch.
Ajuste aqui em vez de espalhar constantes pelo código.
"""

# Universo de ativos (5 ativos: 4 ETFs de risco + CDI como ativo livre de risco)
TICKERS = ["ISUS11", "GOVE11", "REVE11", "BOVA11"]
RISK_FREE = "CDI"

# Temas monitorados pela LLM por ticker (usados no prompt de sentiment, dão contexto)
TICKER_THEMES = {
    "ISUS11": "sustentabilidade, emissões, clima, ESG (Índice de Sustentabilidade Empresarial)",
    "GOVE11": "governança corporativa, diretoria, conselho de administração, escândalos corporativos",
    "REVE11": "transição energética, energia limpa, renováveis, carros elétricos",
    "BOVA11": "Ibovespa, economia brasileira, PIB, situação fiscal",
}

# Palavras-chave por ticker, usadas pra FILTRAR quais manchetes vão pra cada
# scoring (sem isso, todo ticker recebia o mesmo pool de texto e o LLM acabava
# só refletindo o humor geral do dia, sem diferenciar governança de energia etc)
TICKER_KEYWORDS = {
    "ISUS11": ["sustentabilidade", "esg", "emiss", "clima", "carbono", "sustentável"],
    "GOVE11": ["governança", "conselho", "diretoria", "escândalo", "acionista", "fraude"],
    "REVE11": ["energia", "renovável", "elétric", "transição energética", "solar", "eólic"],
    "BOVA11": ["ibovespa", "bolsa", "b3", "selic", "juros", "pib", "fiscal", "dólar", "câmbio",
               "petróleo", "brent", "fed", "guerra", "geopolít", "recessão", "inflação"],
}

# Temas de estresse macro/geopolítico (usados para o Índice de Estresse do circuit breaker)
STRESS_THEMES = "eleições, conflitos armados, choques globais, risco geopolítico, crise cambial"

# Suavização por DECAIMENTO EXPONENCIAL BASEADO EM DIA DE CALENDÁRIO (halflife),
# não mais em "posição na tabela" (o antigo EMA_SPAN=5 tratava 5 OBSERVAÇÕES
# como se fossem 5 dias consecutivos — com fonte esparsa tipo CVM, isso fazia
# uma notícia de meses atrás "contar" como se fosse de poucos dias atrás).
EMA_HALFLIFE_DAYS = 5

# Otimização (Markowitz modificado)
RISK_AVERSION = 3.0          # penalidade de risco (lambda) na função objetivo
MAX_WEIGHT_PER_ETF = 0.5     # teto de concentração por ETF
MIN_WEIGHT_PER_ETF = 0.0     # long-only

# Circuit breaker (overlay de gestão de risco)
STRESS_THRESHOLD_Z = 0.6     # índice de estresse (0 a 1) acima do qual entra Risk-Off
RISK_OFF_MAX_EQUITY = 0.30   # teto de exposição total a ETFs em modo Risk-Off

# Rebalanceamento
REBALANCE_FREQ = "W"         # semanal (pandas offset alias)

# Janela de covariância (dias úteis) usada para estimar Sigma
COV_LOOKBACK_DAYS = 60

# Sentiment score bruto do LLM: intervalo esperado
SENTIMENT_MIN, SENTIMENT_MAX = -1.0, 1.0

# ---------------------------------------------------------------------------
# Protocolo de treino/teste — seção 5.1 do pré-relatório ("parâmetros escolhidos
# em treino/validação e congelados antes do teste final").
#
# É UMA DATA FIXA, não uma fração. Fração (os antigos 70%) dá uma data de corte
# diferente pra cada série, conforme o tamanho dela — o que significa que o
# "teste" do sentiment e o "teste" do stress index cobriam períodos distintos, e
# nenhum dos dois correspondia ao período em que o backtest final foi avaliado.
# Com data fixa, treino e teste querem dizer a mesma coisa em todo lugar.
#
# 31/12/2024 divide os ~5 anos de preço (a partir de ago/2021) em ~3,4 anos de
# treino e ~1,6 de teste, e cai numa virada de ano — corte em data redonda é
# mais difícil de ser acusado de ter sido escolhido depois de ver o resultado.
DATA_CORTE_TREINO_TESTE = "2024-12-31"

# Força do tilt de sentimento POR TICKER.
#
# CONGELADO em 16/08/2026 a partir de validate_signal_matrix.py rodado SÓ NO
# TREINO (até DATA_CORTE_TREINO_TESTE), sobre o corpus filtrado por
# text_filter.py e escorado pelo motor GenAI estruturado. A regra de decisão
# está impressa pelo próprio script e foi declarada antes de olhar o teste.
#
# Resultado: ZERO para os quatro ativos.
#   ISUS11: IC treino -0.129, coerente mas NEGATIVO -> contradiz a hipótese
#           direcional do projeto. IC negativo não vira tilt negativo: inverter
#           o sinal depois de ver o dado é escolher a direção pelo resultado.
#   GOVE11: direção do IC troca entre horizontes -> sem leitura estável
#   REVE11: coerente e positivo, mas IC +0.019 e apenas 3 dias com sinal em
#           1049 -> magnitude irrelevante sobre amostra inexistente
#   BOVA11: direção do IC troca entre horizontes -> sem leitura estável
#
# Consequência honesta: com esta calibração o módulo textual é INERTE, e a
# estratégia em produção equivale à configuração A3 (só circuit breaker). Isso
# é o resultado, não uma falha de configuração — mexer nesses números para
# "ligar" o módulo depois de ver o teste é exatamente o vício que o protocolo
# de congelamento existe para impedir.
#
# HISTÓRICO — a versão anterior tinha ISUS11=0.30, o tilt mais agressivo do
# modelo, justificado como "evidência forte, sobrevive Bonferroni". Aquele
# sinal foi medido sobre um corpus em que 96% dos textos do ISUS11 eram
# "Emissão de Debêntures", capturados pela keyword "emiss" que pretendia pegar
# *emissões de carbono*. A correlação provavelmente era real, mas o mecanismo
# era ciclo de crédito, não sentimento ESG: empresas emitem debêntures quando
# as condições de crédito estão boas, e essas condições movem a bolsa.
# Significância estatística não substituiu validade de construto.
SENTIMENT_TILT_STRENGTH_BASE = 0.0  # fallback para ticker fora do dict abaixo
SENTIMENT_TILT_STRENGTH_POR_TICKER = {
    "ISUS11": 0.0,
    "GOVE11": 0.0,
    "REVE11": 0.0,
    "BOVA11": 0.0,
}

# Tilt usado APENAS nos braços contrafactuais da matriz de ablação, para
# responder "e se o sinal fosse aplicado mesmo assim?". Não é usado em
# produção — ver run_ablations.py --tilt-contrafactual.
TILT_CONTRAFACTUAL_ABLACAO = 0.15

# Sentiment por ticker (alimenta o Markowitz): sempre via FinBERT-PT-BR — ver
# momentum_punch/sentiment.py. Modelo usado:
FINBERT_PTBR_MODEL = "lucas-leme/FinBERT-PT-BR"

# Ollama/Groq: usados só pelo score_stress_index() (índice de estresse via
# GDELT), que é DORMENTE no backtest de produção (o circuit breaker real usa
# o stress index objetivo do Bacen Focus). Mantidos por causa de
# run_diagnostics.py e build_sentiment_dataset.py --stress.
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"
GROQ_MODEL = "llama-3.3-70b-versatile"
