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

# Suavização EMA do Sentiment Score (em dias)
EMA_SPAN = 5

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

# Quanto o sentiment score "empurra" o retorno esperado histórico (fator de escala)
SENTIMENT_TILT_STRENGTH = 0.15  # ex: score +1.0 -> +15 p.p. de tilt no mu anualizado do ativo

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
