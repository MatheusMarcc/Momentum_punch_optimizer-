"""Central configuration for the Momentum Punch research pipeline.

Values here are hypotheses to be validated, not claims of optimality.
"""
from __future__ import annotations

TICKERS = ["ISUS11", "GOVE11", "REVE11", "BOVA11"]
RISK_FREE = "CDI"

TICKER_THEMES = {
    "ISUS11": "sustentabilidade, emissões, clima, ESG e ISE",
    "GOVE11": "governança corporativa, conselho, diretoria e integridade",
    "REVE11": "transição energética, energia limpa, renováveis e mobilidade elétrica",
    "BOVA11": "Ibovespa, atividade econômica, inflação, fiscal e risco Brasil",
}
STRESS_THEMES = (
    "eleições, conflitos armados, choques globais, crise cambial, "
    "instabilidade fiscal e risco geopolítico"
)

TRADING_DAYS = 252
EMA_SPAN = 5
COV_LOOKBACK_DAYS = 60
MIN_OBSERVATIONS = 40

RISK_AVERSION = 3.0
MAX_WEIGHT_PER_ETF = 0.50
MIN_WEIGHT_PER_ETF = 0.0
COVARIANCE_RIDGE = 1e-8

# Additive alpha avoids reversing the intended effect when historical mu is negative.
SENTIMENT_ALPHA_MODE = "additive"  # additive | multiplicative | none
SENTIMENT_ALPHA_ANNUAL = 0.04
SENTIMENT_TILT_STRENGTH = 0.15  # retained only for the multiplicative ablation

STRESS_THRESHOLD = 0.60
RISK_OFF_MAX_EQUITY = 0.30

REBALANCE_FREQ = "W-FRI"
EXECUTION_LAG_DAYS = 1
TRANSACTION_COST_BPS = 10.0  # scenario assumption; always report sensitivity

SENTIMENT_MIN = -1.0
SENTIMENT_MAX = 1.0
MAX_TEXTS_PER_REQUEST = 40
MAX_CHARS_PER_TEXT = 1200

SENTIMENT_PROVIDER = "ollama"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Backward-compatible alias. New code should use STRESS_THRESHOLD.
STRESS_THRESHOLD_Z = STRESS_THRESHOLD
