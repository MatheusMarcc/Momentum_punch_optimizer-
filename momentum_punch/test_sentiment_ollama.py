"""
Teste rápido do backend Ollama do sentiment.py. Rode isso NA SUA MÁQUINA
(não funciona daqui do sandbox — preciso de acesso a localhost:11434).

Pré-requisitos:
    1. Instale o Ollama: https://ollama.com/download
    2. Baixe o modelo (uma vez só, ~2GB):
         ollama pull qwen2.5:3b-instruct-q4_K_M
    3. Confirme que o serviço está rodando (geralmente já sobe sozinho):
         ollama list
    4. python test_sentiment_ollama.py
"""
from momentum_punch import sentiment

textos_exemplo = [
    "Ibovespa fecha em alta de 2% puxado por bancos e commodities",
    "Investidores otimistas com corte de juros mais cedo que o esperado",
    "Volume de negociação bate recorde do mês em dia de otimismo generalizado",
]

resultado = sentiment.score_texts_for_ticker("BOVA11", textos_exemplo, provider="ollama")
print(f"\nTicker: {resultado.ticker}")
print(f"Score: {resultado.score}")
print(f"Rationale: {resultado.rationale}")

print("\n--- Teste do índice de estresse ---")
textos_estresse = [
    "Tensão eleitoral aumenta incerteza sobre política fiscal",
    "Conflito geopolítico no Oriente Médio eleva preço do petróleo",
]
stress = sentiment.score_stress_index(textos_estresse, provider="ollama")
print(f"Stress index: {stress}")
