# Tabelas do relatório

Sinal: `data/processed/sentiment_scores_genai.csv` | custo: 10 bps | corte treino/teste: 2024-12-31
Janela de preço: 2021-08-17 a 2026-08-14


## Performance — treino

| Carteira                            | CAGR   | Volatilidade anual   |   Sharpe (excedente ao CDI) |   Sortino (excedente ao CDI) |   Calmar | Max drawdown   | Tempo submerso                           |
|:------------------------------------|:-------|:---------------------|----------------------------:|-----------------------------:|---------:|:---------------|:-----------------------------------------|
| Momentum Punch                      | 0.89%  | 14.70%               |                       -0.64 |                        -0.85 |     0.04 | -23.00%        | 98% dos dias (maior sequência: 682 dias) |
| 100% BOVA11                         | 5.77%  | 18.24%               |                       -0.22 |                        -0.35 |     0.27 | -21.03%        | 94% dos dias (maior sequência: 326 dias) |
| 60% BOVA11 / 40% CDI                | 8.65%  | 10.94%               |                       -0.22 |                        -0.35 |     0.73 | -11.92%        | 90% dos dias (maior sequência: 153 dias) |
| Pesos iguais entre ETFs             | 4.22%  | 14.88%               |                       -0.41 |                        -0.62 |     0.22 | -18.77%        | 95% dos dias (maior sequência: 418 dias) |
| ESG estática (ISUS11/GOVE11/REVE11) | 3.53%  | 14.79%               |                       -0.46 |                        -0.68 |     0.18 | -19.39%        | 95% dos dias (maior sequência: 502 dias) |
| 40% BOVA11 / 40% IVVB11 / 20% CDI   | 10.58% | 9.96%                |                       -0.08 |                        -0.11 |     0.75 | -14.16%        | 85% dos dias (maior sequência: 370 dias) |

## Performance — teste

| Carteira                            | CAGR   | Volatilidade anual   |   Sharpe (excedente ao CDI) |   Sortino (excedente ao CDI) |   Calmar | Max drawdown   | Tempo submerso                          |
|:------------------------------------|:-------|:---------------------|----------------------------:|-----------------------------:|---------:|:---------------|:----------------------------------------|
| Momentum Punch                      | 30.05% | 13.63%               |                        1.01 |                         1.75 |     4    | -7.51%         | 85% dos dias (maior sequência: 78 dias) |
| 100% BOVA11                         | 21.71% | 17.41%               |                        0.45 |                         0.68 |     1.34 | -16.26%        | 83% dos dias (maior sequência: 83 dias) |
| 60% BOVA11 / 40% CDI                | 19.15% | 10.45%               |                        0.45 |                         0.68 |     2.21 | -8.66%         | 79% dos dias (maior sequência: 83 dias) |
| Pesos iguais entre ETFs             | 16.51% | 14.27%               |                        0.2  |                         0.34 |     1.79 | -9.23%         | 85% dos dias (maior sequência: 79 dias) |
| ESG estática (ISUS11/GOVE11/REVE11) | 14.69% | 13.98%               |                        0.09 |                         0.15 |     1.79 | -8.19%         | 88% dos dias (maior sequência: 79 dias) |
| 40% BOVA11 / 40% IVVB11 / 20% CDI   | 15.31% | 9.45%                |                        0.13 |                         0.2  |     2.17 | -7.05%         | 80% dos dias (maior sequência: 66 dias) |

## Performance — completo

| Carteira                            | CAGR   | Volatilidade anual   |   Sharpe (excedente ao CDI) |   Sortino (excedente ao CDI) |   Calmar | Max drawdown   | Tempo submerso                           |
|:------------------------------------|:-------|:---------------------|----------------------------:|-----------------------------:|---------:|:---------------|:-----------------------------------------|
| Momentum Punch                      | 9.63%  | 14.37%               |                       -0.13 |                        -0.18 |     0.42 | -23.00%        | 94% dos dias (maior sequência: 694 dias) |
| 100% BOVA11                         | 10.90% | 17.95%               |                       -0.01 |                        -0.01 |     0.52 | -21.03%        | 92% dos dias (maior sequência: 326 dias) |
| 60% BOVA11 / 40% CDI                | 12.09% | 10.77%               |                       -0.01 |                        -0.01 |     1.01 | -11.92%        | 88% dos dias (maior sequência: 153 dias) |
| Pesos iguais entre ETFs             | 8.01%  | 14.67%               |                       -0.22 |                        -0.35 |     0.43 | -18.77%        | 93% dos dias (maior sequência: 418 dias) |
| ESG estática (ISUS11/GOVE11/REVE11) | 6.89%  | 14.52%               |                       -0.3  |                        -0.46 |     0.36 | -19.39%        | 94% dos dias (maior sequência: 502 dias) |
| 40% BOVA11 / 40% IVVB11 / 20% CDI   | 12.04% | 9.79%                |                       -0.02 |                        -0.03 |     0.85 | -14.16%        | 84% dos dias (maior sequência: 370 dias) |

## Significância da contribuição marginal (período de teste)

| Comparação   | O que isola                           | Dif. de retorno anual   |   Correlação entre as pernas |   t (Newey-West) |   p-valor | Conclusão              |
|:-------------|:--------------------------------------|:------------------------|-----------------------------:|-----------------:|----------:|:-----------------------|
| A2 - A0      | Texto isolado (sem overlay de risco)  | +0.14%                  |                       0.9981 |             0.23 |     0.817 | indistinguível de zero |
| A4 - A3      | Texto sobre o overlay — TESTE DA TESE | +0.02%                  |                       0.9977 |             0.03 |     0.974 | indistinguível de zero |
| A3 - A0      | Overlay de risco isolado              | +1.85%                  |                       0.9279 |             0.39 |     0.7   | indistinguível de zero |

## Cobertura do corpus por ativo

| Ticker   |   Textos relevantes | % do corpus   | Testável?                 |
|:---------|--------------------:|:--------------|:--------------------------|
| ISUS11   |                  32 | 0.3%          | NÃO (corpus insuficiente) |
| GOVE11   |                 949 | 9.4%          | sim                       |
| REVE11   |                  14 | 0.1%          | NÃO (corpus insuficiente) |
| BOVA11   |                 254 | 2.5%          | sim                       |

## Matriz de ablação

| config                 | descricao                    | cagr   |   sharpe_excedente_cdi |   sortino |   calmar | max_drawdown   | exposicao_media   |   custo_acumulado |
|:-----------------------|:-----------------------------|:-------|-----------------------:|----------:|---------:|:---------------|:------------------|------------------:|
| A0_baseline            | sem sentimento, sem risk-off | 27.25% |                   0.75 |      1.23 |     3.14 | -8.67%         | 100.0%            |               nan |
| A1_sentimento_bruto    | sentimento sem EMA           | 27.25% |                   0.75 |      1.23 |     3.14 | -8.67%         | 100.0%            |               nan |
| A2_sentimento_ema      | sentimento com EMA           | 27.25% |                   0.75 |      1.23 |     3.14 | -8.67%         | 100.0%            |               nan |
| A3_so_circuit_breaker  | só risk-off                  | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A4_completo            | sentimento + risk-off        | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A5_sem_relevancia      | sem gate de relevância       | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A6_com_relevancia      | com gate de relevância       | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A7_tilt_multiplicativo | tilt multiplicativo          | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A8_alpha_aditivo       | tilt aditivo                 | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A9_sem_custos          | sem custo de transação       | 33.19% |                   1.19 |      2.06 |     4.68 | -7.09%         | 83.3%             |               nan |
| A10_com_custos         | com custo (10 bps)           | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A11_cov_amostral       | covariância amostral         | 30.05% |                   1.01 |      1.75 |     4    | -7.51%         | 83.3%             |               nan |
| A12_cov_ledoit_wolf    | covariância Ledoit-Wolf      | 30.20% |                   1.02 |      1.76 |     4.04 | -7.48%         | 83.3%             |               nan |
