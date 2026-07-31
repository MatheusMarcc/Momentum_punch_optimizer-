# Momentum Punch — Sentiment Alpha Portfolio

**Equipe MMA · Desafio Itaú Asset Quant AI 2026**

> Texto vira visão. Visão vira peso. Risco define o quanto confiar.

Momentum Punch is an auditable research pipeline for dynamic allocation among
Brazilian ETFs and CDI. It combines timestamped text classification, an
explicit sentiment-alpha transformation, constrained mean-variance allocation
and a normalized stress overlay.

## Research status

- **Engineering pipeline:** implemented and tested.
- **Synthetic demo:** available only for software validation.
- **Real-data backtest:** pending an auditable dataset.
- **Outperformance claim:** not validated.

The repository intentionally refuses to present synthetic metrics as financial
evidence.

## What changed in the winner-grade hardening patch

- removed same-day execution leakage;
- moved all decisions from signal date `t` to execution date `t+1`;
- added transaction costs, turnover and execution audit trails;
- changed the default sentiment transformation to additive alpha;
- retained multiplicative/none modes for ablation;
- validated SLSQP solver status and added deterministic fallback;
- renamed the stress input as a normalized 0–1 index;
- added excess-over-CDI Sharpe, Sortino, Calmar, monthly extremes and
  underwater duration;
- added schema validation and prompt-injection resistance for LLM scoring;
- added automated tests and GitHub Actions.

## Run

```bash
cd momentum_punch
python -m pip install -r requirements-dev.txt
pytest
python main.py
```

`main.py` prints **TESTE DE ENGENHARIA — DADOS SINTÉTICOS**. Those values must
not appear in the submission as real performance.

## Core experiment matrix

| ID | Sentiment | EMA | Stress overlay | Costs | Covariance |
|---|---|---:|---:|---:|---|
| A0 | none | no | no | yes | sample |
| A1 | raw | no | no | yes | sample |
| A2 | additive | yes | no | yes | sample |
| A3 | none | no | yes | yes | sample |
| A4 | additive | yes | yes | yes | sample |
| A7 | multiplicative | yes | yes | yes | sample |
| A8 | additive | yes | yes | yes | sample |
| A9 | additive | yes | yes | no | sample |
| A10 | additive | yes | yes | yes | sample |
| A11 | additive | yes | yes | yes | sample |
| A12 | additive | yes | yes | yes | EWMA |

Every unexecuted real-data experiment must remain marked **NOT VALIDATED**.

## Disclaimer

Research document and software prototype. Not investment advice.
