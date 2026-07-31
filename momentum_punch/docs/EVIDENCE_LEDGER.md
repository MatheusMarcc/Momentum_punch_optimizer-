# Evidence Ledger

| Claim ID | Claim | Evidence | Status |
|---|---|---|---|
| ENG-001 | Signals are executed only on the next trading session | `tests/test_temporal_integrity.py` | TESTED IN PATCH |
| ENG-002 | Higher transaction-cost assumptions cannot improve terminal wealth | `tests/test_costs_and_risk.py` | TESTED IN PATCH |
| ENG-003 | Portfolio weights obey bounds and sum to one | `tests/test_optimizer.py` | TESTED IN PATCH |
| ENG-004 | Invalid LLM scores are rejected | `tests/test_sentiment.py` | TESTED IN PATCH |
| FIN-001 | Momentum Punch outperforms a benchmark | Real-data walk-forward run required | NOT VALIDATED |
| FIN-002 | Circuit breaker reduces drawdown | A3/A4 ablation on real data required | NOT VALIDATED |
| FIN-003 | Sentiment adds alpha | A0/A1/A2/A4/A7/A8 real-data ablations required | NOT VALIDATED |
