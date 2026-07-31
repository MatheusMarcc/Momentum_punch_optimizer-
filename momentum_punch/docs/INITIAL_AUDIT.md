# Initial Audit — Momentum Punch

## Verdict

The initial repository was a useful engineering prototype, but it was not yet
submission-grade quantitative research.

## Confirmed critical issues

1. **Same-day execution leakage:** weights calculated with data through date `t`
   were applied to the return on date `t`.
2. **Ambiguous sentiment transformation:** the multiplicative tilt can move a
   negative historical expected return in the opposite direction from the
   intended economic interpretation.
3. **No transaction costs or turnover accounting.**
4. **Optimizer status not audited:** any non-null solution was accepted and
   fallback usage was not exposed.
5. **Stress nomenclature conflict:** a normalized 0–1 index was named as a
   z-score threshold.
6. **Metrics were incomplete and Sharpe did not subtract the risk-free return.**
7. **Synthetic results were available, but there was no machine-readable
   evidence ledger separating engineering tests from real-market results.**

## Remediation in this patch

- next-session execution;
- additive sentiment-alpha default plus explicit ablation modes;
- transaction costs and turnover;
- excess-over-CDI Sharpe, Sortino, Calmar and underwater duration;
- strict optimizer validation and deterministic fallback;
- normalized stress terminology;
- structured LLM schema and injection-resistant document boundaries;
- temporal-integrity, cost, optimizer, risk and sentiment tests;
- CI and reproducible packaging.

## Honest status

The implementation is materially stronger, but no claim of financial
outperformance is valid until real, timestamped, auditable data are supplied and
the frozen walk-forward protocol is executed.
