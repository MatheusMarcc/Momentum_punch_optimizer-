# Methodology

## Information timeline

1. Texts and market observations are accepted only when timestamped and
   available by the declared cutoff.
2. A signal is calculated after the cutoff on trading date `t`.
3. The target portfolio is executed on the next available trading date.
4. Transaction costs are charged at execution.
5. Performance is evaluated only after execution.

## Sentiment alpha

The default specification is additive:

`mu_adjusted = mu_historical + alpha_scale * sentiment_score`

The multiplicative formulation is retained only as a registered ablation. This
prevents a positive score from mechanically making a negative `mu` more
negative.

## Risk overlay

The stress score is normalized to `[0, 1]`. It is not called a z-score. Above
the configured threshold, total equity exposure is capped and the remainder is
allocated to CDI.

## Results policy

Synthetic data validate software behavior only. Real-market performance must be
generated from timestamped sources with frozen parameters and a walk-forward
test.
