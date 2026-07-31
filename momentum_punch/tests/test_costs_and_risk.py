import numpy as np
import pandas as pd

from momentum_punch import backtest, risk_overlay, synthetic_data


def test_circuit_breaker_sums_to_one():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    final = risk_overlay.apply_circuit_breaker(
        weights, stress_index=0.9, threshold=0.6, max_equity_risk_off=0.3
    )
    total = sum(v for k, v in final.items() if not k.startswith("_"))
    assert np.isclose(total, 1.0)
    assert np.isclose(final["CDI"], 0.7)


def test_costs_do_not_improve_terminal_value():
    prices = synthetic_data.generate_prices(n_days=180)
    cdi = synthetic_data.generate_cdi_daily_return(n_days=len(prices), dates=prices.index)
    sentiment = synthetic_data.generate_sentiment_scores(prices.index)
    stress = synthetic_data.generate_stress_index(prices.index)

    no_cost = backtest.run_backtest(
        prices, cdi, sentiment, stress, transaction_cost_bps=0.0
    )
    with_cost = backtest.run_backtest(
        prices, cdi, sentiment, stress, transaction_cost_bps=25.0
    )
    assert with_cost["equity_curve"].iloc[-1] <= no_cost["equity_curve"].iloc[-1] + 1e-12
