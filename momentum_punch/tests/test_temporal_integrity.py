import pandas as pd

from momentum_punch import backtest, synthetic_data


def _inputs(n=180):
    prices = synthetic_data.generate_prices(n_days=n)
    cdi = synthetic_data.generate_cdi_daily_return(n_days=n, dates=prices.index)
    sentiment = synthetic_data.generate_sentiment_scores(prices.index)
    stress = pd.Series(0.0, index=prices.index)
    return prices, cdi, sentiment, stress


def test_signal_is_executed_on_next_trading_day():
    prices, cdi, sentiment, stress = _inputs()
    result = backtest.run_backtest(prices, cdi, sentiment, stress)
    assert result["weights_history"]
    for execution_date, record in result["weights_history"].items():
        signal_date = pd.Timestamp(record["_signal_date"])
        assert execution_date > signal_date


def test_future_price_change_does_not_change_past_returns():
    prices, cdi, sentiment, stress = _inputs()
    baseline = backtest.run_backtest(prices, cdi, sentiment, stress)

    changed = prices.copy()
    cutoff = changed.index[-15]
    changed.loc[cutoff:, "BOVA11"] *= 5.0
    altered = backtest.run_backtest(changed, cdi, sentiment, stress)

    common_past = baseline["portfolio_returns"].index[
        baseline["portfolio_returns"].index < cutoff
    ]
    pd.testing.assert_series_equal(
        baseline["portfolio_returns"].loc[common_past],
        altered["portfolio_returns"].loc[common_past],
    )
