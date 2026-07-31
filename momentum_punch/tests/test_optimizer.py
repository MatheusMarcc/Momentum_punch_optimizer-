import numpy as np
import pandas as pd

from momentum_punch.optimizer import adjust_mu_by_sentiment, optimize_weights


def test_additive_sentiment_has_expected_direction():
    mu = pd.Series({"A": -0.10, "B": 0.10})
    scores = pd.Series({"A": 1.0, "B": -1.0})
    adjusted = adjust_mu_by_sentiment(mu, scores, mode="additive", additive_alpha_annual=0.04)
    assert adjusted["A"] > mu["A"]
    assert adjusted["B"] < mu["B"]


def test_optimizer_constraints():
    assets = ["A", "B", "C"]
    mu = pd.Series([0.08, 0.06, 0.04], index=assets)
    sigma = pd.DataFrame(np.eye(3) * 0.04, index=assets, columns=assets)
    weights = optimize_weights(mu, sigma, max_weight=0.5)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert (weights <= 0.5 + 1e-8).all()
