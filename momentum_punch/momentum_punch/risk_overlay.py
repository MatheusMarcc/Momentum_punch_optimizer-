"""Normalized 0–1 stress overlay for risk-on/risk-off allocation."""
from __future__ import annotations

import math
import pandas as pd

from . import config


def apply_circuit_breaker(
    etf_weights: pd.Series,
    stress_index: float | None,
    threshold: float = config.STRESS_THRESHOLD,
    max_equity_risk_off: float = config.RISK_OFF_MAX_EQUITY,
) -> dict[str, float]:
    if etf_weights.empty:
        raise ValueError("etf_weights is empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0.0 <= max_equity_risk_off <= 1.0:
        raise ValueError("max_equity_risk_off must be in [0, 1]")

    stress = 0.0 if stress_index is None else float(stress_index)
    if not math.isfinite(stress):
        raise ValueError("stress_index must be finite")
    stress = min(1.0, max(0.0, stress))

    is_risk_off = stress >= threshold
    equity_scale = max_equity_risk_off if is_risk_off else 1.0

    clean = etf_weights.astype(float)
    clean = clean / clean.sum()
    final = (clean * equity_scale).to_dict()
    final[config.RISK_FREE] = float(1.0 - equity_scale)
    final["_mode"] = "RISK-OFF" if is_risk_off else "RISK-ON"
    final["_stress_index"] = stress
    final["_stress_scale"] = "normalized_0_1"

    investable_sum = sum(v for k, v in final.items() if not k.startswith("_"))
    if not math.isclose(investable_sum, 1.0, abs_tol=1e-8):
        raise RuntimeError("final investable weights do not sum to one")
    return final
