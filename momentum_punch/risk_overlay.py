"""
Overlay de gestão de risco: circuit breaker de estresse geopolítico.

- stress_index < STRESS_THRESHOLD_Z  -> modo Risk-On: usa os pesos do otimizador
  de Markowitz direto, 100% alocado entre os ETFs de risco.
- stress_index >= STRESS_THRESHOLD_Z -> modo Risk-Off: exposição total a ETFs é
  limitada a RISK_OFF_MAX_EQUITY; o capital excedente vai para o CDI.
"""
from __future__ import annotations

import pandas as pd

from . import config


def apply_circuit_breaker(
    etf_weights: pd.Series,
    stress_index: float,
    threshold: float = config.STRESS_THRESHOLD_Z,
    max_equity_risk_off: float = config.RISK_OFF_MAX_EQUITY,
) -> dict[str, float]:
    """
    Recebe os pesos ótimos entre ETFs (soma=1) e o índice de estresse do dia.
    Retorna um dict com todos os pesos finais da carteira, incluindo CDI, já
    somando 1 no total.
    """
    if stress_index is None:
        stress_index = 0.0

    if stress_index >= threshold:
        equity_scale = max_equity_risk_off
        mode = "RISK-OFF"
    else:
        equity_scale = 1.0
        mode = "RISK-ON"

    final_weights = (etf_weights * equity_scale).to_dict()
    final_weights[config.RISK_FREE] = round(1.0 - equity_scale, 10)

    final_weights["_mode"] = mode
    final_weights["_stress_index"] = stress_index
    return final_weights
