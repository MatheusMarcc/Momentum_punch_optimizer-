"""
Constrói o Índice de Estresse a partir do Bacen Focus (dispersão das
expectativas de Selic/câmbio/IPCA) — SEM precisar de LLM nem de histórico de
notícia. Isso desbloqueia testar o circuit breaker com ~2 anos de dado real
HOJE, enquanto o sentiment por ticker (que depende de RSS/Ollama) ainda tá
acumulando histórico.

Lógica: dispersão (desvio-padrão) das expectativas crescendo = mercado mais
incerto = estresse maior. Normalizo cada série via z-score móvel (trailing
window, sem look-ahead) e comprimo pra [0,1] com uma sigmoide, depois faço a
média das 3 séries disponíveis (Selic, Taxa de câmbio, IPCA).

Uso:
    python collect_data.py --only bacen_focus     (se ainda não rodou)
    python build_stress_index_focus.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOW = 60  # dias úteis pra estimar média/desvio móvel de cada série


def _zscore_to_stress(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Z-score móvel (trailing, sem look-ahead) comprimido em [0,1] via sigmoide."""
    rolling_mean = series.rolling(window, min_periods=10).mean()
    rolling_std = series.rolling(window, min_periods=10).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return 1 / (1 + np.exp(-z))


def build_stress_index(
    focus_csv: str = "data/raw/bacen_focus.csv",
    out: str = "data/processed/stress_index.csv",
) -> pd.Series:
    df = pd.read_csv(focus_csv, index_col=0, parse_dates=True)

    std_cols = [c for c in df.columns if c.endswith("_std")]
    if not std_cols:
        raise ValueError(f"Nenhuma coluna '_std' encontrada em {focus_csv} — rode collect_data.py --only bacen_focus antes")

    componentes = pd.DataFrame({col: _zscore_to_stress(df[col]) for col in std_cols})
    stress = componentes.mean(axis=1).dropna()
    stress.name = "stress_index"

    stress.to_csv(out)
    print(f"[build_stress_index_focus] {len(stress)} dias de estresse real (de {stress.index.min().date()} a {stress.index.max().date()})")
    print(f"[build_stress_index_focus] Salvo em {out}")
    print(f"\nEstatísticas: média={stress.mean():.3f}, min={stress.min():.3f}, max={stress.max():.3f}")
    print(f"Dias acima do threshold padrão (0.6): {(stress > 0.6).sum()} ({(stress > 0.6).mean():.1%})")
    return stress


if __name__ == "__main__":
    build_stress_index()
