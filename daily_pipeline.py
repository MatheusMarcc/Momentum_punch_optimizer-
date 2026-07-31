"""
Roda a coleta + scoring do dia numa tacada só, com log em arquivo — pensado pra
ser chamado automaticamente todo dia (Agendador de Tarefas no Windows, cron no
Linux/Mac).

Uso manual:
    python daily_pipeline.py

O que faz, na ordem:
    1. collect_data.py --only rss_news        (acumula notícia nova no CSV)
    2. build_sentiment_dataset.py             (escora só os dias novos via Ollama)

Bacen SGS/Focus e GDELT eu deixei de fora do job diário de propósito: SGS/Focus
já trazem histórico completo numa chamada só (não faz sentido rodar todo dia),
e GDELT tem rate limit agressivo — rode esses via collect_data.py manualmente
de vez em quando, não precisa ser diário.
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

LOG_PATH = Path("data/daily_pipeline.log")


def _log(msg: str):
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(nome: str, comando: list[str]) -> bool:
    _log(f"Iniciando: {nome}")
    result = subprocess.run(comando, capture_output=True, text=True)
    if result.returncode != 0:
        _log(f"FALHOU: {nome} (código {result.returncode})")
        _log(f"stderr: {result.stderr[-1000:]}")  # só o final, pra não inchar o log
        return False
    _log(f"OK: {nome}")
    return True


def main():
    _log("=== Início do pipeline diário ===")

    ok_coleta = run_step("coleta de notícias (rss_news)", [sys.executable, "collect_data.py", "--only", "rss_news"])
    if not ok_coleta:
        _log("Coleta falhou — abortando scoring pra não rodar sobre dado desatualizado.")
        return

    run_step("scoring de sentimento (Ollama)", [sys.executable, "build_sentiment_dataset.py"])

    _log("=== Fim do pipeline diário ===\n")


if __name__ == "__main__":
    main()
