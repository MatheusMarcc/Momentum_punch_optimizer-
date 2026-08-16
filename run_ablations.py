"""
Matriz de ablação A0-A12 — a matriz completa da Tabela 3 do pré-relatório.

Cada linha isola UM eixo do desenho, mantendo o resto fixo, pra responder
"quanto esse módulo contribui?" em vez de "o sistema todo é bom?". Os eixos:

  EMA           A1 (bruto) vs A2 (suavizado)
  circuit break A2 (só sentimento) vs A4 (sentimento + overlay)
  relevância    A5 (sem gate) vs A6 (com gate)
  forma do tilt A7 (multiplicativo) vs A8 (aditivo)
  custos        A9 (sem) vs A10 (com)
  covariância   A11 (amostral) vs A12 (Ledoit-Wolf)

Uso:
    python run_ablations.py                              # janela inteira
    python run_ablations.py --periodo teste              # só o período de teste congelado
    python run_ablations.py --scores data/processed/sentiment_scores_genai.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from momentum_punch import backtest, config, manifesto, sentiment
from run_real_backtest import load_prices, load_cdi, load_sentiment

CUSTO_PADRAO_BPS = 10.0


def _carrega_variantes_de_sinal(scores_path: str, price_index: pd.DatetimeIndex) -> dict:
    """
    Devolve as três formas do sinal que a matriz precisa:
      ema      — o score de produção (com gate de relevância e EMA)
      bruto    — mesmo score, SEM EMA (eixo A1 vs A2)
      sem_gate — sentimento puro, sem multiplicar pela relevância (eixo A5 vs A6)

    O 'sem_gate' é reconstruído dividindo o score cru pela relevância, porque o
    que fica gravado é o produto (sentimento * relevância). Onde a relevância é
    0 o sentimento é irrecuperável — mas nesses casos o texto foi julgado
    irrelevante, e tratar como neutro é a leitura certa, não uma perda.
    """
    neutro = pd.DataFrame(0.0, index=price_index, columns=config.TICKERS)
    variantes = {"ema": neutro, "bruto": neutro, "sem_gate": neutro}

    if not os.path.exists(scores_path):
        print(f"[run_ablations] {scores_path} não existe — todas as configs rodam com sentimento neutro")
        return variantes

    variantes["ema"] = load_sentiment(scores_path, price_index)

    raw_path = scores_path.replace(".csv", "_raw.csv")
    rel_path = scores_path.replace(".csv", "_relevancia.csv")

    if os.path.exists(raw_path):
        variantes["bruto"] = load_sentiment(raw_path, price_index)
        if os.path.exists(rel_path):
            raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            rel = pd.read_csv(rel_path, index_col=0, parse_dates=True).reindex(raw.index)
            sem_gate = (raw / rel.where(rel > 0)).fillna(0.0)
            sem_gate = sem_gate.apply(sentiment.apply_ema)
            tmp = scores_path.replace(".csv", "_sem_gate_tmp.csv")
            sem_gate.to_csv(tmp)
            variantes["sem_gate"] = load_sentiment(tmp, price_index)
            os.remove(tmp)
        else:
            print(f"[run_ablations] {rel_path} não existe — A5 (sem gate) cai pro mesmo sinal de A6")
            variantes["sem_gate"] = variantes["ema"]
    else:
        print(f"[run_ablations] {raw_path} não existe — A1 (bruto) cai pro mesmo sinal de A2")
        variantes["bruto"] = variantes["ema"]
        variantes["sem_gate"] = variantes["ema"]

    return variantes


def monta_configs(sinal: dict, stress_real: pd.Series, stress_zero: pd.Series) -> dict:
    """id -> (descrição, kwargs de run_backtest). Configs repetidas entre eixos
    (A4 == A6 == A8 == A10 == A11) são executadas mesmo assim: cada eixo precisa
    do seu par explícito na tabela, e o custo de recomputar é baixo."""
    base = dict(transaction_cost_bps=CUSTO_PADRAO_BPS, tilt_mode="aditivo", cov_estimator="amostral")

    def cfg(sent, stress, **over):
        return {"sentiment_scores": sent, "stress_index": stress, **base, **over}

    return {
        "A0_baseline":            ("sem sentimento, sem risk-off",     cfg(sinal["ema"] * 0, stress_zero)),
        "A1_sentimento_bruto":    ("sentimento sem EMA",               cfg(sinal["bruto"], stress_zero)),
        "A2_sentimento_ema":      ("sentimento com EMA",               cfg(sinal["ema"], stress_zero)),
        "A3_so_circuit_breaker":  ("só risk-off",                      cfg(sinal["ema"] * 0, stress_real)),
        "A4_completo":            ("sentimento + risk-off",            cfg(sinal["ema"], stress_real)),
        "A5_sem_relevancia":      ("sem gate de relevância",           cfg(sinal["sem_gate"], stress_real)),
        "A6_com_relevancia":      ("com gate de relevância",           cfg(sinal["ema"], stress_real)),
        "A7_tilt_multiplicativo": ("tilt multiplicativo",              cfg(sinal["ema"], stress_real, tilt_mode="multiplicativo")),
        "A8_alpha_aditivo":       ("tilt aditivo",                     cfg(sinal["ema"], stress_real)),
        "A9_sem_custos":          ("sem custo de transação",           cfg(sinal["ema"], stress_real, transaction_cost_bps=0.0)),
        "A10_com_custos":         (f"com custo ({CUSTO_PADRAO_BPS:.0f} bps)", cfg(sinal["ema"], stress_real)),
        "A11_cov_amostral":       ("covariância amostral",             cfg(sinal["ema"], stress_real)),
        "A12_cov_ledoit_wolf":    ("covariância Ledoit-Wolf",          cfg(sinal["ema"], stress_real, cov_estimator="ledoit_wolf")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/processed/sentiment_scores.csv")
    parser.add_argument("--periodo", default="tudo", choices=["tudo", "treino", "teste"])
    parser.add_argument("--mu-method", default="tilt_linear", choices=["tilt_linear", "black_litterman"])
    parser.add_argument("--out", default="data/processed/ablation_results.csv")
    parser.add_argument("--tilt-contrafactual", action="store_true",
                        help="força um tilt uniforme (config.TILT_CONTRAFACTUAL_ABLACAO) nos braços "
                             "de sentimento. Necessário quando a calibração congelada deu 0 em todos "
                             "os ativos: sem isso, os braços com sentimento viram cópias do baseline "
                             "e a matriz não mede nada. NÃO é a configuração de produção.")
    args = parser.parse_args()

    if args.tilt_contrafactual:
        k = config.TILT_CONTRAFACTUAL_ABLACAO
        config.SENTIMENT_TILT_STRENGTH_BASE = k
        config.SENTIMENT_TILT_STRENGTH_POR_TICKER = {t: k for t in config.TICKERS}
        print(f"[run_ablations] MODO CONTRAFACTUAL: tilt={k} em todos os ativos.")
        print("[run_ablations] Estes números respondem 'e se o sinal fosse aplicado?', não")
        print("[run_ablations] descrevem a estratégia calibrada (que tem tilt 0 por decisão de treino).\n")

    prices = load_prices()
    cdi = load_cdi().reindex(prices.index).ffill().fillna(0.0)
    sinal = _carrega_variantes_de_sinal(args.scores, prices.index)

    try:
        stress_real = load_sentiment("data/processed/stress_index.csv", prices.index).iloc[:, 0]
    except FileNotFoundError:
        print("[run_ablations] stress_index.csv não encontrado — risk-off desligado em todas as configs")
        stress_real = pd.Series(0.0, index=prices.index)
    stress_zero = pd.Series(0.0, index=prices.index)

    corte = config.DATA_CORTE_TREINO_TESTE
    inicio, fim = {"tudo": (None, None), "treino": (None, corte), "teste": (corte, None)}[args.periodo]
    print(f"[run_ablations] período: {args.periodo}" + (f" (corte {corte})" if args.periodo != "tudo" else ""))
    print(f"[run_ablations] sinal: {args.scores}\n")

    linhas = []
    for nome, (descricao, kwargs) in monta_configs(sinal, stress_real, stress_zero).items():
        resultado = backtest.run_backtest(prices, cdi, mu_method=args.mu_method, benchmark="60_40", **kwargs)
        m = backtest.metricas_no_periodo(resultado, cdi, inicio, fim)["Momentum Punch"]
        linhas.append({
            "config": nome,
            "descricao": descricao,
            "cagr": m["CAGR"],
            "sharpe_excedente_cdi": m["Sharpe (excedente ao CDI)"],
            "sortino": m["Sortino (excedente ao CDI)"],
            "calmar": m["Calmar"],
            "max_drawdown": m["Max drawdown"],
            "exposicao_media": m.get("Exposição média a risco", "n/a"),
            "custo_acumulado": m.get("Custo acumulado (giro)", "n/a"),
        })
        print(f"  {nome:24s} Sharpe={m['Sharpe (excedente ao CDI)']:>6s}  CAGR={m['CAGR']:>8s}  MDD={m['Max drawdown']:>8s}")

    df = pd.DataFrame(linhas)
    df.to_csv(args.out, index=False)
    print(f"\nSalvo em {args.out}")

    manifesto.salvar(
        manifesto.gerar(
            experimento=f"ablacao_A0_A12_{args.periodo}",
            categoria="VERIFICADO",
            entradas={
                "precos": "data/raw/etf_prices.csv",
                "cdi": "data/raw/bacen_sgs.csv",
                "sentiment": args.scores,
                "sentiment_bruto": args.scores.replace(".csv", "_raw.csv"),
                "relevancia": args.scores.replace(".csv", "_relevancia.csv"),
                "stress": "data/processed/stress_index.csv",
            },
            parametros_execucao={
                "periodo": args.periodo,
                "mu_method": args.mu_method,
                "custo_padrao_bps": CUSTO_PADRAO_BPS,
                "configs": list(df["config"]),
            },
            metricas={r["config"]: r for r in linhas},
            saidas=[args.out],
        ),
        f"data/processed/manifesto_ablacao_{args.periodo}.json",
    )

    # ---- Critério de aceitação (seção 6.1) --------------------------------
    # São DUAS comparações, não uma. Passar em (a) e falhar em (b) significa
    # que o ganho veio do overlay de risco e nenhum do texto — conclusão
    # diferente, e é ela que precisa ser reportada.
    def _s(nome: str) -> float:
        return float(df[df["config"] == nome]["sharpe_excedente_cdi"].iloc[0])

    a0, a3, a4 = _s("A0_baseline"), _s("A3_so_circuit_breaker"), _s("A4_completo")
    print(f"\n--- Critério de aceitação (6.1), período={args.periodo}, Sharpe excedente ao CDI ---")
    print(f"A0 (baseline) {a0:+.2f} | A3 (só risk-off) {a3:+.2f} | A4 (completo) {a4:+.2f}")
    print(f"(a) A4 vs A0: {a4 - a0:+.2f} " + ("-> ATENDIDO" if a4 > a0 else "-> NÃO atendido"))
    if a4 > a3:
        print(f"(b) A4 vs A3: {a4 - a3:+.2f} -> ATENDIDO: o texto agrega sobre o risk-off sozinho.")
    else:
        print(f"(b) A4 vs A3: {a4 - a3:+.2f} -> NÃO atendido: o ganho vem do risk-off; o texto não agrega\n"
              f"    nessa janela. Resultado negativo a REPORTAR, não a esconder.")


if __name__ == "__main__":
    main()
