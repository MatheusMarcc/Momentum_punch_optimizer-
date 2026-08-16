"""
Gera TODAS as tabelas do relatório a partir do código, em LaTeX (booktabs) e em
Markdown, numa execução só.

Motivo de existir: a seção 8.2 do pré-relatório diz que "o repositório e o
relatório devem consumir as mesmas tabelas produzidas por código. Números não
serão digitados manualmente no PDF". Um script que gera o .tex é o que torna
essa promessa verificável — se o número no PDF diverge do que sai daqui, a
divergência é detectável.

Uso:
    python gerar_tabelas_relatorio.py
    python gerar_tabelas_relatorio.py --scores data/processed/sentiment_scores_genai.csv

Saída: data/processed/tabelas/*.tex  (\\input{} direto no Overleaf)
       data/processed/tabelas/resumo.md
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from momentum_punch import backtest, config, manifesto, text_filter
from run_real_backtest import load_prices, load_cdi, load_sentiment, load_benchmark_prices

# Na raiz, não enterrado em data/processed: é entregável, não subproduto de
# processamento. As mesmas tabelas alimentam o PDF (.tex) e o dashboard (.csv),
# de um único arquivo-fonte por tabela — número que aparece nos dois lugares não
# pode ser digitado duas vezes.
OUT_DIR = "relatorio/tabelas"

# Métricas que entram na tabela principal, na ordem em que devem aparecer
METRICAS_PRINCIPAIS = [
    "CAGR", "Volatilidade anual", "Sharpe (excedente ao CDI)",
    "Sortino (excedente ao CDI)", "Calmar", "Max drawdown", "Tempo submerso",
]


def _escapa_latex(valor) -> str:
    """% e & são comandos em LaTeX — sem escapar, a tabela quebra a compilação
    justamente porque quase toda métrica aqui é percentual. Aceita qualquer
    tipo porque as tabelas vêm de CSV, onde uma coluna pode chegar como float."""
    texto = str(valor)
    for de, para in [("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        texto = texto.replace(de, para)
    return texto


def salva_tabela(df: pd.DataFrame, nome: str, legenda: str, rotulo: str):
    os.makedirs(OUT_DIR, exist_ok=True)

    corpo = df.copy()
    corpo.columns = [_escapa_latex(str(c)) for c in corpo.columns]
    for c in corpo.columns:
        corpo[c] = corpo[c].astype(str).map(_escapa_latex)

    alinhamento = "l" + "r" * (len(corpo.columns) - 1)
    linhas = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{_escapa_latex(legenda)}}}", rf"\label{{tab:{rotulo}}}",
        rf"\begin{{tabular}}{{{alinhamento}}}", r"\toprule",
        " & ".join(corpo.columns) + r" \\", r"\midrule",
    ]
    linhas += [" & ".join(r) + r" \\" for r in corpo.astype(str).values]
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    caminho_tex = os.path.join(OUT_DIR, f"{nome}.tex")
    with open(caminho_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")

    # CSV da MESMA tabela: é o que o dashboard consome. Assim o painel e o PDF
    # não podem divergir — se divergirem, é porque alguém editou um à mão.
    caminho_csv = os.path.join(OUT_DIR, f"{nome}.csv")
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    print(f"  -> {caminho_tex} + .csv")
    return df.to_markdown(index=False)


def tabela_performance(result, cdi, periodo: str, inicio, fim) -> pd.DataFrame:
    """Estratégia contra os cinco benchmarks da seção 5.2."""
    metrics = backtest.metricas_no_periodo(result, cdi, inicio, fim)
    metrics.pop("Benchmark", None)  # alias duplicado
    linhas = []
    for nome, m in metrics.items():
        linha = {"Carteira": nome}
        linha.update({k: m.get(k, "n/a") for k in METRICAS_PRINCIPAIS})
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_significancia(prices, cdi, sent, stress, custo_bps: float, inicio, fim) -> pd.DataFrame:
    """
    Contribuição marginal de cada módulo, com teste de significância PAREADO.

    Por que pareado: as configurações comparadas compartilham quase todo o
    risco (A3 e A4 diferem só pelo tilt e têm correlação diária ~0,9998).
    Comparar os Sharpes isolados não responde nada — o erro-padrão de cada um é
    da ordem de 0,6 e engole qualquer diferença. Já a série de DIFERENÇAS entre
    as duas é estimada com precisão muito maior, e é ela que diz se o módulo
    contribui. Newey-West (5 lags) corrige a autocorrelação.

    Os braços de texto usam o tilt CONTRAFACTUAL: a calibração congelada deu 0
    em todos os ativos, então com o valor de produção A2 seria idêntica a A0 e
    o teste seria 0/0. A pergunta que esta tabela responde é "se o sinal fosse
    aplicado, contribuiria?" — e a resposta continua sendo não.
    """
    import numpy as np
    import statsmodels.api as sm

    k = config.TILT_CONTRAFACTUAL_ABLACAO
    tilt_original = dict(config.SENTIMENT_TILT_STRENGTH_POR_TICKER)
    base_original = config.SENTIMENT_TILT_STRENGTH_BASE
    config.SENTIMENT_TILT_STRENGTH_POR_TICKER = {t: k for t in config.TICKERS}
    config.SENTIMENT_TILT_STRENGTH_BASE = k

    try:
        neutro = pd.DataFrame(0.0, index=prices.index, columns=config.TICKERS)
        zero = pd.Series(0.0, index=prices.index)
        configs = {"A0": (neutro, zero), "A2": (sent, zero), "A3": (neutro, stress), "A4": (sent, stress)}

        retornos = {}
        for nome, (s, st) in configs.items():
            r = backtest.run_backtest(prices, cdi, s, st, benchmark="60_40", transaction_cost_bps=custo_bps)
            curva = r["equity_curve"].loc[inicio:fim]
            retornos[nome] = curva.pct_change().dropna()

        linhas = []
        comparacoes = [
            ("A2 - A0", "A2", "A0", "Texto isolado (sem overlay de risco)"),
            ("A4 - A3", "A4", "A3", "Texto sobre o overlay — TESTE DA TESE"),
            ("A3 - A0", "A3", "A0", "Overlay de risco isolado"),
        ]
        for rotulo, a, b, descricao in comparacoes:
            d = (retornos[a] - retornos[b]).dropna()
            corr = float(np.corrcoef(retornos[a], retornos[b])[0, 1])
            modelo = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            p = float(modelo.pvalues[0])
            linhas.append({
                "Comparação": rotulo,
                "O que isola": descricao,
                "Dif. de retorno anual": f"{d.mean() * 252:+.2%}",
                "Correlação entre as pernas": f"{corr:.4f}",
                "t (Newey-West)": f"{modelo.tvalues[0]:+.2f}",
                "p-valor": f"{p:.3f}",
                "Conclusão": "contribui" if p < 0.05 and d.mean() > 0 else
                             ("prejudica" if p < 0.05 else "indistinguível de zero"),
            })
        return pd.DataFrame(linhas)
    finally:
        config.SENTIMENT_TILT_STRENGTH_POR_TICKER = tilt_original
        config.SENTIMENT_TILT_STRENGTH_BASE = base_original


def tabela_cobertura_corpus(cvm_csv: str = "data/raw/cvm_fatos_relevantes.csv") -> pd.DataFrame:
    """Quantos textos cada ticker realmente recebe depois do filtro corrigido.

    Esta tabela é a mais importante do relatório em termos de honestidade: ela
    mostra que dois dos quatro ativos praticamente não têm corpus, e portanto
    que a tese não é testável para eles com esta fonte."""
    if not os.path.exists(cvm_csv):
        return pd.DataFrame([{"Ticker": t, "Textos": "fonte ausente"} for t in config.TICKERS])

    df = pd.read_csv(cvm_csv, low_memory=False)
    col = next((c for c in df.columns if "assunto" in c.lower()), None)
    textos = df[col].fillna("").astype(str).tolist() if col else []

    linhas = []
    for t in config.TICKERS:
        n = sum(1 for x in textos if text_filter.texto_e_relevante(x, t))
        linhas.append({
            "Ticker": t,
            "Textos relevantes": str(n),
            "% do corpus": f"{n / len(textos):.1%}" if textos else "n/a",
            "Testável?": "sim" if n >= 200 else "NÃO (corpus insuficiente)",
        })
    return pd.DataFrame(linhas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/processed/sentiment_scores.csv")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    prices = load_prices()
    cdi = load_cdi().reindex(prices.index).ffill().fillna(0.0)
    sent = load_sentiment(args.scores, prices.index)
    stress = load_sentiment("data/processed/stress_index.csv", prices.index).iloc[:, 0]

    result = backtest.run_backtest(
        prices, cdi, sent, stress, benchmark="60_40",
        transaction_cost_bps=args.cost_bps, benchmark_prices=load_benchmark_prices(),
    )

    corte = config.DATA_CORTE_TREINO_TESTE
    md = [f"# Tabelas do relatório\n",
          f"Sinal: `{args.scores}` | custo: {args.cost_bps:.0f} bps | corte treino/teste: {corte}",
          f"Janela de preço: {prices.index.min().date()} a {prices.index.max().date()}\n"]

    print("Gerando tabelas:")
    for periodo, ini, fim in [("treino", None, corte), ("teste", corte, None), ("completo", None, None)]:
        df = tabela_performance(result, cdi, periodo, ini, fim)
        texto_md = salva_tabela(
            df, f"performance_{periodo}",
            f"Desempenho da estratégia contra os benchmarks — período de {periodo}.",
            f"performance-{periodo}",
        )
        md += [f"\n## Performance — {periodo}\n", texto_md]

    df_sig = tabela_significancia(prices, cdi, sent, stress, args.cost_bps, corte, None)
    texto_md = salva_tabela(
        df_sig, "significancia",
        "Contribuição marginal de cada módulo no período de teste, com teste pareado "
        "e erro-padrão de Newey-West.",
        "significancia",
    )
    md += ["\n## Significância da contribuição marginal (período de teste)\n", texto_md]

    df_cob = tabela_cobertura_corpus()
    texto_md = salva_tabela(
        df_cob, "cobertura_corpus",
        "Cobertura textual por ativo após o filtro de relevância corrigido.",
        "cobertura-corpus",
    )
    md += ["\n## Cobertura do corpus por ativo\n", texto_md]

    if os.path.exists("data/processed/ablation_results.csv"):
        df_abl = pd.read_csv("data/processed/ablation_results.csv")
        texto_md = salva_tabela(
            df_abl, "ablacoes", "Matriz de ablação A0–A12.", "ablacoes",
        )
        md += ["\n## Matriz de ablação\n", texto_md]

    caminho_md = os.path.join(OUT_DIR, "resumo.md")
    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"  -> {caminho_md}")

    manifesto.salvar(
        manifesto.gerar(
            experimento="tabelas_do_relatorio",
            categoria="VERIFICADO",
            entradas={"precos": "data/raw/etf_prices.csv", "cdi": "data/raw/bacen_sgs.csv",
                      "sentiment": args.scores, "stress": "data/processed/stress_index.csv"},
            parametros_execucao={"custo_bps": args.cost_bps, "corte": corte},
            saidas=[caminho_md],
        ),
        os.path.join(OUT_DIR, "manifesto_tabelas.json"),
    )


if __name__ == "__main__":
    main()
