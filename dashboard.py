"""
Dashboard do Momentum Punch — implementação real do "Dashboard Interativo"
descrito no deck original, com painel de execução: botões que rodam
os scripts reais do pipeline via subprocess, sem precisar do terminal.
Depois de cada execução, os dados exibidos abaixo recarregam automaticamente
(Streamlit re-roda o script e relê os CSVs/JSON, que já foram atualizados).

Uso:
    pip install streamlit plotly
    streamlit run dashboard.py

Fontes que o dashboard lê (os botões do painel geram esses arquivos sozinhos —
não precisa rodar nada pelo terminal antes, a não ser que prefira):
    data/raw/rss_news.csv              <- botão "Notícias agora"
    data/processed/sentiment_scores.csv <- botão "Escorar sentiment"
    data/processed/stress_index.csv     <- botão "Construir Índice de Estresse"
    data/processed/equity_curves.csv    <- botão "Rodar backtest"
    data/processed/ablation_results.csv <- botão "Rodar ablação"
    data/decisao_atual.json             <- botão "Puxar notícia AGORA"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from momentum_punch import config

st.set_page_config(page_title="Momentum Punch — Dashboard", layout="wide", page_icon="■")

# ---------------------------------------------------------------------------
# CSS — Dark Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
        border-right: 1px solid #333333;
    }

    [data-testid="stSidebar"] * {
        color: #e5e5e5 !important;
    }

    .main-header {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 2.4rem;
        font-weight: 300;
        letter-spacing: 0.1em;
        color: #ffffff;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }

    .subtitle {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 0.9rem;
        font-weight: 300;
        letter-spacing: 0.05em;
        color: #999999;
        margin-bottom: 2rem;
    }

    h2, h3 {
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #ffffff !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.7rem !important;
        font-weight: 300 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #999999 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d5f2e 100%);
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 300;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.8rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2d5f2e 0%, #1f4420 100%);
        border-color: #2d5f2e;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #4a1414 0%, #8B0000 100%) !important;
        border-color: #8B0000 !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #8B0000 0%, #5a0d0d 100%) !important;
    }

    hr {
        border-color: #333333;
        margin: 2rem 0;
    }

    p, span, label, li {
        color: #e5e5e5 !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid #333333 !important;
        border-radius: 8px;
        background: rgba(255,255,255,0.02);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        color: #999999;
        background-color: transparent;
        border-bottom: 2px solid transparent;
        padding: 0.5rem 0;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 300;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .stTabs [aria-selected="true"] {
        color: #2d5f2e;
        border-bottom: 2px solid #2d5f2e;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #333333;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Momentum Punch</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sentiment Alpha Portfolio — Desafio Itaú Asset Quant AI 2026 · Equipe MMA</p>', unsafe_allow_html=True)

# Tema Plotly escuro, consistente com o resto da interface
PLOTLY_THEME = dict(
    plot_bgcolor='#0a0a0a',
    paper_bgcolor='#0a0a0a',
    font=dict(family='Helvetica Neue, Arial', size=11, color='#e5e5e5'),
    xaxis=dict(showgrid=True, gridcolor='#333333', gridwidth=0.5, color='#999999', zerolinecolor='#333333'),
    yaxis=dict(showgrid=True, gridcolor='#333333', gridwidth=0.5, color='#999999', zerolinecolor='#333333'),
    legend=dict(bgcolor='rgba(26,26,26,0.8)', bordercolor='#333333', borderwidth=1, font=dict(color='#e5e5e5')),
    margin=dict(t=30, b=30, l=10, r=10),
)

PALETA = ['#2d5f2e', '#8B0000', '#666666', '#4a7c4c', '#a04040', '#999999']


# ---------------------------------------------------------------------------
def _carrega_csv_seguro(path: str, **kwargs) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        st.warning(f"Falha ao ler {path}: {exc}")
        return None


def _carrega_json_seguro(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _roda_comando(cmd: list[str], label: str, timeout: int = 1800) -> bool:
    """Executa um script do pipeline via subprocess, mostra spinner + output
    real (stdout/stderr) num expander, e devolve se rodou com sucesso.
    timeout generoso (30 min padrão) porque alguns passos (CVM multi-ano,
    backtest de 3 anos) demoram de verdade — não é instantâneo."""
    with st.spinner(f"Rodando: {label}..."):
        try:
            resultado = subprocess.run(
                [sys.executable] + cmd,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            st.error(f"{label}: excedeu o tempo limite ({timeout}s) — pode precisar rodar pelo terminal pra algo tão longo.")
            return False

    sucesso = resultado.returncode == 0
    icone = "✅" if sucesso else "❌"
    with st.expander(f"{icone} {label} — output", expanded=not sucesso):
        if resultado.stdout:
            st.code(resultado.stdout[-4000:], language="text")
        if resultado.stderr:
            st.code(resultado.stderr[-4000:], language="text")
    if sucesso:
        st.success(f"{label}: concluído.")
    else:
        st.error(f"{label}: falhou (código {resultado.returncode}).")
    return sucesso


# ---------------------------------------------------------------------------
# Sidebar — painel de execução (dispensa terminal: cada botão roda o script real)

st.sidebar.markdown("### Painel de execução")
st.sidebar.caption("Cada botão roda o script real via subprocess e recarrega os dados ao lado.")

with st.sidebar.expander("1 · Dados de mercado", expanded=False):
    if st.button("Preço dos ETFs (yfinance)", key="btn_precos"):
        _roda_comando(["fetch_prices_yfinance.py"], "Preço dos ETFs")
        st.rerun()
    if st.button("CDI / Selic / IPCA (Bacen SGS)", key="btn_sgs"):
        _roda_comando(["collect_data.py", "--only", "bacen_sgs"], "Bacen SGS")
        st.rerun()
    if st.button("Expectativas macro (Bacen Focus)", key="btn_focus"):
        _roda_comando(["collect_data.py", "--only", "bacen_focus"], "Bacen Focus")
        st.rerun()
    if st.button("Construir Índice de Estresse", key="btn_estresse"):
        _roda_comando(["build_stress_index_focus.py"], "Índice de Estresse (Focus)")
        st.rerun()

with st.sidebar.expander("2 · Notícia e sentimento", expanded=False):
    if st.button("Notícias agora (RSS)", key="btn_rss"):
        _roda_comando(["collect_data.py", "--only", "rss_news"], "Coleta RSS")
        st.rerun()
    if st.button("Histórico CVM (2023-2026)", key="btn_cvm"):
        _roda_comando(["pull_cvm_historical.py", "--anos", "2023", "2024", "2025", "2026"], "Histórico CVM")
        st.rerun()
    if st.button("Escorar sentiment (FinBERT, via CVM)", key="btn_sentiment"):
        _roda_comando(["build_sentiment_dataset.py", "--source", "cvm", "--force"], "Sentiment (FinBERT)")
        st.rerun()

with st.sidebar.expander("3 · Backtest e ablação", expanded=False):
    if st.button("Rodar backtest (tilt linear)", key="btn_bt_linear"):
        _roda_comando(["run_real_backtest.py"], "Backtest")
        st.rerun()
    if st.button("Rodar backtest (Black-Litterman)", key="btn_bt_bl"):
        _roda_comando(["run_real_backtest.py", "--mu-method", "black_litterman"], "Backtest (Black-Litterman)")
        st.rerun()
    if st.button("Rodar ablação (A0/A2/A3/A4, 10bps)", key="btn_ablacao"):
        _roda_comando(["run_ablations.py", "--cost-bps", "10"], "Matriz de ablação")
        st.rerun()

with st.sidebar.expander("4 · Decisão ao vivo", expanded=True):
    if st.button("Puxar notícia AGORA e recalcular", key="btn_live", type="primary"):
        _roda_comando(["run_live.py", "--json", "data/decisao_atual.json"], "Decisão ao vivo")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Documento/dashboard de pesquisa. Não constitui recomendação de investimento.")


# ---------------------------------------------------------------------------
# Conteúdo principal — abas

tab_res, tab_live, tab_news, tab_sent, tab_bt, tab_abl = st.tabs([
    "Resultados finais", "Decisão ao vivo", "Notícias", "Sentiment", "Backtest", "Ablação"
])

# --- Resultados finais -----------------------------------------------------
# Lê os MESMOS CSVs que geram os .tex do relatório (relatorio/tabelas/), pra o
# painel e o PDF não poderem divergir. Nada aqui é recalculado ou digitado.
with tab_res:
    TAB_DIR = "relatorio/tabelas"
    perf_teste = _carrega_csv_seguro(f"{TAB_DIR}/performance_teste.csv")
    perf_completo = _carrega_csv_seguro(f"{TAB_DIR}/performance_completo.csv")
    perf_treino = _carrega_csv_seguro(f"{TAB_DIR}/performance_treino.csv")
    significancia = _carrega_csv_seguro(f"{TAB_DIR}/significancia.csv")
    cobertura = _carrega_csv_seguro(f"{TAB_DIR}/cobertura_corpus.csv")

    if perf_teste is None:
        st.info("Rode `python gerar_tabelas_relatorio.py --scores data/processed/sentiment_scores_genai.csv` "
                "pra gerar as tabelas do relatório.")
    else:
        col_sharpe = "Sharpe (excedente ao CDI)"

        def _linha(df, nome="Momentum Punch"):
            achado = df[df["Carteira"] == nome]
            return achado.iloc[0] if len(achado) else None

        mp_teste, mp_completo = _linha(perf_teste), _linha(perf_completo)

        st.subheader("Período de teste congelado (a partir de 31/12/2024)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe (excedente ao CDI)", mp_teste[col_sharpe])
        c2.metric("CAGR", mp_teste["CAGR"])
        c3.metric("Max drawdown", mp_teste["Max drawdown"])

        if mp_completo is not None:
            st.warning(
                f"**Na janela completa de 5 anos o resultado se inverte:** Sharpe "
                f"{mp_completo[col_sharpe]} e CAGR {mp_completo['CAGR']}, contra "
                f"{_linha(perf_completo, '100% BOVA11')[col_sharpe]} do BOVA11 e "
                f"{_linha(perf_completo, '60% BOVA11 / 40% CDI')[col_sharpe]} do 60/40. "
                "O desempenho do período de teste é dependente de regime, não uma propriedade "
                "estável da estratégia — as duas tabelas precisam ser lidas juntas."
            )

        st.markdown("---")
        periodo = st.radio("Período", ["Teste", "Treino", "Completo"], horizontal=True, key="res_periodo")
        st.dataframe({"Teste": perf_teste, "Treino": perf_treino, "Completo": perf_completo}[periodo],
                     use_container_width=True, hide_index=True)
        st.caption("Sharpe e Sortino são calculados sobre o retorno EXCEDENTE ao CDI. "
                   "A razão retorno/volatilidade crua, frequentemente reportada como 'Sharpe', "
                   "é cerca de 2,5x maior nesta janela e não aparece aqui.")

        if significancia is not None:
            st.markdown("---")
            st.subheader("A tese se sustenta?")
            st.dataframe(significancia, use_container_width=True, hide_index=True)
            st.caption(
                "Teste PAREADO: as configurações comparadas têm correlação diária ~0,998, então a "
                "série de diferenças é estimada com precisão muito maior que os Sharpes isolados. "
                "A linha A4−A3 é o teste da tese — mede o que o módulo textual acrescenta sobre o "
                "overlay de risco rodando sozinho."
            )

        if cobertura is not None:
            st.markdown("---")
            st.subheader("Por que a tese não pôde ser testada em todo o universo")
            st.dataframe(cobertura, use_container_width=True, hide_index=True)
            st.caption(
                "Fatos relevantes da CVM, após o filtro de relevância. O REVE11 replica um índice "
                "de empresas americanas, fora da jurisdição da CVM — para ele a fonte é "
                "estruturalmente inadequada, não apenas escassa."
            )

        st.markdown("---")
        tilts = config.SENTIMENT_TILT_STRENGTH_POR_TICKER
        if all(v == 0 for v in tilts.values()):
            st.error(
                "**Calibração congelada: tilt de sentimento = 0 em todos os ativos.** Nenhum "
                "ativo passou no critério definido no treino (IC coerente entre horizontes, "
                "positivo e de magnitude relevante). A estratégia avaliada equivale, na prática, "
                "ao overlay de risco isolado. Ver o histórico em `momentum_punch/config.py`."
            )
        else:
            st.success(f"Tilt de sentimento calibrado no treino: {tilts}")

# --- Decisão ao vivo -------------------------------------------------------
with tab_live:
    decisao = _carrega_json_seguro("data/decisao_atual.json")

    if decisao is None:
        st.info("Nenhuma decisão ao vivo encontrada ainda. Use o botão **Puxar notícia AGORA** na sidebar, ou rode `python run_live.py --json data/decisao_atual.json`.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Última atualização", decisao["timestamp"][:19].replace("T", " "))
        modo = decisao["modo"]
        cor_modo = "🟢" if modo != "RISK-OFF" else "🔴"
        col2.metric("Modo do circuit breaker", f"{cor_modo} {modo}")
        col3.metric("Índice de estresse (Focus)", f"{decisao['stress_index']:.2f}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Sentimento por ativo agora**")
            sent = decisao["sentiment_scores"]
            for ticker, score in sent.items():
                cor = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
                st.markdown(f"{cor} **{ticker}**: {score:+.3f}")

        with col_b:
            st.markdown("**Alocação recomendada agora**")
            pesos = {k: v for k, v in decisao["pesos"].items() if v > 0.001}
            fig_pizza = go.Figure(go.Pie(
                labels=list(pesos.keys()),
                values=list(pesos.values()),
                hole=0.55,
                marker=dict(colors=PALETA, line=dict(color='#0a0a0a', width=2)),
                textfont=dict(color='#e5e5e5'),
            ))
            fig_pizza.update_layout(**PLOTLY_THEME, height=320, showlegend=True)
            st.plotly_chart(fig_pizza, use_container_width=True, key="pizza_live")

# --- Notícias ----------------------------------------------------------
with tab_news:
    st.subheader("Notícias mais recentes coletadas")
    noticias = _carrega_csv_seguro("data/raw/rss_news.csv")
    if noticias is None or noticias.empty:
        st.info("Nenhuma notícia coletada ainda. Use o botão **Notícias agora (RSS)** na sidebar.")
    else:
        noticias_recentes = noticias.sort_values("publicado_em", ascending=False).head(15)
        for _, row in noticias_recentes.iterrows():
            with st.container():
                st.markdown(f"**{row['titulo']}**")
                st.caption(f"{row['fonte']} — {row['publicado_em']}")
                st.markdown("&nbsp;", unsafe_allow_html=True)

# --- Sentiment ---------------------------------------------------------
with tab_sent:
    st.subheader("Sentiment Alpha Score — histórico")
    scores = _carrega_csv_seguro("data/processed/sentiment_scores.csv", index_col=0, parse_dates=True)
    if scores is None or scores.empty:
        st.info("Nenhum histórico de sentiment ainda. Use o botão **Escorar sentiment** na sidebar.")
    else:
        fig_sent = go.Figure()
        for i, col in enumerate(scores.columns):
            fig_sent.add_trace(go.Scatter(
                x=scores.index, y=scores[col], name=col, mode="lines",
                line=dict(color=PALETA[i % len(PALETA)], width=2),
            ))
        fig_sent.update_layout(**PLOTLY_THEME, height=420, yaxis_title="Sentiment Score")
        st.plotly_chart(fig_sent, use_container_width=True, key="sent_hist")

# --- Backtest ------------------------------------------------------------
with tab_bt:
    st.subheader("Backtest — Momentum Punch vs Benchmark")
    curvas = _carrega_csv_seguro("data/processed/equity_curves.csv", index_col=0, parse_dates=True)
    if curvas is None or curvas.empty:
        st.info("Nenhum backtest rodado ainda. Use os botões **Rodar backtest** na sidebar.")
    else:
        # "Benchmark" é alias do benchmark escolhido na execução e aparece
        # nomeado logo abaixo — plotar os dois desenharia a mesma linha 2x
        colunas_plot = [c for c in curvas.columns if c != "Benchmark"]
        fig_bt = go.Figure()
        for i, col in enumerate(colunas_plot):
            destaque = col == "Momentum Punch"
            fig_bt.add_trace(go.Scatter(
                x=curvas.index, y=(curvas[col] - 1) * 100, name=col, mode="lines",
                line=dict(color=PALETA[i % len(PALETA)], width=3 if destaque else 1.5,
                          dash=None if destaque else "dot"),
            ))
        fig_bt.update_layout(**PLOTLY_THEME, height=440, yaxis_title="Retorno acumulado (%)")
        st.plotly_chart(fig_bt, use_container_width=True, key="bt_curve")

        retorno_final = (curvas.iloc[-1] / curvas.iloc[0] - 1) * 100
        col1, col2 = st.columns(2)
        col1.metric("Momentum Punch — retorno total", f"{retorno_final['Momentum Punch']:.1f}%")
        col2.metric("Benchmark — retorno total", f"{retorno_final['Benchmark']:.1f}%")

# --- Ablação ---------------------------------------------------------------
with tab_abl:
    st.subheader("Matriz de ablação")
    st.caption("Contribuição marginal de cada módulo do sistema.")
    ablacao = _carrega_csv_seguro("data/processed/ablation_results.csv")
    if ablacao is None or ablacao.empty:
        st.info("Nenhuma ablação rodada ainda. Use o botão **Rodar ablação** na sidebar.")
    else:
        st.dataframe(ablacao, use_container_width=True, hide_index=True)
        st.caption(
            "Cada par isola um eixo: A1/A2 (EMA) · A2/A4 (circuit breaker) · A5/A6 (gate de relevância) · "
            "A7/A8 (tilt multiplicativo vs aditivo) · A9/A10 (custos) · A11/A12 (covariância). "
            "A comparação que testa a TESE é A4 vs A3: se o completo não supera o circuit breaker sozinho, "
            "o ganho não veio do texto."
        )