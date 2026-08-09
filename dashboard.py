"""
Dashboard do Momentum Punch — implementação real do "Dashboard Interativo"
descrito no deck original, AGORA com painel de execução: botões que rodam
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

st.set_page_config(page_title="Momentum Punch — Dashboard", layout="wide", page_icon="🥊")

st.title("🥊 Momentum Punch — Sentiment Alpha Portfolio")
st.caption("Desafio Itaú Asset Quant AI 2026 — Equipe MMA")


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
            st.code(resultado.stdout[-4000:], language="text")  # últimas ~4000 chars, log pode ser longo
        if resultado.stderr:
            st.code(resultado.stderr[-4000:], language="text")
    if sucesso:
        st.success(f"{label}: concluído.")
    else:
        st.error(f"{label}: falhou (código {resultado.returncode}).")
    return sucesso


# ---------------------------------------------------------------------------
# Painel de controle — dispensa terminal: cada botão roda o script de verdade

st.subheader("🎛️ Painel de execução")
st.caption("Cada botão roda o script real via subprocess e recarrega os dados abaixo automaticamente.")

with st.expander("1️⃣ Dados de mercado (preço, CDI, estresse macro)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Preço dos ETFs\n(yfinance)"):
        _roda_comando(["fetch_prices_yfinance.py"], "Preço dos ETFs")
        st.rerun()
    if c2.button("CDI/Selic/IPCA\n(Bacen SGS)"):
        _roda_comando(["collect_data.py", "--only", "bacen_sgs"], "Bacen SGS")
        st.rerun()
    if c3.button("Expectativas macro\n(Bacen Focus)"):
        _roda_comando(["collect_data.py", "--only", "bacen_focus"], "Bacen Focus")
        st.rerun()
    if c4.button("Construir\nÍndice de Estresse"):
        _roda_comando(["build_stress_index_focus.py"], "Índice de Estresse (Focus)")
        st.rerun()

with st.expander("2️⃣ Notícia e sentimento", expanded=False):
    c1, c2, c3 = st.columns(3)
    if c1.button("Notícias agora\n(RSS)"):
        _roda_comando(["collect_data.py", "--only", "rss_news"], "Coleta RSS")
        st.rerun()
    if c2.button("Histórico CVM\n(2023-2026)"):
        _roda_comando(["pull_cvm_historical.py", "--anos", "2023", "2024", "2025", "2026"], "Histórico CVM")
        st.rerun()
    if c3.button("Escorar sentiment\n(FinBERT, via CVM)"):
        _roda_comando(["build_sentiment_dataset.py", "--source", "cvm", "--force"], "Sentiment (FinBERT)")
        st.rerun()

with st.expander("3️⃣ Backtest e ablação", expanded=False):
    c1, c2, c3 = st.columns(3)
    if c1.button("Rodar backtest\n(tilt linear)"):
        _roda_comando(["run_real_backtest.py"], "Backtest")
        st.rerun()
    if c2.button("Rodar backtest\n(Black-Litterman)"):
        _roda_comando(["run_real_backtest.py", "--mu-method", "black_litterman"], "Backtest (Black-Litterman)")
        st.rerun()
    if c3.button("Rodar ablação\n(A0/A2/A3/A4, 10bps)"):
        _roda_comando(["run_ablations.py", "--cost-bps", "10"], "Matriz de ablação")
        st.rerun()

with st.expander("4️⃣ Decisão ao vivo", expanded=True):
    if st.button("🔴 Puxar notícia AGORA e recalcular alocação", type="primary"):
        _roda_comando(["run_live.py", "--json", "data/decisao_atual.json"], "Decisão ao vivo")
        st.rerun()

st.divider()


# ---------------------------------------------------------------------------
# Linha 1: decisão AO VIVO (run_live.py --json)

decisao = _carrega_json_seguro("data/decisao_atual.json")

st.subheader("📡 Decisão ao vivo")
if decisao is None:
    st.info("Nenhuma decisão ao vivo encontrada ainda. Rode: `python run_live.py --json data/decisao_atual.json`")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Última atualização", decisao["timestamp"][:19].replace("T", " "))
    modo = decisao["modo"]
    cor_modo = "🔴" if modo == "RISK-OFF" else "🟢"
    col2.metric("Modo do circuit breaker", f"{cor_modo} {modo}")
    col3.metric("Índice de estresse (Focus)", f"{decisao['stress_index']:.2f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Sentimento por ativo agora**")
        sent = decisao["sentiment_scores"]
        for ticker, score in sent.items():
            cor = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
            st.write(f"{cor} **{ticker}**: {score:+.3f}")

    with col_b:
        st.markdown("**Alocação recomendada agora**")
        pesos = {k: v for k, v in decisao["pesos"].items() if v > 0.001}
        fig_pizza = px.pie(names=list(pesos.keys()), values=list(pesos.values()), hole=0.4)
        fig_pizza.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_pizza, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Linha 2: notícias mais recentes

st.subheader("📰 Notícias mais recentes coletadas")
noticias = _carrega_csv_seguro("data/raw/rss_news.csv")
if noticias is None or noticias.empty:
    st.info("Nenhuma notícia coletada ainda. Rode: `python collect_data.py --only rss_news`")
else:
    noticias_recentes = noticias.sort_values("publicado_em", ascending=False).head(15)
    for _, row in noticias_recentes.iterrows():
        with st.container():
            st.markdown(f"**{row['titulo']}**")
            st.caption(f"{row['fonte']} — {row['publicado_em']}")

st.divider()


# ---------------------------------------------------------------------------
# Linha 3: sentiment histórico por ticker

st.subheader("📈 Sentiment Alpha Score — histórico")
scores = _carrega_csv_seguro("data/processed/sentiment_scores.csv", index_col=0, parse_dates=True)
if scores is None or scores.empty:
    st.info("Nenhum histórico de sentiment ainda. Rode: `python build_sentiment_dataset.py`")
else:
    fig_sent = go.Figure()
    for col in scores.columns:
        fig_sent.add_trace(go.Scatter(x=scores.index, y=scores[col], name=col, mode="lines"))
    fig_sent.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Sentiment Score")
    st.plotly_chart(fig_sent, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Linha 4: backtest — equity curve

st.subheader("💰 Backtest — Momentum Punch vs Benchmark")
curvas = _carrega_csv_seguro("data/processed/equity_curves.csv", index_col=0, parse_dates=True)
if curvas is None or curvas.empty:
    st.info("Nenhum backtest rodado ainda. Rode: `python run_real_backtest.py`")
else:
    fig_bt = go.Figure()
    for col in curvas.columns:
        fig_bt.add_trace(go.Scatter(x=curvas.index, y=(curvas[col] - 1) * 100, name=col, mode="lines"))
    fig_bt.update_layout(height=400, margin=dict(t=20, b=20), yaxis_title="Retorno acumulado (%)")
    st.plotly_chart(fig_bt, use_container_width=True)

    retorno_final = (curvas.iloc[-1] / curvas.iloc[0] - 1) * 100
    col1, col2 = st.columns(2)
    col1.metric("Momentum Punch — retorno total", f"{retorno_final['Momentum Punch']:.1f}%")
    col2.metric("Benchmark — retorno total", f"{retorno_final['Benchmark']:.1f}%")

st.divider()


# ---------------------------------------------------------------------------
# Linha 5: matriz de ablação

st.subheader("🔬 Matriz de ablação (contribuição marginal de cada módulo)")
ablacao = _carrega_csv_seguro("data/processed/ablation_results.csv")
if ablacao is None or ablacao.empty:
    st.info("Nenhuma ablação rodada ainda. Rode: `python run_ablations.py`")
else:
    st.dataframe(ablacao, use_container_width=True, hide_index=True)
    st.caption(
        "A0 = baseline puro | A2 = só sentimento | A3 = só circuit breaker | A4 = sistema completo. "
        "Compare A2 vs A0 (efeito isolado do sentimento) e A4 vs A3 (sentimento ajuda por cima do circuit breaker?)."
    )

st.divider()
st.caption("Documento/dashboard de pesquisa. Não constitui recomendação de investimento.")
