"""
Mede o filtro e o classificador contra o conjunto rotulado à mão — a
"validação do classificador" prometida na seção 4 do pré-relatório.

Responde três perguntas que até aqui eram suposição:

  1. O FILTRO acerta? Precisão e recall de text_filter contra o julgamento
     humano de relevância. Recall só é estimável porque o conjunto inclui
     textos que o filtro rejeitou.

  2. A RELEVÂNCIA da LLM acerta? A relevância é usada como gate multiplicativo
     (score = sentimento x relevancia), então relevância errada zera texto bom
     ou deixa passar texto irrelevante — em ambos os casos o erro entra direto
     na carteira.

  3. O SENTIMENTO da LLM acerta? Matriz de confusão 3x3 (neg/neutro/pos) só
     sobre os textos que o humano considerou relevantes — medir sentimento em
     texto irrelevante não significa nada.

Reporta também a taxa de ABSTENÇÃO (fallback), que a seção 4 exige: chamada que
não devolveu JSON válido depois das tentativas vira neutro com confiança 0, e
precisa aparecer como abstenção e não se misturar com "o modelo achou neutro".

Uso:
    python avaliar_classificador.py
    python avaliar_classificador.py --limite 50    # amostra menor, pra testar rápido
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from momentum_punch import manifesto, sentiment, text_filter

LIMIAR_RELEVANCIA = 0.5  # acima disso, considera-se que a LLM julgou relevante


def _classe_sentimento(valor: float, zona_neutra: float = 0.15) -> int:
    """Converte score contínuo em -1/0/+1. A zona neutra existe porque o humano
    rotula em 3 classes e a LLM devolve contínuo: sem ela, um +0,02 viraria
    'positivo' e a matriz mediria arredondamento, não discordância."""
    if valor > zona_neutra:
        return 1
    if valor < -zona_neutra:
        return -1
    return 0


def _matriz_confusao(verdadeiro: list[int], previsto: list[int], rotulos=(-1, 0, 1)) -> pd.DataFrame:
    m = pd.DataFrame(0, index=[f"humano {r:+d}" for r in rotulos],
                     columns=[f"LLM {r:+d}" for r in rotulos])
    for v, p in zip(verdadeiro, previsto):
        m.loc[f"humano {v:+d}", f"LLM {p:+d}"] += 1
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conjunto", default="data/processed/conjunto_rotulagem.csv")
    parser.add_argument("--limite", type=int, default=0, help="0 = usa tudo")
    parser.add_argument("--out", default="data/processed/validacao_classificador.csv")
    args = parser.parse_args()

    if not os.path.exists(args.conjunto):
        raise SystemExit(f"{args.conjunto} não existe. Rode antes: python montar_conjunto_rotulagem.py")

    df = pd.read_csv(args.conjunto)
    rotulados = df[df["relevante_humano"].notna() & (df["relevante_humano"].astype(str).str.strip() != "")]
    if rotulados.empty:
        raise SystemExit(f"{args.conjunto} ainda não tem rótulo humano preenchido — veja o arquivo _INSTRUCOES.txt")

    if args.limite:
        rotulados = rotulados.head(args.limite)
    print(f"[validacao] {len(rotulados)}/{len(df)} linhas rotuladas\n")

    # ---- 1. Filtro ---------------------------------------------------------
    rel_humano = rotulados["relevante_humano"].astype(int)
    aceito = rotulados["filtro_aceitou"].astype(int)
    vp = int(((aceito == 1) & (rel_humano == 1)).sum())
    fp = int(((aceito == 1) & (rel_humano == 0)).sum())
    fn = int(((aceito == 0) & (rel_humano == 1)).sum())
    vn = int(((aceito == 0) & (rel_humano == 0)).sum())
    precisao = vp / (vp + fp) if vp + fp else float("nan")
    recall = vp / (vp + fn) if vp + fn else float("nan")

    print("=== 1. Filtro de relevância (text_filter) ===")
    print(f"  precisão: {precisao:.1%}  (dos aceitos, quantos o humano considerou relevantes)")
    print(f"  recall:   {recall:.1%}  (dos relevantes, quantos o filtro deixou passar)")
    print(f"  VP={vp} FP={fp} FN={fn} VN={vn}\n")

    # ---- 2 e 3. Classificador ---------------------------------------------
    print("=== Escorando o conjunto no motor GenAI (1 chamada por texto) ===")
    saidas = []
    for i, linha in enumerate(rotulados.itertuples(), 1):
        r = sentiment.score_texts_structured(linha.ticker, [linha.texto])
        saidas.append({
            "id": linha.id, "ticker": linha.ticker, "texto": linha.texto,
            "filtro_aceitou": linha.filtro_aceitou,
            "relevante_humano": int(linha.relevante_humano),
            "sentimento_humano": linha.sentimento_humano,
            "llm_sentimento": r.sentimento, "llm_relevancia": r.relevancia,
            "llm_confianca": r.confianca, "llm_fallback": r.is_fallback,
            "llm_justificativa": r.justificativa,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(rotulados)}")
    res = pd.DataFrame(saidas)

    taxa_fallback = res["llm_fallback"].mean()
    print(f"\n=== Abstenção ===")
    print(f"  fallback (JSON inválido após retries): {taxa_fallback:.1%} das chamadas")

    print(f"\n=== 2. Relevância da LLM (limiar {LIMIAR_RELEVANCIA}) ===")
    llm_rel = (res["llm_relevancia"] >= LIMIAR_RELEVANCIA).astype(int)
    concordancia = (llm_rel == res["relevante_humano"]).mean()
    vp2 = int(((llm_rel == 1) & (res["relevante_humano"] == 1)).sum())
    fp2 = int(((llm_rel == 1) & (res["relevante_humano"] == 0)).sum())
    fn2 = int(((llm_rel == 0) & (res["relevante_humano"] == 1)).sum())
    print(f"  concordância com o humano: {concordancia:.1%}")
    print(f"  precisão: {vp2 / (vp2 + fp2):.1%}" if vp2 + fp2 else "  precisão: n/a")
    print(f"  recall:   {vp2 / (vp2 + fn2):.1%}" if vp2 + fn2 else "  recall: n/a")
    print(f"  textos relevantes ZERADOS pelo gate: {fn2} — cada um vira score 0 na carteira")

    print(f"\n=== 3. Sentimento da LLM (só nos textos relevantes segundo o humano) ===")
    rel = res[(res["relevante_humano"] == 1) & res["sentimento_humano"].notna()
              & (res["sentimento_humano"].astype(str).str.strip() != "")]
    if rel.empty:
        print("  nenhum texto relevante com sentimento rotulado — pule ou rotule mais")
    else:
        humano = [int(float(v)) for v in rel["sentimento_humano"]]
        llm = [_classe_sentimento(v) for v in rel["llm_sentimento"]]
        matriz = _matriz_confusao(humano, llm)
        acuracia = sum(h == l for h, l in zip(humano, llm)) / len(humano)
        invertidos = sum(1 for h, l in zip(humano, llm) if h * l < 0)
        print(f"  acurácia: {acuracia:.1%} (n={len(humano)})")
        print(f"  INVERSÕES de sinal (humano + / LLM -, ou vice-versa): {invertidos} "
              f"({invertidos / len(humano):.1%}) — são as que mais custam, "
              f"porque empurram a carteira na direção errada\n")
        print(matriz.to_string())

    res.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nSalvo em {args.out}")

    manifesto.salvar(
        manifesto.gerar(
            experimento="validacao_classificador",
            categoria="VERIFICADO",
            entradas={"conjunto_rotulado": args.conjunto},
            parametros_execucao={
                "n_rotulados": len(rotulados), "limiar_relevancia": LIMIAR_RELEVANCIA,
                "zona_neutra": 0.15,
            },
            metricas={
                "filtro_precisao": round(precisao, 4), "filtro_recall": round(recall, 4),
                "llm_relevancia_concordancia": round(float(concordancia), 4),
                "taxa_fallback": round(float(taxa_fallback), 4),
            },
            saidas=[args.out],
        ),
        "data/processed/manifesto_validacao_classificador.json",
    )


if __name__ == "__main__":
    main()
