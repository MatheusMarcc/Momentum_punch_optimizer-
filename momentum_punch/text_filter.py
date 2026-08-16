"""
Filtro de relevância por ticker: decide QUAIS textos do dia vão pro scoring de
cada ativo. É a primeira camada da "relevância por ativo" que o pré-relatório
exige (box RISCO METODOLÓGICO da seção 4) — a segunda é a relevância que o
próprio LLM devolve, usada como gate no score.

Por que isso virou um módulo em vez de continuar como função solta dentro do
build_sentiment_dataset.py: o filtro anterior fazia `kw in texto.lower()`,
substring pura, e isso estava mandando o corpus errado pra metade dos ativos.
Dois casos reais, medidos no corpus da CVM:

  * ISUS11 (tese ESG) recebia 947 textos, o maior balde do corpus, porque a
    keyword "emiss" — escrita pensando em *emissões de carbono* — casava com
    "Emissão de Debêntures", "Emissão de Ações", "Escritura de Emissão". O
    ativo de sustentabilidade estava sendo escorado com captação financeira.

  * BOVA11 (tese macro) recebia "Posse Membros Conselho Fiscal" e "Renúncia
    membro do Conselho Fiscal", porque a keyword "fiscal" — escrita pensando em
    *política fiscal* — casa com *conselho fiscal*, que é órgão societário.

Nenhum modelo, por melhor que seja, extrai sinal macro de uma ata de conselho
fiscal. O erro não era do classificador, era do que chegava nele.

As três defesas aqui:
  1. NORMALIZAÇÃO: acento e caixa saem dos dois lados, então "Governança",
     "governanca" e "GOVERNANÇA" são a mesma coisa e a lista fica legível.
  2. FRONTEIRA DE PALAVRA: o termo precisa começar em início de palavra. Sem
     isso "b3" casa dentro de qualquer código alfanumérico e "solar" casaria
     em "consolar".
  3. EXCLUSÕES: termos que, se presentes, descartam o texto pro ticket mesmo
     que algum positivo tenha casado. É o que separa "política fiscal" de
     "conselho fiscal" — não existe forma de fazer isso só com lista positiva.

Casa por PREFIXO de palavra de propósito ("eolic" pega eólica/eólico/eólicas),
que é o que resolve gênero e plural do português sem precisar listar variação.
Por isso o singular "emissao" NÃO está na lista do ISUS11 e o plural
"emissoes" está: em português, emissão de valor mobiliário é quase sempre
singular e emissão de carbono quase sempre plural. A exclusão explícita cobre
o resto.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Termos POSITIVOS por ticker (sem acento — a normalização tira dos dois lados)
# ---------------------------------------------------------------------------

KEYWORDS_POSITIVAS: dict[str, list[str]] = {
    # Índice de Sustentabilidade Empresarial — clima, ESG, ambiental
    "ISUS11": [
        "sustentabilidade", "sustentavel", "esg", "emissoes", "efeito estufa",
        "gases de efeito estufa", "carbono", "descarboniza", "mudanca climatica",
        "mudancas climaticas", "climatic", "ambiental", "poluic", "reciclag",
        "energia limpa", "indice de sustentabilidade",
    ],
    # Governança corporativa — órgãos societários, conflito de acionistas
    "GOVE11": [
        "governanca", "conselho de administracao", "conselho fiscal", "diretoria",
        "acionist", "assembleia", "estatuto social", "escandalo", "fraude",
        "corrupc", "acordo de acionistas", "poison pill", "tag along",
        "oferta publica de aquisicao", "conflito de interesse", "auditoria",
    ],
    # Receitas verdes / transição energética
    "REVE11": [
        "energia renovavel", "energia limpa", "eolic", "solar", "fotovoltaic",
        "transicao energetica", "hidrogenio verde", "biocombustivel", "etanol",
        "veiculo eletrico", "carro eletrico", "mobilidade eletrica",
        "parque eolico", "energia sustentavel",
    ],
    # Beta amplo / macro Brasil
    "BOVA11": [
        "ibovespa", "bolsa", "b3", "selic", "juros", "pib", "copom", "inflac",
        "ipca", "deficit", "superavit", "dolar", "cambio", "recessao",
        "petroleo", "brent", "geopolitic", "politica fiscal", "situacao fiscal",
        "arcabouco fiscal", "meta fiscal", "resultado fiscal", "ajuste fiscal",
        "risco fiscal", "atividade economica",
    ],
}

# ---------------------------------------------------------------------------
# Termos de EXCLUSÃO: se casarem, o texto é descartado pro ticker mesmo que
# algum positivo tenha casado. Cada linha aqui existe por um falso positivo
# observado no corpus, não por precaução teórica.
# ---------------------------------------------------------------------------

KEYWORDS_EXCLUSAO: dict[str, list[str]] = {
    # captação/valor mobiliário sendo lida como emissão de carbono
    "ISUS11": [
        "emissao de debentures", "emissao de acoes", "emissao de notas",
        "emissao de cri", "emissao de cra", "escritura de emissao",
        "emissao privada", "nova emissao", "oferta restrita", "resgate",
        "debentur", "commercial paper",
    ],
    "GOVE11": [],
    "REVE11": [],
    # "fiscal" societário/tributário sendo lido como política fiscal
    "BOVA11": [
        "conselho fiscal", "nota fiscal", "incentivo fiscal", "beneficio fiscal",
        "credito fiscal", "prejuizo fiscal", "regime fiscal especial",
    ],
}


def normalizar(texto: str) -> str:
    """Minúscula e sem acento — 'Governança' e 'governanca' viram a mesma coisa."""
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.lower()


def _compilar(termos: list[str]) -> re.Pattern | None:
    """Casa em INÍCIO de palavra, sem exigir fim — pega gênero e plural
    ('eolic' -> eólica/eólico/eólicas) sem listar cada variação."""
    if not termos:
        return None
    alternativas = "|".join(re.escape(normalizar(t)) for t in termos)
    return re.compile(rf"\b(?:{alternativas})", re.IGNORECASE)


_POSITIVOS = {t: _compilar(kws) for t, kws in KEYWORDS_POSITIVAS.items()}
_EXCLUSOES = {t: _compilar(kws) for t, kws in KEYWORDS_EXCLUSAO.items()}


def texto_e_relevante(texto: str, ticker: str) -> bool:
    """True se o texto deve ser escorado para esse ticker."""
    positivo = _POSITIVOS.get(ticker)
    if positivo is None:  # ticker sem lista configurada: não filtra
        return True

    normalizado = normalizar(texto)
    if not positivo.search(normalizado):
        return False

    exclusao = _EXCLUSOES.get(ticker)
    if exclusao is not None and exclusao.search(normalizado):
        return False

    return True


def filtrar_por_ticker(textos: list[str], ticker: str) -> list[str]:
    """Mantém só os textos relevantes pro ticker. Lista vazia é resposta
    legítima (dia sem notícia do tema) — quem chama trata como score neutro,
    não inventa sentimento."""
    return [t for t in textos if texto_e_relevante(t, ticker)]
