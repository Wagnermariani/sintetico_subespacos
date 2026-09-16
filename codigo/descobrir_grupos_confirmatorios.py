import os

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

DIRETORIO_CONFIRM = os.path.join(
    PROJETO,
    "dados",
    "confirmacao"
)

ARQUIVO_MANIFESTO = os.path.join(
    DIRETORIO_CONFIRM,
    "manifesto_realizacoes.csv"
)

ARQUIVO_MAPA_VERDADEIRO = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "grupos_confirmatorios"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# DESCOBERTA DOS GRUPOS DE UMA REALIZAÇÃO
# ============================================================================

def descobrir_grupos(
    realizacao,
    arquivo_treino
):

    print()
    print("=" * 84)
    print(f"{realizacao} — DESCOBERTA USANDO SOMENTE TREINO NORMAL")
    print("=" * 84)

    treino = pd.read_csv(
        arquivo_treino
    )

    features = [
        coluna
        for coluna in treino.columns
        if coluna not in [
            "timestamp",
            "Attack",
            "Tipo",
        ]
    ]

    if len(features) != 400:
        raise RuntimeError(
            f"{realizacao}: esperadas 400 features."
        )

    print(
        f"Features: {len(features)}"
    )

    # ========================================================================
    # CORRELAÇÃO — SEM GABARITO
    # ========================================================================

    print(
        "Calculando matriz de correlação..."
    )

    corr = (
        treino[
            features
        ]
        .corr(
            method="pearson"
        )
        .abs()
    )

    valores = []

    pares = []

    for i in range(
        len(features)
    ):

        for j in range(
            i + 1,
            len(features)
        ):

            valor = float(
                corr.iloc[
                    i,
                    j
                ]
            )

            valores.append(
                valor
            )

            pares.append(
                (
                    features[i],
                    features[j],
                    valor,
                )
            )

    valores = np.asarray(
        valores,
        dtype=np.float64
    )

    print(
        f"Pares: {len(valores)}"
    )

    # ========================================================================
    # KMEANS — DUAS POPULAÇÕES DE CORRELAÇÃO
    #
    # Ainda não existe qualquer consulta ao mapa verdadeiro.
    # ========================================================================

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(
        valores.reshape(
            -1,
            1
        )
    )

    centros = (
        kmeans
        .cluster_centers_
        .ravel()
    )

    classe_fraca = int(
        np.argmin(
            centros
        )
    )

    classe_forte = int(
        np.argmax(
            centros
        )
    )

    valores_fracos = valores[
        labels
        ==
        classe_fraca
    ]

    valores_fortes = valores[
        labels
        ==
        classe_forte
    ]

    max_fraco = float(
        valores_fracos.max()
    )

    min_forte = float(
        valores_fortes.min()
    )

    corte = (
        max_fraco
        +
        min_forte
    ) / 2.0

    print(
        f"Centro fraco: {centros[classe_fraca]:.6f}"
    )

    print(
        f"Centro forte: {centros[classe_forte]:.6f}"
    )

    print(
        f"Máximo fraco: {max_fraco:.6f}"
    )

    print(
        f"Mínimo forte: {min_forte:.6f}"
    )

    print(
        f"Corte descoberto: {corte:.6f}"
    )

    print(
        f"Gap entre classes: "
        f"{min_forte - max_fraco:.6f}"
    )

    # ========================================================================
    # GRAFO
    # ========================================================================

    grafo = nx.Graph()

    grafo.add_nodes_from(
        features
    )

    n_arestas = 0

    for feature_a, feature_b, valor in pares:

        if valor > corte:

            grafo.add_edge(
                feature_a,
                feature_b,
                peso=valor
            )

            n_arestas += 1

    componentes = list(
        nx.connected_components(
            grafo
        )
    )

    componentes = sorted(
        componentes,
        key=lambda componente: (
            -len(
                componente
            ),
            sorted(
                componente
            )[0]
        )
    )

    print(
        f"Arestas fortes: {n_arestas}"
    )

    print(
        f"Componentes encontrados: "
        f"{len(componentes)}"
    )

    tamanhos = [
        len(
            componente
        )
        for componente
        in componentes
    ]

    print(
        f"Tamanhos: {tamanhos}"
    )

    # ========================================================================
    # MAPA DESCOBERTO
    #
    # O agrupamento está encerrado aqui.
    # ========================================================================

    registros = []

    for indice, componente in enumerate(
        componentes,
        start=1
    ):

        grupo_descoberto = (
            f"G{indice:02d}"
        )

        for feature in sorted(
            componente
        ):

            registros.append(
                {
                    "Feature":
                        feature,

                    "Grupo_Descoberto":
                        grupo_descoberto,
                }
            )

    mapa_descoberto = pd.DataFrame(
        registros
    )

    diretorio_realizacao = os.path.join(
        DIRETORIO_SAIDA,
        realizacao
    )

    os.makedirs(
        diretorio_realizacao,
        exist_ok=True
    )

    arquivo_mapa_descoberto = os.path.join(
        diretorio_realizacao,
        "mapa_grupos_descobertos.csv"
    )

    mapa_descoberto.to_csv(
        arquivo_mapa_descoberto,
        index=False
    )

    # ========================================================================
    # SOMENTE AGORA: AVALIAÇÃO COM O GABARITO
    # ========================================================================

    mapa_verdadeiro = pd.read_csv(
        ARQUIVO_MAPA_VERDADEIRO
    )

    avaliacao = (
        mapa_verdadeiro[
            [
                "Feature",
                "Grupo_Verdadeiro",
                "Tipo",
            ]
        ]
        .merge(
            mapa_descoberto,
            on="Feature",
            how="left"
        )
    )

    if (
        avaliacao[
            "Grupo_Descoberto"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            f"{realizacao}: feature sem grupo descoberto."
        )

    labels_true = (
        avaliacao[
            "Grupo_Verdadeiro"
        ]
        .astype(str)
        .to_numpy()
    )

    labels_pred = (
        avaliacao[
            "Grupo_Descoberto"
        ]
        .astype(str)
        .to_numpy()
    )

    ari = adjusted_rand_score(
        labels_true,
        labels_pred
    )

    nmi = normalized_mutual_info_score(
        labels_true,
        labels_pred
    )

    # ------------------------------------------------------------------------
    # PUREZA
    # ------------------------------------------------------------------------

    purezas = []

    for _, parte in avaliacao.groupby(
        "Grupo_Descoberto"
    ):

        contagens = (
            parte[
                "Grupo_Verdadeiro"
            ]
            .value_counts()
        )

        pureza = (
            contagens.iloc[0]
            /
            len(parte)
        )

        purezas.append(
            pureza
        )

    pureza_minima = float(
        min(
            purezas
        )
    )

    print()
    print(
        f"ARI: {ari:.6f}"
    )

    print(
        f"NMI: {nmi:.6f}"
    )

    print(
        f"Pureza mínima: {pureza_minima:.6f}"
    )

    arquivo_avaliacao = os.path.join(
        diretorio_realizacao,
        "avaliacao_grupos.csv"
    )

    avaliacao.to_csv(
        arquivo_avaliacao,
        index=False
    )

    return {
        "Realizacao":
            realizacao,

        "Centro_Fraco":
            float(
                centros[
                    classe_fraca
                ]
            ),

        "Centro_Forte":
            float(
                centros[
                    classe_forte
                ]
            ),

        "Max_Fraco":
            max_fraco,

        "Min_Forte":
            min_forte,

        "Gap":
            (
                min_forte
                -
                max_fraco
            ),

        "Corte":
            corte,

        "N_Arestas":
            n_arestas,

        "N_Componentes":
            len(
                componentes
            ),

        "Min_Tamanho_Grupo":
            min(
                tamanhos
            ),

        "Max_Tamanho_Grupo":
            max(
                tamanhos
            ),

        "ARI":
            ari,

        "NMI":
            nmi,

        "Pureza_Minima":
            pureza_minima,

        "Arquivo_Mapa_Descoberto":
            arquivo_mapa_descoberto,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 84)
    print("DESCOBERTA DOS GRUPOS — REALIZAÇÕES CONFIRMATÓRIAS")
    print("=" * 84)

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    if len(manifesto) != 5:

        raise RuntimeError(
            "Esperadas exatamente 5 realizações."
        )

    resultados = []

    for _, linha in manifesto.iterrows():

        resultado = descobrir_grupos(
            realizacao=str(
                linha[
                    "Realizacao"
                ]
            ),

            arquivo_treino=str(
                linha[
                    "Arquivo_Treino"
                ]
            )
        )

        resultados.append(
            resultado
        )

    resumo = pd.DataFrame(
        resultados
    )

    arquivo_resumo = os.path.join(
        DIRETORIO_SAIDA,
        "resumo_grupos_confirmatorios.csv"
    )

    resumo.to_csv(
        arquivo_resumo,
        index=False
    )

    print()
    print("=" * 84)
    print("RESUMO FINAL")
    print("=" * 84)

    print()

    print(
        resumo[
            [
                "Realizacao",
                "Gap",
                "N_Componentes",
                "Min_Tamanho_Grupo",
                "Max_Tamanho_Grupo",
                "ARI",
                "NMI",
                "Pureza_Minima",
            ]
        ]
        .to_string(
            index=False
        )
    )

    todas_20 = bool(
        (
            resumo[
                "N_Componentes"
            ]
            ==
            20
        ).all()
    )

    todos_20_features = bool(
        (
            resumo[
                "Min_Tamanho_Grupo"
            ]
            ==
            20
        ).all()
        and
        (
            resumo[
                "Max_Tamanho_Grupo"
            ]
            ==
            20
        ).all()
    )

    ari_perfeito = bool(
        np.allclose(
            resumo[
                "ARI"
            ],
            1.0
        )
    )

    nmi_perfeito = bool(
        np.allclose(
            resumo[
                "NMI"
            ],
            1.0
        )
    )

    passou = (
        todas_20
        and
        todos_20_features
        and
        ari_perfeito
        and
        nmi_perfeito
    )

    print()
    print(
        "RESULTADO: "
        f"{'APROVADO' if passou else 'REVISAR'}"
    )

    print()
    print(
        "Resumo salvo em:"
    )

    print(
        arquivo_resumo
    )


if __name__ == "__main__":
    main()