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

DIRETORIO_PROJETO = (
    "/home/wagner/areas/sintetico_subespacos"
)

ARQUIVO_TREINO = os.path.join(
    DIRETORIO_PROJETO,
    "dados",
    "treino_normal_piloto.csv"
)

ARQUIVO_MAPA = os.path.join(
    DIRETORIO_PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv"
)

DIRETORIO_SAIDA = os.path.join(
    DIRETORIO_PROJETO,
    "resultado",
    "agrupamento_normal"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "============================================"
    )
    print(
        "DESCOBERTA DOS GRUPOS — SOMENTE NORMAL"
    )
    print(
        "============================================"
    )

    print(
        "\nCarregando treino normal..."
    )

    df = pd.read_csv(
        ARQUIVO_TREINO
    )

    features = [
        c
        for c in df.columns
        if c not in [
            "timestamp",
            "Attack",
            "Tipo",
        ]
    ]

    print(
        f"Features encontradas: {len(features)}"
    )

    if len(features) != 400:
        raise RuntimeError(
            "Esperadas 400 features."
        )

    # ========================================================================
    # MATRIZ DE CORRELAÇÃO
    #
    # ATÉ AQUI NÃO EXISTE QUALQUER INFORMAÇÃO
    # SOBRE OS GRUPOS VERDADEIROS.
    # ========================================================================

    print(
        "Calculando correlações..."
    )

    corr = (
        df[features]
        .corr(method="pearson")
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
                    valor
                )
            )

    valores = np.asarray(
        valores,
        dtype=np.float64
    )

    print(
        f"Pares analisados: {len(valores)}"
    )

    # ========================================================================
    # DESCOBERTA NÃO SUPERVISIONADA DO CORTE
    # ========================================================================

    print(
        "\nDescobrindo automaticamente "
        "correlações fracas e fortes..."
    )

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(
        valores.reshape(-1, 1)
    )

    centros = (
        kmeans.cluster_centers_
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
        "\nCentros encontrados pelo KMeans:"
    )

    print(
        f"Correlação fraca: "
        f"{centros[classe_fraca]:.6f}"
    )

    print(
        f"Correlação forte: "
        f"{centros[classe_forte]:.6f}"
    )

    print(
        f"Máximo classe fraca: "
        f"{max_fraco:.6f}"
    )

    print(
        f"Mínimo classe forte: "
        f"{min_forte:.6f}"
    )

    print(
        f"Corte descoberto: "
        f"{corte:.6f}"
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
        key=lambda x: (
            -len(x),
            sorted(x)[0]
        )
    )

    print(
        "\n============================================"
    )
    print(
        "GRUPOS DESCOBERTOS"
    )
    print(
        "============================================"
    )

    print(
        f"Arestas fortes: {n_arestas}"
    )

    print(
        f"Componentes encontrados: "
        f"{len(componentes)}"
    )

    for indice, componente in enumerate(
        componentes,
        start=1
    ):

        print(
            f"Grupo descoberto {indice:02d}: "
            f"{len(componente)} features"
        )

    # ========================================================================
    # CRIA MAPA DESCOBERTO
    #
    # SOMENTE AGORA TEMOS UM RESULTADO DE AGRUPAMENTO.
    # ========================================================================

    registros = []

    labels_preditos = {}

    for indice, componente in enumerate(
        componentes,
        start=1
    ):

        nome_grupo = (
            f"G{indice:02d}"
        )

        for feature in sorted(
            componente
        ):

            labels_preditos[
                feature
            ] = nome_grupo

            registros.append(
                {
                    "Feature":
                        feature,

                    "Grupo_Descoberto":
                        nome_grupo,
                }
            )

    mapa_descoberto = pd.DataFrame(
        registros
    )

    # ========================================================================
    # AVALIAÇÃO COM O GABARITO
    #
    # O MAPA VERDADEIRO É ABERTO APENAS DEPOIS
    # QUE TODO O AGRUPAMENTO JÁ FOI CONCLUÍDO.
    # ========================================================================

    print(
        "\n============================================"
    )
    print(
        "AVALIAÇÃO CONTRA O MAPA VERDADEIRO"
    )
    print(
        "============================================"
    )

    mapa_verdadeiro = pd.read_csv(
        ARQUIVO_MAPA
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
            "Alguma feature ficou sem grupo."
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

    print(
        f"ARI: {ari:.6f}"
    )

    print(
        f"NMI: {nmi:.6f}"
    )

    # ========================================================================
    # PUREZA DOS COMPONENTES
    # ========================================================================

    tabela_componentes = []

    for grupo_descoberto, parte in (
        avaliacao.groupby(
            "Grupo_Descoberto"
        )
    ):

        contagens = (
            parte[
                "Grupo_Verdadeiro"
            ]
            .value_counts()
        )

        dominante = (
            contagens.index[0]
        )

        n_dominante = int(
            contagens.iloc[0]
        )

        pureza = (
            n_dominante
            /
            len(parte)
        )

        tabela_componentes.append(
            {
                "Grupo_Descoberto":
                    grupo_descoberto,

                "N_Features":
                    len(parte),

                "Grupo_Verdadeiro_Dominante":
                    dominante,

                "Pureza":
                    pureza,
            }
        )

    tabela_componentes = pd.DataFrame(
        tabela_componentes
    )

    print(
        "\nPureza mínima dos grupos: "
        f"{tabela_componentes['Pureza'].min():.6f}"
    )

    # ========================================================================
    # SALVAMENTO
    # ========================================================================

    mapa_descoberto.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "mapa_grupos_descobertos.csv"
        ),
        index=False
    )

    avaliacao.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "avaliacao_grupos.csv"
        ),
        index=False
    )

    tabela_componentes.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_componentes.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [
            {
                "Centro_Fraco":
                    centros[
                        classe_fraca
                    ],

                "Centro_Forte":
                    centros[
                        classe_forte
                    ],

                "Max_Fraco":
                    max_fraco,

                "Min_Forte":
                    min_forte,

                "Corte":
                    corte,

                "N_Componentes":
                    len(
                        componentes
                    ),

                "ARI":
                    ari,

                "NMI":
                    nmi,

                "Pureza_Minima":
                    tabela_componentes[
                        "Pureza"
                    ].min(),
            }
        ]
    ).to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_agrupamento.csv"
        ),
        index=False
    )

    print(
        "\nResultados salvos em:"
    )

    print(
        DIRETORIO_SAIDA
    )


if __name__ == "__main__":
    main()