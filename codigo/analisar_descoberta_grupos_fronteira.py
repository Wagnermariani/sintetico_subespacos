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

ARQUIVO_MANIFESTO = os.path.join(
    PROJETO,
    "dados",
    "fronteira_estrutural",
    "manifesto_cenarios_estruturais.csv",
)

ARQUIVO_MAPA_VERDADEIRO = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "descoberta_grupos",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)


# ============================================================================
# DESCOBERTA DE UM CENÁRIO
# ============================================================================

def analisar_cenario(
    cenario,
    arquivo_treino,
    alpha,
    beta,
):

    print()
    print("=" * 104)
    print(
        f"{cenario} | alpha={alpha:.2f} | beta={beta:.2f}"
    )
    print("=" * 104)

    treino = pd.read_csv(
        arquivo_treino
    )

    # ------------------------------------------------------------------------
    # SOMENTE FEATURES OBSERVADAS
    #
    # Nenhum mapa verdadeiro é consultado nesta etapa.
    # ------------------------------------------------------------------------

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
            f"{cenario}: esperadas 400 features."
        )

    # ========================================================================
    # MATRIZ DE CORRELAÇÃO ABSOLUTA
    # ========================================================================

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
        dtype=np.float64,
    )

    # ========================================================================
    # KMEANS EM DUAS POPULAÇÕES DE CORRELAÇÃO
    #
    # Ainda sem usar o mapa verdadeiro.
    # ========================================================================

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20,
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

    gap_classes = (
        min_forte
        -
        max_fraco
    )

    # ========================================================================
    # GRAFO E COMPONENTES
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
                peso=valor,
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
            )[0],
        ),
    )

    tamanhos = [
        len(
            componente
        )
        for componente in componentes
    ]

    # ------------------------------------------------------------------------
    # MAPA DESCOBERTO
    #
    # A descoberta termina aqui.
    # ------------------------------------------------------------------------

    registros = []

    for indice, componente in enumerate(
        componentes,
        start=1,
    ):

        grupo_descoberto = (
            f"G{indice:03d}"
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

    diretorio_cenario = os.path.join(
        DIRETORIO_SAIDA,
        cenario,
    )

    os.makedirs(
        diretorio_cenario,
        exist_ok=True,
    )

    arquivo_mapa = os.path.join(
        diretorio_cenario,
        "mapa_grupos_descobertos.csv",
    )

    mapa_descoberto.to_csv(
        arquivo_mapa,
        index=False,
    )

    # ========================================================================
    # SOMENTE AGORA: GABARITO PARA AVALIAÇÃO
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
            how="left",
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
            f"{cenario}: feature sem grupo descoberto."
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

    ari = float(
        adjusted_rand_score(
            labels_true,
            labels_pred,
        )
    )

    nmi = float(
        normalized_mutual_info_score(
            labels_true,
            labels_pred,
        )
    )

    # ------------------------------------------------------------------------
    # PUREZA PONDERADA
    # ------------------------------------------------------------------------

    corretos_pureza = 0

    for _, parte in avaliacao.groupby(
        "Grupo_Descoberto"
    ):

        contagens = (
            parte[
                "Grupo_Verdadeiro"
            ]
            .value_counts()
        )

        corretos_pureza += int(
            contagens.iloc[0]
        )

    pureza = (
        corretos_pureza
        /
        len(
            avaliacao
        )
    )

    # ========================================================================
    # CORRELAÇÃO INTRA × INTER
    #
    # Aqui usamos o gabarito somente para caracterizar o cenário,
    # nunca para construir os grupos.
    # ========================================================================

    grupo_verdadeiro_por_feature = (
        mapa_verdadeiro
        .set_index(
            "Feature"
        )[
            "Grupo_Verdadeiro"
        ]
        .to_dict()
    )

    intra = []

    inter = []

    for feature_a, feature_b, valor in pares:

        if (
            grupo_verdadeiro_por_feature[
                feature_a
            ]
            ==
            grupo_verdadeiro_por_feature[
                feature_b
            ]
        ):

            intra.append(
                valor
            )

        else:

            inter.append(
                valor
            )

    intra = np.asarray(
        intra,
        dtype=np.float64,
    )

    inter = np.asarray(
        inter,
        dtype=np.float64,
    )

    contraste_mediana = (
        float(
            np.median(
                intra
            )
        )
        -
        float(
            np.median(
                inter
            )
        )
    )

    print(
        f"Centro fraco:       {centros[classe_fraca]:.6f}"
    )

    print(
        f"Centro forte:       {centros[classe_forte]:.6f}"
    )

    print(
        f"Corte automático:   {corte:.6f}"
    )

    print(
        f"Gap KMeans:          {gap_classes:.6f}"
    )

    print(
        f"Componentes:         {len(componentes)}"
    )

    print(
        f"Tamanho mínimo:      {min(tamanhos)}"
    )

    print(
        f"Tamanho máximo:      {max(tamanhos)}"
    )

    print(
        f"ARI:                 {ari:.6f}"
    )

    print(
        f"NMI:                 {nmi:.6f}"
    )

    print(
        f"Pureza:              {pureza:.6f}"
    )

    print(
        f"Mediana intra:       {np.median(intra):.6f}"
    )

    print(
        f"Mediana inter:       {np.median(inter):.6f}"
    )

    print(
        f"Contraste observado: {contraste_mediana:.6f}"
    )

    return {
        "Cenario":
            cenario,

        "Alpha_Grupo":
            alpha,

        "Beta_Global":
            beta,

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

        "Gap_KMeans":
            gap_classes,

        "Corte":
            corte,

        "N_Arestas":
            n_arestas,

        "N_Componentes":
            len(
                componentes
            ),

        "Min_Tamanho":
            min(
                tamanhos
            ),

        "Max_Tamanho":
            max(
                tamanhos
            ),

        "ARI":
            ari,

        "NMI":
            nmi,

        "Pureza":
            pureza,

        "Mediana_Intra":
            float(
                np.median(
                    intra
                )
            ),

        "Mediana_Inter":
            float(
                np.median(
                    inter
                )
            ),

        "Contraste_Observado":
            contraste_mediana,

        "Arquivo_Mapa_Descoberto":
            arquivo_mapa,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 104)
    print("DESCOBERTA DE GRUPOS — EXPERIMENTO 4: FRONTEIRA ESTRUTURAL")
    print("=" * 104)

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    if len(manifesto) != 12:

        raise RuntimeError(
            "Esperados exatamente 12 cenários."
        )

    resultados = []

    for _, linha in manifesto.iterrows():

        resultado = analisar_cenario(
            cenario=str(
                linha[
                    "Cenario"
                ]
            ),

            arquivo_treino=str(
                linha[
                    "Arquivo_Treino"
                ]
            ),

            alpha=float(
                linha[
                    "Alpha_Grupo"
                ]
            ),

            beta=float(
                linha[
                    "Beta_Global"
                ]
            ),
        )

        resultados.append(
            resultado
        )

    resumo = pd.DataFrame(
        resultados
    )

    arquivo_resumo = os.path.join(
        DIRETORIO_SAIDA,
        "resumo_descoberta_grupos.csv",
    )

    resumo.to_csv(
        arquivo_resumo,
        index=False,
    )

    print()
    print("=" * 104)
    print("RESUMO DA FRONTEIRA DE DESCOBERTA")
    print("=" * 104)

    print()

    print(
        resumo[
            [
                "Cenario",
                "Alpha_Grupo",
                "Beta_Global",
                "Contraste_Observado",
                "N_Componentes",
                "Min_Tamanho",
                "Max_Tamanho",
                "ARI",
                "NMI",
                "Pureza",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Os grupos foram construídos exclusivamente a partir "
        "do treino normal de cada cenário."
    )

    print(
        "O mapa verdadeiro foi consultado somente após a descoberta "
        "para calcular ARI, NMI, pureza e caracterizar intra/inter."
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