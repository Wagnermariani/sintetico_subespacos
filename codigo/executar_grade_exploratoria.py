import os

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
)

from gerar_normal_sintetico import GANHO_TANH


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_TREINO = os.path.join(
    PROJETO,
    "dados",
    "treino_normal_piloto.csv"
)

ARQUIVO_VALIDACAO = os.path.join(
    PROJETO,
    "dados",
    "validacao_normal_piloto.csv"
)

ARQUIVO_TESTE_BASE = os.path.join(
    PROJETO,
    "dados",
    "teste_normal_base_piloto.csv"
)

ARQUIVO_GRADE = os.path.join(
    PROJETO,
    "dados",
    "grade_exploratoria.csv"
)

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_ataque_piloto_m10_delta1.csv"
)

ARQUIVO_MAPA_VERDADEIRO = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv"
)

ARQUIVO_GRUPOS_DESCOBERTOS = os.path.join(
    PROJETO,
    "resultado",
    "agrupamento_normal",
    "mapa_grupos_descobertos.csv"
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "grade_exploratoria"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# PARÂMETROS JÁ DEFINIDOS NO PILOTO
# ============================================================================

FPR_ALVO = 0.01

SEED_ORDEM_VARIAVEIS = 2042

N_FEATURES_TOTAL = 400


# ============================================================================
# NORMALIZAÇÃO PELOS LIMITES FÍSICOS
# ============================================================================

def normalizar_fisicamente(
    df,
    features,
    parametros
):

    X = df[
        features
    ].copy()

    for feature in features:

        if feature in parametros:

            minimo = parametros[
                feature
            ][
                "Min_Fisico"
            ]

            maximo = parametros[
                feature
            ][
                "Max_Fisico"
            ]

            X[
                feature
            ] = (
                X[
                    feature
                ]
                -
                minimo
            ) / (
                maximo
                -
                minimo
            )

    return X.to_numpy(
        dtype=np.float64
    )


# ============================================================================
# SCORES ESTÁTICOS
# ============================================================================

def score_topk(
    Z2,
    k
):

    maiores = np.partition(
        Z2,
        -k,
        axis=1
    )[
        :,
        -k:
    ]

    return np.mean(
        maiores,
        axis=1
    )


def calcular_scores_estaticos(
    Z2
):

    return {
        "Energia":
            np.mean(
                Z2,
                axis=1
            ),

        "MAX":
            np.max(
                Z2,
                axis=1
            ),

        "TOP3":
            score_topk(
                Z2,
                3
            ),

        "TOP5":
            score_topk(
                Z2,
                5
            ),

        "TOP10":
            score_topk(
                Z2,
                10
            ),
    }


# ============================================================================
# MÉTRICAS
# ============================================================================

def avaliar_limiar(
    scores,
    y_true,
    limiar
):

    pred = (
        scores
        >
        limiar
    ).astype(
        np.int8
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            pred,
            labels=[0, 1]
        )
        .ravel()
    )

    fpr = (
        fp
        /
        (
            fp
            +
            tn
        )
    )

    precision = precision_score(
        y_true,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        pred
    )

    return {
        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "FPR":
            float(fpr),

        "Precision":
            float(precision),

        "Recall":
            float(recall),

        "F1":
            float(f1),

        "MCC":
            float(mcc),
    }


# ============================================================================
# DIAGNÓSTICO SOB FPR <= 1% NO TESTE
#
# Usa os rótulos do teste.
# Portanto NÃO é limiar operacional.
# Serve apenas para mapear potência em FPR controlado.
# ============================================================================

def diagnostico_fpr_controlado(
    scores,
    y_true,
    fpr_maximo
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        drop_intermediate=False
    )

    validos = np.where(
        fpr
        <=
        (
            fpr_maximo
            +
            1e-12
        )
    )[0]

    if len(validos) == 0:
        raise RuntimeError(
            "Nenhum ponto ROC dentro do FPR solicitado."
        )

    melhor_tpr = np.max(
        tpr[
            validos
        ]
    )

    candidatos = validos[
        np.isclose(
            tpr[
                validos
            ],
            melhor_tpr
        )
    ]

    # Em empate de TPR, escolhe o maior FPR ainda permitido.
    indice = candidatos[
        np.argmax(
            fpr[
                candidatos
            ]
        )
    ]

    return {
        "Recall_Diagnostico":
            float(
                tpr[
                    indice
                ]
            ),

        "FPR_Diagnostico":
            float(
                fpr[
                    indice
                ]
            ),

        "Limiar_Diagnostico":
            float(
                thresholds[
                    indice
                ]
            ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 84)
    print("EXECUÇÃO DA GRADE EXPLORATÓRIA")
    print("=" * 84)

    print(
        "A grade foi definida antes desta execução."
    )

    print(
        "Nenhum parâmetro estrutural será alterado durante a varredura."
    )

    # ------------------------------------------------------------------------
    # LEITURA
    # ------------------------------------------------------------------------

    print(
        "\nCarregando arquivos..."
    )

    treino = pd.read_csv(
        ARQUIVO_TREINO
    )

    validacao = pd.read_csv(
        ARQUIVO_VALIDACAO
    )

    teste_base = pd.read_csv(
        ARQUIVO_TESTE_BASE
    )

    grade = pd.read_csv(
        ARQUIVO_GRADE
    )

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    mapa_verdadeiro = pd.read_csv(
        ARQUIVO_MAPA_VERDADEIRO
    )

    grupos_df = pd.read_csv(
        ARQUIVO_GRUPOS_DESCOBERTOS
    )

    parametros_df = pd.read_csv(
        ARQUIVO_PARAMETROS
    )

    print(
        f"Treino:       {treino.shape}"
    )

    print(
        f"Validação:    {validacao.shape}"
    )

    print(
        f"Teste base:   {teste_base.shape}"
    )

    print(
        f"Condições:    {len(grade)}"
    )

    print(
        f"Eventos:      {len(eventos)}"
    )

    # ------------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------------

    features = (
        grupos_df[
            "Feature"
        ]
        .tolist()
    )

    if len(features) != N_FEATURES_TOTAL:
        raise RuntimeError(
            "Esperadas exatamente 400 features."
        )

    indice_feature = {
        feature: i
        for i, feature
        in enumerate(
            features
        )
    }

    parametros = (
        parametros_df
        .set_index(
            "Feature"
        )[
            [
                "Min_Fisico",
                "Max_Fisico",
            ]
        ]
        .to_dict(
            orient="index"
        )
    )

    # ------------------------------------------------------------------------
    # NORMALIZAÇÃO FÍSICA
    # ------------------------------------------------------------------------

    print(
        "\nNormalizando pelos ranges físicos..."
    )

    X_train = normalizar_fisicamente(
        treino,
        features,
        parametros
    )

    X_val = normalizar_fisicamente(
        validacao,
        features,
        parametros
    )

    X_test_base = normalizar_fisicamente(
        teste_base,
        features,
        parametros
    )

    # ------------------------------------------------------------------------
    # MÁSCARA DOS EVENTOS
    # ------------------------------------------------------------------------

    mascara_ataque = np.zeros(
        len(teste_base),
        dtype=bool
    )

    for _, evento in eventos.iterrows():

        inicio = int(
            evento[
                "Indice_Inicio"
            ]
        )

        fim = int(
            evento[
                "Indice_Fim"
            ]
        )

        mascara_ataque[
            inicio:fim + 1
        ] = True

    indices_ataque = np.where(
        mascara_ataque
    )[0]

    y_test = (
        mascara_ataque
        .astype(
            np.int8
        )
    )

    if len(indices_ataque) != 2000:
        raise RuntimeError(
            "Esperados exatamente 2000 pontos de ataque."
        )

    posicao_ataque = np.full(
        len(teste_base),
        -1,
        dtype=np.int64
    )

    posicao_ataque[
        indices_ataque
    ] = np.arange(
        len(
            indices_ataque
        )
    )

    # ------------------------------------------------------------------------
    # ORDEM FIXA E ANINHADA DAS FLOATS
    #
    # O mapa verdadeiro é usado SOMENTE pelo GERADOR do ataque.
    # Não participa dos detectores.
    # ------------------------------------------------------------------------

    print(
        "\nConstruindo ordem fixa das variáveis atacáveis..."
    )

    rng_ordem = np.random.default_rng(
        SEED_ORDEM_VARIAVEIS
    )

    grupos_verdadeiros = sorted(
        mapa_verdadeiro[
            "Grupo_Verdadeiro"
        ]
        .unique()
        .tolist()
    )

    ordem_por_grupo = {}

    for grupo in grupos_verdadeiros:

        floats = (
            mapa_verdadeiro.loc[
                (
                    mapa_verdadeiro[
                        "Grupo_Verdadeiro"
                    ]
                    ==
                    grupo
                )
                &
                (
                    mapa_verdadeiro[
                        "Tipo"
                    ]
                    ==
                    "float"
                ),
                "Feature"
            ]
            .tolist()
        )

        if len(floats) != 14:
            raise RuntimeError(
                f"{grupo}: esperadas 14 floats."
            )

        ordem_por_grupo[
            grupo
        ] = list(
            rng_ordem.permutation(
                floats
            )
        )

    # ------------------------------------------------------------------------
    # GRUPOS DESCOBERTOS
    # ------------------------------------------------------------------------

    grupos_descobertos = sorted(
        grupos_df[
            "Grupo_Descoberto"
        ]
        .unique()
        .tolist()
    )

    indices_por_grupo = {}

    for grupo in grupos_descobertos:

        features_grupo = (
            grupos_df.loc[
                grupos_df[
                    "Grupo_Descoberto"
                ]
                ==
                grupo,
                "Feature"
            ]
            .tolist()
        )

        indices_por_grupo[
            grupo
        ] = np.asarray(
            [
                indice_feature[
                    feature
                ]
                for feature
                in features_grupo
            ],
            dtype=np.int64
        )

    feature_para_grupo_descoberto = (
        grupos_df
        .set_index(
            "Feature"
        )[
            "Grupo_Descoberto"
        ]
        .to_dict()
    )

    pos_grupo_descoberto = {
        grupo: i
        for i, grupo
        in enumerate(
            grupos_descobertos
        )
    }

    # ========================================================================
    # PADRONIZAÇÃO ESTÁTICA — APENAS TREINO
    # ========================================================================

    print(
        "\nPreparando Energia/MAX/TOP-k..."
    )

    media_train = np.mean(
        X_train,
        axis=0
    )

    desvio_train = np.std(
        X_train,
        axis=0,
        ddof=0
    )

    if np.any(
        desvio_train
        <
        1e-12
    ):
        raise RuntimeError(
            "Feature com desvio padrão praticamente zero."
        )

    Z2_val = (
        (
            X_val
            -
            media_train
        )
        /
        desvio_train
    ) ** 2

    Z2_test_base = (
        (
            X_test_base
            -
            media_train
        )
        /
        desvio_train
    ) ** 2

    nomes_estaticos = [
        "Energia",
        "MAX",
        "TOP3",
        "TOP5",
        "TOP10",
    ]

    scores_val = {}

    scores_base_teste = {}

    # ------------------------------------------------------------------------
    # GLOBAL ESTÁTICO
    # ------------------------------------------------------------------------

    estaticos_val_global = (
        calcular_scores_estaticos(
            Z2_val
        )
    )

    estaticos_base_global = (
        calcular_scores_estaticos(
            Z2_test_base
        )
    )

    for nome in nomes_estaticos:

        scores_val[
            f"Global_{nome}"
        ] = (
            estaticos_val_global[
                nome
            ]
        )

        scores_base_teste[
            f"Global_{nome}"
        ] = (
            estaticos_base_global[
                nome
            ]
        )

    # ------------------------------------------------------------------------
    # LOCAL ESTÁTICO
    # ------------------------------------------------------------------------

    val_local_por_metodo = {
        nome: []
        for nome
        in nomes_estaticos
    }

    base_local_por_metodo = {
        nome: []
        for nome
        in nomes_estaticos
    }

    for grupo in grupos_descobertos:

        idx = (
            indices_por_grupo[
                grupo
            ]
        )

        grupo_val = calcular_scores_estaticos(
            Z2_val[
                :,
                idx
            ]
        )

        grupo_base = calcular_scores_estaticos(
            Z2_test_base[
                :,
                idx
            ]
        )

        for nome in nomes_estaticos:

            val_local_por_metodo[
                nome
            ].append(
                grupo_val[
                    nome
                ]
            )

            base_local_por_metodo[
                nome
            ].append(
                grupo_base[
                    nome
                ]
            )

    for nome in nomes_estaticos:

        scores_val[
            f"Local_{nome}"
        ] = np.max(
            np.column_stack(
                val_local_por_metodo[
                    nome
                ]
            ),
            axis=1
        )

        scores_base_teste[
            f"Local_{nome}"
        ] = np.max(
            np.column_stack(
                base_local_por_metodo[
                    nome
                ]
            ),
            axis=1
        )

    # ========================================================================
    # MAHALANOBIS GLOBAL
    # ========================================================================

    print(
        "\nAjustando Mahalanobis global..."
    )

    modelo_global = LedoitWolf(
        store_precision=True,
        assume_centered=False
    )

    modelo_global.fit(
        X_train
    )

    print(
        "Shrinkage global: "
        f"{modelo_global.shrinkage_:.10f}"
    )

    scores_val[
        "Global_Mahalanobis"
    ] = modelo_global.mahalanobis(
        X_val
    )

    centered_base = (
        X_test_base
        -
        modelo_global.location_
    )

    # Produto calculado uma única vez.
    centered_precision = (
        centered_base
        @
        modelo_global.precision_
    )

    scores_global_base = np.sum(
        centered_base
        *
        centered_precision,
        axis=1
    )

    scores_base_teste[
        "Global_Mahalanobis"
    ] = scores_global_base

    # ========================================================================
    # MAHALANOBIS LOCAL
    # ========================================================================

    print(
        "\nAjustando Mahalanobis local..."
    )

    modelos_locais = {}

    scores_val_locais = []

    scores_base_locais = []

    for grupo in grupos_descobertos:

        idx = (
            indices_por_grupo[
                grupo
            ]
        )

        modelo = LedoitWolf(
            store_precision=True,
            assume_centered=False
        )

        modelo.fit(
            X_train[
                :,
                idx
            ]
        )

        modelos_locais[
            grupo
        ] = modelo

        score_val_grupo = (
            modelo.mahalanobis(
                X_val[
                    :,
                    idx
                ]
            )
        )

        score_base_grupo = (
            modelo.mahalanobis(
                X_test_base[
                    :,
                    idx
                ]
            )
        )

        scores_val_locais.append(
            score_val_grupo
        )

        scores_base_locais.append(
            score_base_grupo
        )

        print(
            f"{grupo} | "
            f"shrinkage="
            f"{modelo.shrinkage_:.8f}"
        )

    scores_val_locais = np.column_stack(
        scores_val_locais
    )

    scores_base_locais = np.column_stack(
        scores_base_locais
    )

    scores_val[
        "Local_Mahalanobis"
    ] = np.max(
        scores_val_locais,
        axis=1
    )

    scores_base_teste[
        "Local_Mahalanobis"
    ] = np.max(
        scores_base_locais,
        axis=1
    )

    # ========================================================================
    # LIMIARES OPERACIONAIS — SOMENTE VALIDAÇÃO NORMAL
    # ========================================================================

    print()
    print("=" * 84)
    print("LIMIARES DA VALIDAÇÃO NORMAL")
    print("=" * 84)

    limiares = {}

    for metodo, score in scores_val.items():

        limiar = float(
            np.quantile(
                score,
                1.0
                -
                FPR_ALVO
            )
        )

        limiares[
            metodo
        ] = limiar

        positivos = int(
            np.sum(
                score
                >
                limiar
            )
        )

        print(
            f"{metodo:22s} | "
            f"limiar={limiar:.8f} | "
            f"positivos={positivos}"
        )

    # ========================================================================
    # QUANTIS DO TREINO PARA PLAUSIBILIDADE
    # ========================================================================

    q001 = np.quantile(
        X_train,
        0.001,
        axis=0
    )

    q999 = np.quantile(
        X_train,
        0.999,
        axis=0
    )

    q01 = np.quantile(
        X_train,
        0.01,
        axis=0
    )

    q99 = np.quantile(
        X_train,
        0.99,
        axis=0
    )

    min_train = np.min(
        X_train,
        axis=0
    )

    max_train = np.max(
        X_train,
        axis=0
    )

    # ========================================================================
    # BASES PARA ATUALIZAÇÃO EFICIENTE DO MAHALANOBIS
    # ========================================================================

    X_base_ataque = (
        X_test_base[
            indices_ataque
        ]
    )

    score_global_base_ataque = (
        scores_global_base[
            indices_ataque
        ]
    )

    cp_base_ataque = (
        centered_precision[
            indices_ataque
        ]
    )

    scores_local_base_ataque = (
        scores_base_locais[
            indices_ataque
        ]
    )

    score_local_max_base_ataque = np.max(
        scores_local_base_ataque,
        axis=1
    )

    # ========================================================================
    # VARREDURA
    # ========================================================================

    resultados_metodos = []

    resultados_condicoes = []

    print()
    print("=" * 84)
    print("INICIANDO AS 56 CONDIÇÕES")
    print("=" * 84)

    for _, condicao in grade.iterrows():

        nome_condicao = str(
            condicao[
                "Condicao"
            ]
        )

        m = int(
            condicao[
                "m"
            ]
        )

        delta = float(
            condicao[
                "delta_latente"
            ]
        )

        # --------------------------------------------------------------------
        # Apenas os 2000 pontos de ataque precisam ser modificados.
        # --------------------------------------------------------------------

        X_ataque = (
            X_base_ataque.copy()
        )

        modificacoes = []

        n_modificados = 0
        fora_range = 0
        fora_minmax = 0
        fora_p001_p999 = 0
        fora_p01_p99 = 0

        for _, evento in eventos.iterrows():

            grupo_verdadeiro = str(
                evento[
                    "Grupo_Atacado"
                ]
            )

            inicio = int(
                evento[
                    "Indice_Inicio"
                ]
            )

            fim = int(
                evento[
                    "Indice_Fim"
                ]
            )

            direcao = int(
                evento[
                    "Direcao"
                ]
            )

            linhas_globais = np.arange(
                inicio,
                fim + 1
            )

            pos = (
                posicao_ataque[
                    linhas_globais
                ]
            )

            features_atacadas = (
                ordem_por_grupo[
                    grupo_verdadeiro
                ][
                    :m
                ]
            )

            idx = np.asarray(
                [
                    indice_feature[
                        feature
                    ]
                    for feature
                    in features_atacadas
                ],
                dtype=np.int64
            )

            grupos_destino = {
                feature_para_grupo_descoberto[
                    feature
                ]
                for feature
                in features_atacadas
            }

            if len(
                grupos_destino
            ) != 1:
                raise RuntimeError(
                    f"{nome_condicao}: ataque atravessou "
                    f"mais de um grupo descoberto."
                )

            grupo_descoberto = next(
                iter(
                    grupos_destino
                )
            )

            # ---------------------------------------------------------------
            # Valores normalizados fisicamente já estão em (0,1).
            # Recupera coordenada latente, aplica delta e retorna a (0,1).
            # ---------------------------------------------------------------

            valores_base = (
                X_base_ataque[
                    np.ix_(
                        pos,
                        idx
                    )
                ]
            )

            eps = 1e-12

            valores_seguros = np.clip(
                valores_base,
                eps,
                1.0
                -
                eps
            )

            latente = (
                np.arctanh(
                    2.0
                    *
                    valores_seguros
                    -
                    1.0
                )
                /
                GANHO_TANH
            )

            valores_atacados = (
                np.tanh(
                    GANHO_TANH
                    *
                    (
                        latente
                        +
                        direcao
                        *
                        delta
                    )
                )
                +
                1.0
            ) / 2.0

            X_ataque[
                np.ix_(
                    pos,
                    idx
                )
            ] = valores_atacados

            diferenca = (
                valores_atacados
                -
                valores_base
            )

            modificacoes.append(
                {
                    "pos":
                        pos,

                    "idx":
                        idx,

                    "diferenca":
                        diferenca,

                    "grupo_descoberto":
                        grupo_descoberto,
                }
            )

            # ---------------------------------------------------------------
            # PLAUSIBILIDADE UNIVARIADA
            # ---------------------------------------------------------------

            n_modificados += (
                valores_atacados.size
            )

            fora_range += int(
                np.sum(
                    (
                        valores_atacados
                        <
                        0.0
                    )
                    |
                    (
                        valores_atacados
                        >
                        1.0
                    )
                )
            )

            fora_minmax += int(
                np.sum(
                    (
                        valores_atacados
                        <
                        min_train[
                            idx
                        ]
                    )
                    |
                    (
                        valores_atacados
                        >
                        max_train[
                            idx
                        ]
                    )
                )
            )

            fora_p001_p999 += int(
                np.sum(
                    (
                        valores_atacados
                        <
                        q001[
                            idx
                        ]
                    )
                    |
                    (
                        valores_atacados
                        >
                        q999[
                            idx
                        ]
                    )
                )
            )

            fora_p01_p99 += int(
                np.sum(
                    (
                        valores_atacados
                        <
                        q01[
                            idx
                        ]
                    )
                    |
                    (
                        valores_atacados
                        >
                        q99[
                            idx
                        ]
                    )
                )
            )

        if fora_range != 0:
            raise RuntimeError(
                f"{nome_condicao}: ataque saiu do range físico."
            )

        # ====================================================================
        # SCORES ESTÁTICOS DA CONDIÇÃO
        # ====================================================================

        Z2_ataque = (
            (
                X_ataque
                -
                media_train
            )
            /
            desvio_train
        ) ** 2

        scores_condicao_ataque = {}

        # --------------------------------------------------------------------
        # GLOBAL
        # --------------------------------------------------------------------

        estaticos_global = (
            calcular_scores_estaticos(
                Z2_ataque
            )
        )

        for nome in nomes_estaticos:

            scores_condicao_ataque[
                f"Global_{nome}"
            ] = (
                estaticos_global[
                    nome
                ]
            )

        # --------------------------------------------------------------------
        # LOCAL
        # --------------------------------------------------------------------

        locais_por_metodo = {
            nome: []
            for nome
            in nomes_estaticos
        }

        for grupo in grupos_descobertos:

            idx_grupo = (
                indices_por_grupo[
                    grupo
                ]
            )

            scores_grupo = (
                calcular_scores_estaticos(
                    Z2_ataque[
                        :,
                        idx_grupo
                    ]
                )
            )

            for nome in nomes_estaticos:

                locais_por_metodo[
                    nome
                ].append(
                    scores_grupo[
                        nome
                    ]
                )

        for nome in nomes_estaticos:

            scores_condicao_ataque[
                f"Local_{nome}"
            ] = np.max(
                np.column_stack(
                    locais_por_metodo[
                        nome
                    ]
                ),
                axis=1
            )

        # ====================================================================
        # MAHALANOBIS GLOBAL — ATUALIZAÇÃO DO QUADRÁTICO
        # ====================================================================

        score_mah_global = (
            score_global_base_ataque.copy()
        )

        for mod in modificacoes:

            pos = mod[
                "pos"
            ]

            idx = mod[
                "idx"
            ]

            d = mod[
                "diferenca"
            ]

            termo_linear = (
                2.0
                *
                np.sum(
                    d
                    *
                    cp_base_ataque[
                        np.ix_(
                            pos,
                            idx
                        )
                    ],
                    axis=1
                )
            )

            precisao_sub = (
                modelo_global.precision_[
                    np.ix_(
                        idx,
                        idx
                    )
                ]
            )

            termo_quadratico = np.einsum(
                "ij,jk,ik->i",
                d,
                precisao_sub,
                d
            )

            score_mah_global[
                pos
            ] = (
                score_global_base_ataque[
                    pos
                ]
                +
                termo_linear
                +
                termo_quadratico
            )

        scores_condicao_ataque[
            "Global_Mahalanobis"
        ] = score_mah_global

        # ====================================================================
        # MAHALANOBIS LOCAL
        # ====================================================================

        score_mah_local = (
            score_local_max_base_ataque.copy()
        )

        for mod in modificacoes:

            pos = mod[
                "pos"
            ]

            grupo = mod[
                "grupo_descoberto"
            ]

            idx_grupo = (
                indices_por_grupo[
                    grupo
                ]
            )

            modelo = (
                modelos_locais[
                    grupo
                ]
            )

            novo_score_grupo = (
                modelo.mahalanobis(
                    X_ataque[
                        np.ix_(
                            pos,
                            idx_grupo
                        )
                    ]
                )
            )

            indice_grupo = (
                pos_grupo_descoberto[
                    grupo
                ]
            )

            outros = np.delete(
                scores_local_base_ataque[
                    pos,
                    :
                ],
                indice_grupo,
                axis=1
            )

            max_outros = np.max(
                outros,
                axis=1
            )

            score_mah_local[
                pos
            ] = np.maximum(
                novo_score_grupo,
                max_outros
            )

        scores_condicao_ataque[
            "Local_Mahalanobis"
        ] = score_mah_local

        # ====================================================================
        # AVALIAÇÃO DOS 12 MÉTODOS
        # ====================================================================

        linhas_condicao = []

        for metodo in sorted(
            scores_condicao_ataque.keys()
        ):

            scores_completos = (
                scores_base_teste[
                    metodo
                ].copy()
            )

            scores_completos[
                indices_ataque
            ] = (
                scores_condicao_ataque[
                    metodo
                ]
            )

            operacional = avaliar_limiar(
                scores_completos,
                y_test,
                limiares[
                    metodo
                ]
            )

            diagnostico = (
                diagnostico_fpr_controlado(
                    scores_completos,
                    y_test,
                    FPR_ALVO
                )
            )

            modo = (
                "Global"
                if metodo.startswith(
                    "Global_"
                )
                else
                "Local"
            )

            estatistica = metodo.split(
                "_",
                1
            )[1]

            registro = {
                "Condicao":
                    nome_condicao,

                "m":
                    m,

                "delta_latente":
                    delta,

                "Modo":
                    modo,

                "Estatistica":
                    estatistica,

                "Metodo":
                    metodo,

                "Limiar_Validacao":
                    limiares[
                        metodo
                    ],

                "TP_Validacao":
                    operacional[
                        "TP"
                    ],

                "FP_Validacao":
                    operacional[
                        "FP"
                    ],

                "FN_Validacao":
                    operacional[
                        "FN"
                    ],

                "TN_Validacao":
                    operacional[
                        "TN"
                    ],

                "Recall_Validacao":
                    operacional[
                        "Recall"
                    ],

                "FPR_Teste_Validacao":
                    operacional[
                        "FPR"
                    ],

                "Precision_Validacao":
                    operacional[
                        "Precision"
                    ],

                "F1_Validacao":
                    operacional[
                        "F1"
                    ],

                "MCC_Validacao":
                    operacional[
                        "MCC"
                    ],

                "Recall_Diagnostico_FPR1":
                    diagnostico[
                        "Recall_Diagnostico"
                    ],

                "FPR_Diagnostico":
                    diagnostico[
                        "FPR_Diagnostico"
                    ],

                "Limiar_Diagnostico":
                    diagnostico[
                        "Limiar_Diagnostico"
                    ],

                "N_Valores_Modificados":
                    n_modificados,

                "Pct_Fora_MinMax_Treino":
                    (
                        100.0
                        *
                        fora_minmax
                        /
                        n_modificados
                    ),

                "Pct_Fora_P001_P999":
                    (
                        100.0
                        *
                        fora_p001_p999
                        /
                        n_modificados
                    ),

                "Pct_Fora_P01_P99":
                    (
                        100.0
                        *
                        fora_p01_p99
                        /
                        n_modificados
                    ),
            }

            resultados_metodos.append(
                registro
            )

            linhas_condicao.append(
                registro
            )

        # ====================================================================
        # RESUMO DA CONDIÇÃO
        # ====================================================================

        globais = [
            r
            for r
            in linhas_condicao
            if r[
                "Modo"
            ]
            ==
            "Global"
        ]

        locais = [
            r
            for r
            in linhas_condicao
            if r[
                "Modo"
            ]
            ==
            "Local"
        ]

        melhor_global = max(
            globais,
            key=lambda r:
                r[
                    "Recall_Diagnostico_FPR1"
                ]
        )

        melhor_local = max(
            locais,
            key=lambda r:
                r[
                    "Recall_Diagnostico_FPR1"
                ]
        )

        mah_global = next(
            r
            for r
            in linhas_condicao
            if r[
                "Metodo"
            ]
            ==
            "Global_Mahalanobis"
        )

        mah_local = next(
            r
            for r
            in linhas_condicao
            if r[
                "Metodo"
            ]
            ==
            "Local_Mahalanobis"
        )

        resumo = {
            "Condicao":
                nome_condicao,

            "m":
                m,

            "delta_latente":
                delta,

            "Melhor_Global":
                melhor_global[
                    "Metodo"
                ],

            "Recall_Melhor_Global_FPR1":
                melhor_global[
                    "Recall_Diagnostico_FPR1"
                ],

            "Melhor_Local":
                melhor_local[
                    "Metodo"
                ],

            "Recall_Melhor_Local_FPR1":
                melhor_local[
                    "Recall_Diagnostico_FPR1"
                ],

            "Vantagem_Local_Absoluta":
                (
                    melhor_local[
                        "Recall_Diagnostico_FPR1"
                    ]
                    -
                    melhor_global[
                        "Recall_Diagnostico_FPR1"
                    ]
                ),

            "Local_Supera_Todos_Globais":
                (
                    melhor_local[
                        "Recall_Diagnostico_FPR1"
                    ]
                    >
                    melhor_global[
                        "Recall_Diagnostico_FPR1"
                    ]
                ),

            "Recall_Mah_Global_FPR1":
                mah_global[
                    "Recall_Diagnostico_FPR1"
                ],

            "Recall_Mah_Local_FPR1":
                mah_local[
                    "Recall_Diagnostico_FPR1"
                ],

            "Vantagem_Mah_Local":
                (
                    mah_local[
                        "Recall_Diagnostico_FPR1"
                    ]
                    -
                    mah_global[
                        "Recall_Diagnostico_FPR1"
                    ]
                ),

            "Pct_Fora_MinMax_Treino":
                (
                    100.0
                    *
                    fora_minmax
                    /
                    n_modificados
                ),

            "Pct_Fora_P001_P999":
                (
                    100.0
                    *
                    fora_p001_p999
                    /
                    n_modificados
                ),

            "Pct_Fora_P01_P99":
                (
                    100.0
                    *
                    fora_p01_p99
                    /
                    n_modificados
                ),
        }

        resultados_condicoes.append(
            resumo
        )

        print(
            f"{nome_condicao} | "
            f"m={m:2d} "
            f"delta={delta:4.2f} | "
            f"Mah G="
            f"{resumo['Recall_Mah_Global_FPR1']:.4f} "
            f"L="
            f"{resumo['Recall_Mah_Local_FPR1']:.4f} | "
            f"best G="
            f"{resumo['Recall_Melhor_Global_FPR1']:.4f} "
            f"best L="
            f"{resumo['Recall_Melhor_Local_FPR1']:.4f} | "
            f"P01-P99 fora="
            f"{resumo['Pct_Fora_P01_P99']:.2f}%"
        )

        # --------------------------------------------------------------------
        # CHECKPOINT APÓS CADA CONDIÇÃO
        # --------------------------------------------------------------------

        pd.DataFrame(
            resultados_metodos
        ).to_csv(
            os.path.join(
                DIRETORIO_SAIDA,
                "resultados_metodos_parcial.csv"
            ),
            index=False
        )

        pd.DataFrame(
            resultados_condicoes
        ).to_csv(
            os.path.join(
                DIRETORIO_SAIDA,
                "resumo_condicoes_parcial.csv"
            ),
            index=False
        )

    # ========================================================================
    # RESULTADOS FINAIS
    # ========================================================================

    df_metodos = pd.DataFrame(
        resultados_metodos
    )

    df_condicoes = pd.DataFrame(
        resultados_condicoes
    )

    df_metodos.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resultados_metodos.csv"
        ),
        index=False
    )

    df_condicoes.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_condicoes.csv"
        ),
        index=False
    )

    # ------------------------------------------------------------------------
    # CONDIÇÕES EM QUE ALGUM LOCAL SUPERA TODOS OS GLOBAIS
    # ------------------------------------------------------------------------

    positivos = df_condicoes[
        df_condicoes[
            "Local_Supera_Todos_Globais"
        ]
        ==
        True
    ].copy()

    positivos = positivos.sort_values(
        "Vantagem_Local_Absoluta",
        ascending=False
    )

    positivos.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "condicoes_local_supera_globais.csv"
        ),
        index=False
    )

    print()
    print("=" * 84)
    print("VARREDURA CONCLUÍDA")
    print("=" * 84)

    print(
        f"Condições avaliadas: "
        f"{len(df_condicoes)}"
    )

    print(
        "Condições em que algum método local "
        "superou todos os globais sob FPR diagnóstico <= 1%: "
        f"{len(positivos)}"
    )

    if len(
        positivos
    ) > 0:

        print()
        print(
            "TOP 10 condições:"
        )

        colunas = [
            "Condicao",
            "m",
            "delta_latente",
            "Melhor_Global",
            "Recall_Melhor_Global_FPR1",
            "Melhor_Local",
            "Recall_Melhor_Local_FPR1",
            "Vantagem_Local_Absoluta",
            "Recall_Mah_Global_FPR1",
            "Recall_Mah_Local_FPR1",
            "Pct_Fora_P01_P99",
        ]

        print(
            positivos[
                colunas
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Resultados salvos em:"
    )

    print(
        DIRETORIO_SAIDA
    )

    print()
    print(
        "ATENÇÃO METODOLÓGICA:"
    )

    print(
        "Recall_Diagnostico_FPR1 usa rótulos do teste "
        "apenas para mapear a fronteira exploratória."
    )

    print(
        "A confirmação posterior deverá usar nova realização "
        "e limiares definidos somente na validação normal."
    )


if __name__ == "__main__":
    main()