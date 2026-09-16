import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


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

ARQUIVO_TESTE = os.path.join(
    PROJETO,
    "dados",
    "teste_ataque_piloto_m10_delta1.csv"
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv"
)

ARQUIVO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "agrupamento_normal",
    "mapa_grupos_descobertos.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "baselines_estaticos_global_local"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


FPR_ALVO = 0.01


# ============================================================================
# NORMALIZAÇÃO FÍSICA
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
# SCORES
# ============================================================================

def score_topk(
    erro_quadratico,
    k
):
    maiores = np.partition(
        erro_quadratico,
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


def calcular_scores(
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
# AVALIAÇÃO
# ============================================================================

def avaliar(
    metodo,
    scores_val,
    scores_test,
    y_test
):

    limiar = float(
        np.quantile(
            scores_val,
            1.0
            -
            FPR_ALVO
        )
    )

    pred = (
        scores_test
        >
        limiar
    ).astype(
        np.int8
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
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
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        pred
    )

    print(
        f"{metodo:20s} | "
        f"TP={tp:4d} "
        f"FP={fp:4d} "
        f"Recall={recall:.6f} "
        f"FPR={100.0 * fpr:.4f}% "
        f"F1={f1:.6f}"
    )

    return {
        "Metodo":
            metodo,

        "Limiar":
            limiar,

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "MCC":
            mcc,

        "FPR_Teste":
            fpr,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("BASELINES ESTÁTICOS — GLOBAL vs LOCAL")
    print("=" * 80)

    print(
        f"FPR alvo na validação: "
        f"{100.0 * FPR_ALVO:.2f}%"
    )

    # ------------------------------------------------------------------------
    # DADOS
    # ------------------------------------------------------------------------

    treino = pd.read_csv(
        ARQUIVO_TREINO
    )

    validacao = pd.read_csv(
        ARQUIVO_VALIDACAO
    )

    teste = pd.read_csv(
        ARQUIVO_TESTE
    )

    grupos_df = pd.read_csv(
        ARQUIVO_GRUPOS
    )

    parametros_df = pd.read_csv(
        ARQUIVO_PARAMETROS
    )

    features = grupos_df[
        "Feature"
    ].tolist()

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

    print(
        f"Treino:    {treino.shape}"
    )

    print(
        f"Validação: {validacao.shape}"
    )

    print(
        f"Teste:     {teste.shape}"
    )

    print(
        f"Features:  {len(features)}"
    )

    # ------------------------------------------------------------------------
    # RANGE FÍSICO
    # ------------------------------------------------------------------------

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

    X_test = normalizar_fisicamente(
        teste,
        features,
        parametros
    )

    y_test = teste[
        "Attack"
    ].to_numpy(
        dtype=np.int8
    )

    # ------------------------------------------------------------------------
    # PADRONIZAÇÃO APRENDIDA SOMENTE NO TREINO
    # ------------------------------------------------------------------------

    media = np.mean(
        X_train,
        axis=0
    )

    desvio = np.std(
        X_train,
        axis=0,
        ddof=0
    )

    if np.any(
        desvio
        <
        1e-12
    ):
        raise RuntimeError(
            "Existe feature com desvio padrão praticamente zero."
        )

    Z_val = (
        X_val
        -
        media
    ) / desvio

    Z_test = (
        X_test
        -
        media
    ) / desvio

    Z2_val = (
        Z_val
        **
        2
    )

    Z2_test = (
        Z_test
        **
        2
    )

    # ========================================================================
    # GLOBAL
    # ========================================================================

    print()
    print("=" * 80)
    print("GLOBAL")
    print("=" * 80)

    scores_global_val = calcular_scores(
        Z2_val
    )

    scores_global_test = calcular_scores(
        Z2_test
    )

    resultados = []

    for nome in [
        "Energia",
        "MAX",
        "TOP3",
        "TOP5",
        "TOP10",
    ]:

        resultado = avaliar(
            metodo=
                f"Global_{nome}",

            scores_val=
                scores_global_val[
                    nome
                ],

            scores_test=
                scores_global_test[
                    nome
                ],

            y_test=
                y_test
        )

        resultados.append(
            resultado
        )

    # ========================================================================
    # LOCAL
    # ========================================================================

    print()
    print("=" * 80)
    print("LOCAL + MAX ENTRE GRUPOS")
    print("=" * 80)

    grupos = sorted(
        grupos_df[
            "Grupo_Descoberto"
        ]
        .unique()
        .tolist()
    )

    nomes_scores = [
        "Energia",
        "MAX",
        "TOP3",
        "TOP5",
        "TOP10",
    ]

    val_por_score = {
        nome: []
        for nome in nomes_scores
    }

    test_por_score = {
        nome: []
        for nome in nomes_scores
    }

    for grupo in grupos:

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

        indices = [
            features.index(
                feature
            )
            for feature
            in features_grupo
        ]

        scores_val_grupo = calcular_scores(
            Z2_val[
                :,
                indices
            ]
        )

        scores_test_grupo = calcular_scores(
            Z2_test[
                :,
                indices
            ]
        )

        for nome in nomes_scores:

            val_por_score[
                nome
            ].append(
                scores_val_grupo[
                    nome
                ]
            )

            test_por_score[
                nome
            ].append(
                scores_test_grupo[
                    nome
                ]
            )

    for nome in nomes_scores:

        score_val_local = np.max(
            np.column_stack(
                val_por_score[
                    nome
                ]
            ),
            axis=1
        )

        score_test_local = np.max(
            np.column_stack(
                test_por_score[
                    nome
                ]
            ),
            axis=1
        )

        resultado = avaliar(
            metodo=
                f"Local_{nome}",

            scores_val=
                score_val_local,

            scores_test=
                score_test_local,

            y_test=
                y_test
        )

        resultados.append(
            resultado
        )

    # ========================================================================
    # SALVAMENTO
    # ========================================================================

    resultado_df = pd.DataFrame(
        resultados
    )

    resultado_df.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo.csv"
        ),
        index=False
    )

    print()
    print("=" * 80)
    print("CONCLUÍDO")
    print("=" * 80)

    print(
        "\nResultados salvos em:"
    )

    print(
        DIRETORIO_SAIDA
    )

    print()
    print(
        "OBSERVAÇÃO:"
    )

    print(
        "'Energia' é média de desvios padronizados ao quadrado."
    )

    print(
        "Não deve ser chamada ainda de MSE de reconstrução."
    )


if __name__ == "__main__":
    main()