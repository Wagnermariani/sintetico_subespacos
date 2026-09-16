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

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_ataque_piloto_m10_delta1.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "baseline_mahalanobis_global_local"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# PARÂMETROS
# ============================================================================

FPR_ALVO = 0.01


# ============================================================================
# NORMALIZAÇÃO FÍSICA
# ============================================================================

def normalizar(
    df,
    features,
    parametros_fisicos
):
    """
    Floats:
        normalização pelos limites físicos conhecidos.

    Binárias:
        permanecem 0/1.

    Nenhuma estatística da validação ou teste
    participa da normalização.
    """

    X = df[
        features
    ].copy()

    for feature in features:

        if feature in parametros_fisicos:

            minimo = (
                parametros_fisicos[
                    feature
                ][
                    "Min_Fisico"
                ]
            )

            maximo = (
                parametros_fisicos[
                    feature
                ][
                    "Max_Fisico"
                ]
            )

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
# LIMIAR
# ============================================================================

def calcular_limiar(
    scores_validacao
):
    return float(
        np.quantile(
            scores_validacao,
            1.0 - FPR_ALVO
        )
    )


# ============================================================================
# MÉTRICAS
# ============================================================================

def avaliar(
    nome,
    y_true,
    scores,
    limiar
):
    y_pred = (
        scores
        >
        limiar
    ).astype(
        np.int8
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )
        .ravel()
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
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

    print()
    print("=" * 68)
    print(nome)
    print("=" * 68)

    print(
        f"Limiar: {limiar:.6f}"
    )

    print(
        f"TN={tn} | FP={fp} | "
        f"FN={fn} | TP={tp}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall:    {recall:.6f}"
    )

    print(
        f"F1:        {f1:.6f}"
    )

    print(
        f"MCC:       {mcc:.6f}"
    )

    print(
        f"FPR teste: {100.0 * fpr:.6f}%"
    )

    return {
        "Metodo":
            nome,

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
    }, y_pred


# ============================================================================
# EVENTOS
# ============================================================================

def avaliar_eventos(
    eventos,
    pred_global,
    pred_local
):
    registros = []

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

        detectado_global = bool(
            np.any(
                pred_global[
                    inicio:fim + 1
                ]
            )
        )

        detectado_local = bool(
            np.any(
                pred_local[
                    inicio:fim + 1
                ]
            )
        )

        registros.append(
            {
                "Evento":
                    evento[
                        "Evento"
                    ],

                "Grupo_Atacado":
                    evento[
                        "Grupo_Atacado"
                    ],

                "Detectado_Global":
                    detectado_global,

                "Detectado_Local":
                    detectado_local,
            }
        )

    return pd.DataFrame(
        registros
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 68)
    print("MAHALANOBIS GLOBAL vs LOCAL")
    print("=" * 68)

    print(
        f"FPR alvo na validação: "
        f"{100.0 * FPR_ALVO:.2f}%"
    )

    # ------------------------------------------------------------------------
    # LEITURA
    # ------------------------------------------------------------------------

    print(
        "\nCarregando dados..."
    )

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

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    features = (
        grupos_df[
            "Feature"
        ]
        .tolist()
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

    print(
        "Grupos descobertos: "
        f"{grupos_df['Grupo_Descoberto'].nunique()}"
    )

    # ------------------------------------------------------------------------
    # PARÂMETROS FÍSICOS
    # ------------------------------------------------------------------------

    parametros_fisicos = (
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
    # NORMALIZAÇÃO
    # ------------------------------------------------------------------------

    print(
        "\nNormalizando pelos ranges físicos..."
    )

    X_train = normalizar(
        treino,
        features,
        parametros_fisicos
    )

    X_val = normalizar(
        validacao,
        features,
        parametros_fisicos
    )

    X_test = normalizar(
        teste,
        features,
        parametros_fisicos
    )

    y_test = (
        teste[
            "Attack"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    # ========================================================================
    # MAHALANOBIS GLOBAL
    # ========================================================================

    print()
    print("=" * 68)
    print("AJUSTANDO MAHALANOBIS GLOBAL — 400 DIMENSÕES")
    print("=" * 68)

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

    score_val_global = (
        modelo_global.mahalanobis(
            X_val
        )
    )

    score_test_global = (
        modelo_global.mahalanobis(
            X_test
        )
    )

    limiar_global = calcular_limiar(
        score_val_global
    )

    positivos_val_global = int(
        np.sum(
            score_val_global
            >
            limiar_global
        )
    )

    print(
        "Positivos validação global: "
        f"{positivos_val_global} / "
        f"{len(score_val_global)} "
        f"("
        f"{100.0 * positivos_val_global / len(score_val_global):.4f}%"
        f")"
    )

    # ========================================================================
    # MAHALANOBIS LOCAL
    # ========================================================================

    print()
    print("=" * 68)
    print("AJUSTANDO MAHALANOBIS LOCAL")
    print("=" * 68)

    grupos = sorted(
        grupos_df[
            "Grupo_Descoberto"
        ]
        .unique()
        .tolist()
    )

    score_val_local_por_grupo = []
    score_test_local_por_grupo = []

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

        X_train_grupo = (
            X_train[
                :,
                indices
            ]
        )

        X_val_grupo = (
            X_val[
                :,
                indices
            ]
        )

        X_test_grupo = (
            X_test[
                :,
                indices
            ]
        )

        modelo_local = LedoitWolf(
            store_precision=True,
            assume_centered=False
        )

        modelo_local.fit(
            X_train_grupo
        )

        score_val_grupo = (
            modelo_local.mahalanobis(
                X_val_grupo
            )
        )

        score_test_grupo = (
            modelo_local.mahalanobis(
                X_test_grupo
            )
        )

        score_val_local_por_grupo.append(
            score_val_grupo
        )

        score_test_local_por_grupo.append(
            score_test_grupo
        )

        print(
            f"{grupo} | "
            f"d={len(features_grupo):2d} | "
            f"shrinkage="
            f"{modelo_local.shrinkage_:.8f}"
        )

    score_val_local_por_grupo = np.column_stack(
        score_val_local_por_grupo
    )

    score_test_local_por_grupo = np.column_stack(
        score_test_local_por_grupo
    )

    # ------------------------------------------------------------------------
    # FUSÃO LOCAL:
    # máximo dos 20 Mahalanobis locais
    # ------------------------------------------------------------------------

    score_val_local = np.max(
        score_val_local_por_grupo,
        axis=1
    )

    score_test_local = np.max(
        score_test_local_por_grupo,
        axis=1
    )

    limiar_local = calcular_limiar(
        score_val_local
    )

    positivos_val_local = int(
        np.sum(
            score_val_local
            >
            limiar_local
        )
    )

    print()
    print(
        "Positivos validação local+max: "
        f"{positivos_val_local} / "
        f"{len(score_val_local)} "
        f"("
        f"{100.0 * positivos_val_local / len(score_val_local):.4f}%"
        f")"
    )

    # ========================================================================
    # TESTE
    # ========================================================================

    resultado_global, pred_global = avaliar(
        nome="Mahalanobis_Global",
        y_true=y_test,
        scores=score_test_global,
        limiar=limiar_global
    )

    resultado_local, pred_local = avaliar(
        nome="Mahalanobis_Local_Max",
        y_true=y_test,
        scores=score_test_local,
        limiar=limiar_local
    )

    # ========================================================================
    # EVENTOS
    # ========================================================================

    eventos_resultado = avaliar_eventos(
        eventos,
        pred_global,
        pred_local
    )

    n_global = int(
        eventos_resultado[
            "Detectado_Global"
        ].sum()
    )

    n_local = int(
        eventos_resultado[
            "Detectado_Local"
        ].sum()
    )

    print()
    print("=" * 68)
    print("DETECÇÃO POR EVENTO")
    print("=" * 68)

    print(
        f"Global: {n_global} / "
        f"{len(eventos_resultado)}"
    )

    print(
        f"Local:  {n_local} / "
        f"{len(eventos_resultado)}"
    )

    print()
    print(
        eventos_resultado.to_string(
            index=False
        )
    )

    # ========================================================================
    # SALVAMENTO
    # ========================================================================

    resumo = pd.DataFrame(
        [
            resultado_global,
            resultado_local,
        ]
    )

    resumo[
        "FPR_Alvo_Validacao"
    ] = FPR_ALVO

    resumo[
        "Positivos_Validacao"
    ] = [
        positivos_val_global,
        positivos_val_local,
    ]

    resumo.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo.csv"
        ),
        index=False
    )

    eventos_resultado.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "eventos.csv"
        ),
        index=False
    )

    pd.DataFrame(
        {
            "Score_Global":
                score_test_global,

            "Score_Local_Max":
                score_test_local,

            "Attack":
                y_test,
        }
    ).to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "scores_teste.csv"
        ),
        index=False
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
        "IMPORTANTE:"
    )

    print(
        "Este é um teste exploratório do fenômeno."
    )

    print(
        "m=10 e delta=1.0 ainda não foram "
        "selecionados como condição definitiva."
    )


if __name__ == "__main__":
    main()