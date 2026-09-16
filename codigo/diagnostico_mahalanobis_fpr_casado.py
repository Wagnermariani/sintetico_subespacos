import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_SCORES = os.path.join(
    PROJETO,
    "resultado",
    "baseline_mahalanobis_global_local",
    "scores_teste.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "diagnostico_mahalanobis_fpr_casado"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# LIMIAR PARA UM ORÇAMENTO EXATO DE FP
# ============================================================================

def limiar_para_n_fp(
    scores,
    y_true,
    n_fp
):
    scores_normais = scores[
        y_true == 0
    ]

    if n_fp <= 0:
        return float(
            np.nextafter(
                scores_normais.max(),
                np.inf
            )
        )

    if n_fp >= len(scores_normais):
        raise ValueError(
            "n_fp deve ser menor que o número de normais."
        )

    ordenados = np.sort(
        scores_normais
    )

    indice = (
        len(ordenados)
        -
        n_fp
    )

    abaixo = ordenados[
        indice - 1
    ]

    acima = ordenados[
        indice
    ]

    limiar = (
        abaixo
        +
        acima
    ) / 2.0

    return float(
        limiar
    )


# ============================================================================
# AVALIAÇÃO
# ============================================================================

def avaliar(
    metodo,
    scores,
    y_true,
    n_fp_alvo
):
    limiar = limiar_para_n_fp(
        scores,
        y_true,
        n_fp_alvo
    )

    y_pred = (
        scores
        >
        limiar
    ).astype(
        np.int8
    )

    normal = (
        y_true == 0
    )

    ataque = (
        y_true == 1
    )

    fp = int(
        np.sum(
            y_pred[
                normal
            ]
        )
    )

    tn = int(
        np.sum(
            1
            -
            y_pred[
                normal
            ]
        )
    )

    tp = int(
        np.sum(
            y_pred[
                ataque
            ]
        )
    )

    fn = int(
        np.sum(
            1
            -
            y_pred[
                ataque
            ]
        )
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

    return {
        "Metodo":
            metodo,

        "FP_Alvo":
            n_fp_alvo,

        "Limiar_Diagnostico":
            limiar,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp,

        "FPR":
            fpr,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "MCC":
            mcc,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("DIAGNÓSTICO — MAHALANOBIS COM FPR CASADO NO TESTE")
    print("=" * 70)

    print()
    print(
        "ATENÇÃO: os rótulos do teste são usados para casar o orçamento."
    )

    print(
        "Este resultado é DIAGNÓSTICO e não define o limiar final."
    )

    df = pd.read_csv(
        ARQUIVO_SCORES
    )

    y_true = (
        df[
            "Attack"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    score_global = (
        df[
            "Score_Global"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    score_local = (
        df[
            "Score_Local_Max"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    n_normais = int(
        np.sum(
            y_true == 0
        )
    )

    n_ataques = int(
        np.sum(
            y_true == 1
        )
    )

    print()
    print(
        f"Normais: {n_normais}"
    )

    print(
        f"Ataques: {n_ataques}"
    )

    # ------------------------------------------------------------------------
    # DOIS ORÇAMENTOS
    # ------------------------------------------------------------------------

    orcamentos = [
        121,
        int(
            round(
                0.01
                *
                n_normais
            )
        ),
    ]

    resultados = []

    for n_fp in orcamentos:

        print()
        print("=" * 70)

        print(
            f"ORÇAMENTO COMUM: {n_fp} FP"
        )

        print(
            f"FPR correspondente: "
            f"{100.0 * n_fp / n_normais:.6f}%"
        )

        print("=" * 70)

        global_resultado = avaliar(
            "Mahalanobis_Global",
            score_global,
            y_true,
            n_fp
        )

        local_resultado = avaliar(
            "Mahalanobis_Local_Max",
            score_local,
            y_true,
            n_fp
        )

        resultados.extend(
            [
                global_resultado,
                local_resultado,
            ]
        )

        for resultado in [
            global_resultado,
            local_resultado,
        ]:

            print()
            print(
                resultado[
                    "Metodo"
                ]
            )

            print(
                f"TN={resultado['TN']} | "
                f"FP={resultado['FP']} | "
                f"FN={resultado['FN']} | "
                f"TP={resultado['TP']}"
            )

            print(
                f"Recall: "
                f"{resultado['Recall']:.6f}"
            )

            print(
                f"Precision: "
                f"{resultado['Precision']:.6f}"
            )

            print(
                f"F1: "
                f"{resultado['F1']:.6f}"
            )

            print(
                f"MCC: "
                f"{resultado['MCC']:.6f}"
            )

    resultado_df = pd.DataFrame(
        resultados
    )

    arquivo_saida = os.path.join(
        DIRETORIO_SAIDA,
        "resumo_fpr_casado.csv"
    )

    resultado_df.to_csv(
        arquivo_saida,
        index=False
    )

    print()
    print("=" * 70)
    print("DIAGNÓSTICO CONCLUÍDO")
    print("=" * 70)

    print()
    print(
        "Resultado salvo em:"
    )

    print(
        arquivo_saida
    )


if __name__ == "__main__":
    main()