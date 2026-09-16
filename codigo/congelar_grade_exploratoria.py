import json
import os

import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_GRADE = os.path.join(
    PROJETO,
    "dados",
    "grade_exploratoria.csv"
)

ARQUIVO_PROTOCOLO = os.path.join(
    PROJETO,
    "dados",
    "protocolo_grade_exploratoria.json"
)


# ============================================================================
# ESTRUTURA CONGELADA DO DATASET
# ============================================================================

N_GRUPOS = 20

N_FEATURES_POR_GRUPO = 20

N_FLOATS_POR_GRUPO = 14

N_BINARIAS_POR_GRUPO = 6

N_FEATURES_TOTAL = 400

DURACAO_EVENTO_S = 100

FPR_ALVO_VALIDACAO = 0.01


# ============================================================================
# GRADE EXPLORATÓRIA
#
# Definida ANTES da execução da varredura completa.
# ============================================================================

VALORES_M = [
    2,
    4,
    6,
    8,
    10,
    12,
    14,
]

VALORES_DELTA = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]


# ============================================================================
# MÉTODOS QUE SERÃO COMPARADOS
# ============================================================================

METODOS = [
    "Energia",
    "MAX",
    "TOP3",
    "TOP5",
    "TOP10",
    "Mahalanobis",
]

MODOS = [
    "Global",
    "Local",
]


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 72)
    print("CONGELAMENTO DA GRADE EXPLORATÓRIA")
    print("=" * 72)

    print(
        f"Valores de m: "
        f"{VALORES_M}"
    )

    print(
        f"Valores de delta: "
        f"{VALORES_DELTA}"
    )

    registros = []

    indice = 1

    for m in VALORES_M:

        for delta in VALORES_DELTA:

            registros.append(
                {
                    "Condicao":
                        f"C{indice:03d}",

                    "m":
                        m,

                    "delta_latente":
                        delta,

                    "fracao_dimensao_total":
                        m
                        /
                        N_FEATURES_TOTAL,

                    "fracao_grupo_total":
                        m
                        /
                        N_FEATURES_POR_GRUPO,

                    "fracao_floats_grupo":
                        m
                        /
                        N_FLOATS_POR_GRUPO,

                    "duracao_evento_s":
                        DURACAO_EVENTO_S,

                    "fpr_alvo_validacao":
                        FPR_ALVO_VALIDACAO,
                }
            )

            indice += 1

    grade = pd.DataFrame(
        registros
    )

    if len(grade) != 56:
        raise RuntimeError(
            "A grade deveria possuir exatamente 56 condições."
        )

    grade.to_csv(
        ARQUIVO_GRADE,
        index=False
    )

    protocolo = {
        "tipo":
            "grade_exploratoria_predefinida",

        "n_grupos":
            N_GRUPOS,

        "n_features_por_grupo":
            N_FEATURES_POR_GRUPO,

        "n_floats_por_grupo":
            N_FLOATS_POR_GRUPO,

        "n_binarias_por_grupo":
            N_BINARIAS_POR_GRUPO,

        "n_features_total":
            N_FEATURES_TOTAL,

        "duracao_evento_s":
            DURACAO_EVENTO_S,

        "fpr_alvo_validacao":
            FPR_ALVO_VALIDACAO,

        "valores_m":
            VALORES_M,

        "valores_delta":
            VALORES_DELTA,

        "n_condicoes":
            len(grade),

        "metodos":
            METODOS,

        "modos":
            MODOS,

        "observacoes": [
            (
                "Cada evento afeta somente um dos 20 processos."
            ),
            (
                "As m variaveis afetadas pertencem ao mesmo processo."
            ),
            (
                "Somente variaveis float sao perturbadas nesta grade."
            ),
            (
                "As variaveis escolhidas sao aninhadas: "
                "m maior reutiliza as variaveis de m menor."
            ),
            (
                "Os ataques permanecem dentro dos ranges fisicos."
            ),
            (
                "Treino e validacao permanecem exclusivamente normais."
            ),
            (
                "Limiar operacional e calibrado somente na validacao normal."
            ),
            (
                "O casamento exato de FPR no teste sera usado apenas "
                "como diagnostico complementar e sera identificado como tal."
            ),
            (
                "A grade foi definida antes da execucao da varredura completa."
            ),
        ],
    }

    with open(
        ARQUIVO_PROTOCOLO,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            protocolo,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        f"Condições geradas: {len(grade)}"
    )

    print()
    print(
        grade.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("GRADE CONGELADA")
    print("=" * 72)

    print()
    print(
        "CSV:"
    )

    print(
        ARQUIVO_GRADE
    )

    print()
    print(
        "Protocolo:"
    )

    print(
        ARQUIVO_PROTOCOLO
    )


if __name__ == "__main__":
    main()