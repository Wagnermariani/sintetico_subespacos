import json
import os

import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_CONDICOES = os.path.join(
    PROJETO,
    "dados",
    "condicoes_confirmatorias.csv"
)

ARQUIVO_PROTOCOLO = os.path.join(
    PROJETO,
    "dados",
    "protocolo_confirmatorio.json"
)


# ============================================================================
# REGIÃO CONGELADA APÓS A FASE EXPLORATÓRIA
# ============================================================================

VALORES_M = [
    6,
    8,
    10,
]

VALORES_DELTA = [
    1.25,
    1.50,
]

N_REALIZACOES = 5

SEEDS_REALIZACOES = [
    1001,
    2001,
    3001,
    4001,
    5001,
]

FPR_ALVO_VALIDACAO = 0.01


# ============================================================================
# BASELINES GLOBAIS PRÉ-DEFINIDOS
# ============================================================================

BASELINES_GLOBAIS = [
    "Energia",
    "MAX",
    "TOP3",
    "TOP5",
    "TOP10",
    "Mahalanobis",
]

METODO_LOCAL_PRINCIPAL = (
    "Mahalanobis_Local"
)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 76)
    print("CONGELAMENTO DO PROTOCOLO CONFIRMATÓRIO")
    print("=" * 76)

    registros = []

    indice = 1

    for m in VALORES_M:

        for delta in VALORES_DELTA:

            registros.append(
                {
                    "Condicao_Confirmatoria":
                        f"H{indice:02d}",

                    "m":
                        m,

                    "delta_latente":
                        delta,
                }
            )

            indice += 1

    condicoes = pd.DataFrame(
        registros
    )

    if len(condicoes) != 6:
        raise RuntimeError(
            "Esperadas exatamente 6 condições confirmatórias."
        )

    condicoes.to_csv(
        ARQUIVO_CONDICOES,
        index=False
    )

    protocolo = {
        "fase":
            "confirmatoria",

        "origem":
            (
                "Região selecionada após a grade exploratória "
                "pré-definida de 56 condições."
            ),

        "valores_m":
            VALORES_M,

        "valores_delta":
            VALORES_DELTA,

        "n_condicoes":
            len(condicoes),

        "n_realizacoes_independentes":
            N_REALIZACOES,

        "seeds_realizacoes":
            SEEDS_REALIZACOES,

        "fpr_alvo_validacao":
            FPR_ALVO_VALIDACAO,

        "metodo_local_principal":
            METODO_LOCAL_PRINCIPAL,

        "baselines_globais":
            BASELINES_GLOBAIS,

        "hipotese_primaria":
            (
                "Nas novas realizações independentes, o recall médio "
                "do Mahalanobis local será superior ao recall médio "
                "do melhor baseline global pré-definido, com todos "
                "os limiares calibrados exclusivamente em validação normal."
            ),

        "avaliacao_primaria":
            (
                "Delta = Recall(Mahalanobis local) - "
                "max Recall(baselines globais), calculado no teste "
                "sem utilização dos rótulos para ajuste de limiar."
            ),

        "observacoes": [
            (
                "Nenhum parâmetro da região confirmatória será "
                "alterado após a geração das novas realizações."
            ),
            (
                "Os rótulos do teste serão usados somente para avaliação."
            ),
            (
                "Não será realizado casamento de FPR usando o teste "
                "na análise confirmatória principal."
            ),
            (
                "Resultados negativos serão preservados e reportados."
            ),
            (
                "O mapa verdadeiro dos grupos poderá ser usado pelo "
                "gerador dos ataques, mas não pelos detectores."
            ),
            (
                "Os grupos utilizados pelo detector deverão ser "
                "descobertos exclusivamente a partir do normal."
            ),
            (
                "O uso operacional futuro não pressupõe conhecimento "
                "de m, delta ou de qualquer propriedade do ataque zero-day."
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
    print("Condições confirmatórias:")
    print()

    print(
        condicoes.to_string(
            index=False
        )
    )

    print()
    print(
        f"Realizações independentes: "
        f"{N_REALIZACOES}"
    )

    print(
        f"Seeds: {SEEDS_REALIZACOES}"
    )

    print()
    print("=" * 76)
    print("PROTOCOLO CONFIRMATÓRIO CONGELADO")
    print("=" * 76)

    print()
    print("Condições:")

    print(
        ARQUIVO_CONDICOES
    )

    print()
    print("Protocolo:")

    print(
        ARQUIVO_PROTOCOLO
    )


if __name__ == "__main__":
    main()