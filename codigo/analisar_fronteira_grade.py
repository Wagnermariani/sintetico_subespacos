import os

import numpy as np
import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_RESUMO = os.path.join(
    PROJETO,
    "resultado",
    "grade_exploratoria",
    "resumo_condicoes.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "analise_fronteira_grade"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# FAIXAS DESCRITIVAS DE PLAUSIBILIDADE
#
# IMPORTANTE:
# Estas faixas NÃO representam critérios de aprovação.
# Servem apenas para descrever a relação entre plausibilidade univariada
# e vantagem da análise local.
# ============================================================================

LIMITES_FAIXAS = [
    -np.inf,
    5.0,
    10.0,
    15.0,
    20.0,
    np.inf,
]

NOMES_FAIXAS = [
    "0-5%",
    "5-10%",
    "10-15%",
    "15-20%",
    ">20%",
]


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 88)
    print("ANÁLISE DA FRONTEIRA DA GRADE EXPLORATÓRIA")
    print("=" * 88)

    print()
    print(
        "Esta etapa apenas analisa os resultados já produzidos."
    )

    print(
        "Nenhum ataque, limiar ou detector será recalculado."
    )

    # ------------------------------------------------------------------------
    # LEITURA
    # ------------------------------------------------------------------------

    df = pd.read_csv(
        ARQUIVO_RESUMO
    )

    print()
    print(
        f"Condições carregadas: {len(df)}"
    )

    if len(df) != 56:
        raise RuntimeError(
            "Esperadas exatamente 56 condições."
        )

    # ------------------------------------------------------------------------
    # ORDENAÇÃO NATURAL DA GRADE
    # ------------------------------------------------------------------------

    grade_ordenada = df.sort_values(
        [
            "m",
            "delta_latente",
        ]
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------------------------
    # TABELA PRINCIPAL
    # ------------------------------------------------------------------------

    colunas_principais = [
        "Condicao",
        "m",
        "delta_latente",
        "Pct_Fora_P01_P99",
        "Melhor_Global",
        "Recall_Melhor_Global_FPR1",
        "Melhor_Local",
        "Recall_Melhor_Local_FPR1",
        "Vantagem_Local_Absoluta",
        "Local_Supera_Todos_Globais",
        "Recall_Mah_Global_FPR1",
        "Recall_Mah_Local_FPR1",
        "Vantagem_Mah_Local",
    ]

    tabela_principal = (
        grade_ordenada[
            colunas_principais
        ]
        .copy()
    )

    print()
    print("=" * 88)
    print("TODAS AS 56 CONDIÇÕES")
    print("=" * 88)

    print(
        tabela_principal.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # FAIXAS DESCRITIVAS DE PLAUSIBILIDADE
    # ------------------------------------------------------------------------

    grade_ordenada[
        "Faixa_Plausibilidade_P01_P99"
    ] = pd.cut(
        grade_ordenada[
            "Pct_Fora_P01_P99"
        ],
        bins=LIMITES_FAIXAS,
        labels=NOMES_FAIXAS,
        right=False
    )

    registros_faixas = []

    print()
    print("=" * 88)
    print("RESUMO POR FAIXA DE PLAUSIBILIDADE UNIVARIADA")
    print("=" * 88)

    print()
    print(
        "As faixas abaixo são apenas DESCRITIVAS."
    )

    print(
        "Não estamos dizendo que 5%, 10% ou 15% são limites metodológicos."
    )

    for faixa in NOMES_FAIXAS:

        parte = grade_ordenada[
            grade_ordenada[
                "Faixa_Plausibilidade_P01_P99"
            ].astype(str)
            ==
            faixa
        ]

        if len(parte) == 0:

            registros_faixas.append(
                {
                    "Faixa":
                        faixa,

                    "N_Condicoes":
                        0,

                    "N_Local_Supera_Globais":
                        0,

                    "Pct_Local_Supera_Globais":
                        np.nan,

                    "Vantagem_Media":
                        np.nan,

                    "Vantagem_Mediana":
                        np.nan,

                    "Vantagem_Maxima":
                        np.nan,

                    "Melhor_Condicao":
                        None,
                }
            )

            continue

        n_supera = int(
            parte[
                "Local_Supera_Todos_Globais"
            ].sum()
        )

        indice_melhor = (
            parte[
                "Vantagem_Local_Absoluta"
            ]
            .idxmax()
        )

        melhor_condicao = (
            parte.loc[
                indice_melhor,
                "Condicao"
            ]
        )

        registro = {
            "Faixa":
                faixa,

            "N_Condicoes":
                len(parte),

            "N_Local_Supera_Globais":
                n_supera,

            "Pct_Local_Supera_Globais":
                (
                    100.0
                    *
                    n_supera
                    /
                    len(parte)
                ),

            "Vantagem_Media":
                float(
                    parte[
                        "Vantagem_Local_Absoluta"
                    ].mean()
                ),

            "Vantagem_Mediana":
                float(
                    parte[
                        "Vantagem_Local_Absoluta"
                    ].median()
                ),

            "Vantagem_Maxima":
                float(
                    parte[
                        "Vantagem_Local_Absoluta"
                    ].max()
                ),

            "Melhor_Condicao":
                melhor_condicao,
        }

        registros_faixas.append(
            registro
        )

        print()
        print(
            f"Faixa: {faixa}"
        )

        print(
            f"Condições: {len(parte)}"
        )

        print(
            "Local supera todos os globais: "
            f"{n_supera} / {len(parte)} "
            f"("
            f"{100.0 * n_supera / len(parte):.2f}%"
            f")"
        )

        print(
            "Vantagem média: "
            f"{registro['Vantagem_Media']:.4f}"
        )

        print(
            "Vantagem mediana: "
            f"{registro['Vantagem_Mediana']:.4f}"
        )

        print(
            "Vantagem máxima: "
            f"{registro['Vantagem_Maxima']:.4f}"
        )

        print(
            "Melhor condição da faixa: "
            f"{melhor_condicao}"
        )

    resumo_faixas = pd.DataFrame(
        registros_faixas
    )

    # ------------------------------------------------------------------------
    # MELHORES CONDIÇÕES EM CADA NÍVEL DE PLAUSIBILIDADE
    # ------------------------------------------------------------------------

    print()
    print("=" * 88)
    print("MELHOR VANTAGEM OBSERVADA SOB DIFERENTES NÍVEIS DESCRITIVOS")
    print("=" * 88)

    limites_descritivos = [
        5.0,
        10.0,
        15.0,
        20.0,
    ]

    registros_limites = []

    for limite in limites_descritivos:

        parte = grade_ordenada[
            grade_ordenada[
                "Pct_Fora_P01_P99"
            ]
            <=
            limite
        ]

        if len(parte) == 0:
            continue

        indice = (
            parte[
                "Vantagem_Local_Absoluta"
            ]
            .idxmax()
        )

        melhor = parte.loc[
            indice
        ]

        registro = {
            "Limite_Descritivo_Pct_Fora_P01_P99":
                limite,

            "Condicao":
                melhor[
                    "Condicao"
                ],

            "m":
                int(
                    melhor[
                        "m"
                    ]
                ),

            "delta_latente":
                float(
                    melhor[
                        "delta_latente"
                    ]
                ),

            "Pct_Fora_P01_P99":
                float(
                    melhor[
                        "Pct_Fora_P01_P99"
                    ]
                ),

            "Melhor_Global":
                melhor[
                    "Melhor_Global"
                ],

            "Recall_Melhor_Global_FPR1":
                float(
                    melhor[
                        "Recall_Melhor_Global_FPR1"
                    ]
                ),

            "Melhor_Local":
                melhor[
                    "Melhor_Local"
                ],

            "Recall_Melhor_Local_FPR1":
                float(
                    melhor[
                        "Recall_Melhor_Local_FPR1"
                    ]
                ),

            "Vantagem_Local_Absoluta":
                float(
                    melhor[
                        "Vantagem_Local_Absoluta"
                    ]
                ),
        }

        registros_limites.append(
            registro
        )

        print()
        print(
            f"Até {limite:.0f}% fora de P1-P99:"
        )

        print(
            f"  condição: "
            f"{registro['Condicao']}"
        )

        print(
            f"  m={registro['m']} | "
            f"delta={registro['delta_latente']:.2f}"
        )

        print(
            f"  fora P1-P99="
            f"{registro['Pct_Fora_P01_P99']:.4f}%"
        )

        print(
            f"  melhor global="
            f"{registro['Recall_Melhor_Global_FPR1']:.4f} "
            f"({registro['Melhor_Global']})"
        )

        print(
            f"  melhor local="
            f"{registro['Recall_Melhor_Local_FPR1']:.4f} "
            f"({registro['Melhor_Local']})"
        )

        print(
            f"  vantagem="
            f"{registro['Vantagem_Local_Absoluta']:.4f}"
        )

    resumo_limites = pd.DataFrame(
        registros_limites
    )

    # ------------------------------------------------------------------------
    # MATRIZ DA VANTAGEM LOCAL
    # ------------------------------------------------------------------------

    matriz_vantagem = grade_ordenada.pivot(
        index="m",
        columns="delta_latente",
        values="Vantagem_Local_Absoluta"
    )

    print()
    print("=" * 88)
    print("SUPERFÍCIE — VANTAGEM LOCAL ABSOLUTA")
    print("=" * 88)

    print()
    print(
        matriz_vantagem.round(
            4
        ).to_string()
    )

    # ------------------------------------------------------------------------
    # MATRIZ BINÁRIA DA REGIÃO POSITIVA
    # ------------------------------------------------------------------------

    matriz_supera = grade_ordenada.pivot(
        index="m",
        columns="delta_latente",
        values="Local_Supera_Todos_Globais"
    )

    matriz_supera = (
        matriz_supera
        .astype(int)
    )

    print()
    print("=" * 88)
    print("REGIÃO EM QUE O LOCAL SUPERA TODOS OS GLOBAIS")
    print("=" * 88)

    print()
    print(
        "1 = sim | 0 = não"
    )

    print()
    print(
        matriz_supera.to_string()
    )

    # ------------------------------------------------------------------------
    # CONTAGEM GLOBAL
    # ------------------------------------------------------------------------

    n_positivas = int(
        grade_ordenada[
            "Local_Supera_Todos_Globais"
        ].sum()
    )

    print()
    print("=" * 88)
    print("RESUMO FINAL")
    print("=" * 88)

    print(
        f"Condições totais: {len(grade_ordenada)}"
    )

    print(
        "Condições com vantagem local sobre "
        f"todos os globais: {n_positivas}"
    )

    print(
        "Percentual da grade com vantagem local: "
        f"{100.0 * n_positivas / len(grade_ordenada):.2f}%"
    )

    # ------------------------------------------------------------------------
    # SALVAMENTO
    # ------------------------------------------------------------------------

    tabela_principal.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "todas_condicoes.csv"
        ),
        index=False
    )

    resumo_faixas.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_faixas_plausibilidade.csv"
        ),
        index=False
    )

    resumo_limites.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "melhores_por_limite_descritivo.csv"
        ),
        index=False
    )

    matriz_vantagem.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "matriz_vantagem_local.csv"
        )
    )

    matriz_supera.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "matriz_regiao_positiva.csv"
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
        "IMPORTANTE:"
    )

    print(
        "Nenhuma faixa de plausibilidade foi definida "
        "como critério de aceitação."
    )

    print(
        "O objetivo desta etapa é somente visualizar "
        "a fronteira observada na grade exploratória."
    )


if __name__ == "__main__":
    main()