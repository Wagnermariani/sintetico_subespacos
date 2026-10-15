import os
import math

import numpy as np
import pandas as pd


# ============================================================================
# AUDITORIA DA MARGEM DE DOMINÂNCIA — ENERGIA V3
#
# Objetivo:
# quantificar o tamanho das pequenas violações que fizeram a regra de
# dominância exata da V3 classificar cenários como "indefinido".
#
# IMPORTANTE:
# - este script NÃO cria ainda uma nova regra final;
# - NÃO usa ataques, P01/P02, m, delta ou rótulos de teste;
# - calcula apenas grandezas observadas nas CDFs normais;
# - inclui uma banda DKW de referência sob hipótese i.i.d., apenas como escala
#   estatística. Como o gerador possui dependência temporal, essa banda NÃO
#   deve ser adotada automaticamente como critério inferencial final.
# ============================================================================


PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_V3_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "diagnostico_energia_normal_only_v3",
    "diagnostico_energia_v3_por_grupo.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_margem_dominancia_energia_v3",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "margem_dominancia_energia_v3_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "margem_dominancia_energia_v3_por_cenario.csv",
)


ALPHA_REFERENCIA = 0.05
EPS = 1e-15


def tolerancia_dkw_diferenca_duas_cdfs(
    n,
    alpha=0.05,
):
    """
    Escala de referência conservadora para a diferença entre duas CDFs
    empíricas, usando DKW em cada marginal + união de eventos.

    Para cada marginal:
        P(sup |F_n - F| > eps) <= 2 exp(-2 n eps^2)

    Reservando alpha/2 para cada marginal:
        eps = sqrt(log(4/alpha)/(2n))

    Logo uma banda de referência para a diferença entre duas CDFs é:
        2 * eps

    A independência ENTRE q_global e q_local não é necessária para a união
    das duas bandas marginais. Porém, a interpretação probabilística DKW
    pressupõe amostras independentes dentro de cada sequência temporal.
    Como o gerador sintético possui dependência temporal, este valor é usado
    SOMENTE como escala de referência, não como regra final da tese.
    """

    if n <= 0:
        raise ValueError(
            "n deve ser positivo."
        )

    eps_marginal = math.sqrt(
        math.log(
            4.0
            /
            alpha
        )
        /
        (
            2.0
            *
            n
        )
    )

    return (
        eps_marginal,
        2.0 * eps_marginal,
    )


def main():

    print("=" * 124)
    print("AUDITORIA DA MARGEM DE DOMINÂNCIA — ENERGIA V3")
    print("=" * 124)

    print()
    print(
        "Nenhum ataque é usado. "
        "A análise parte apenas das CDFs normais já calculadas na V3."
    )

    df = pd.read_csv(
        ARQUIVO_V3_GRUPOS
    )

    colunas_obrigatorias = {
        "Cenario",
        "Grupo",
        "N_Estados_Elegiveis",
        "CDF_Diferenca_Min_Flocal_menos_Fglobal",
        "CDF_Diferenca_Max_Flocal_menos_Fglobal",
        "Classificacao_Dominancia",
    }

    faltantes = (
        colunas_obrigatorias
        -
        set(
            df.columns
        )
    )

    if faltantes:

        raise RuntimeError(
            "Colunas ausentes no arquivo V3: "
            +
            ", ".join(
                sorted(
                    faltantes
                )
            )
        )

    if len(
        df
    ) == 0:

        raise RuntimeError(
            "Arquivo V3 por grupo está vazio."
        )

    # ------------------------------------------------------------------------
    # Grandezas observadas
    #
    # diff(q) = F_local(q) - F_global(q)
    #
    # Sup_Local:
    #   maior vantagem observada da CDF local sobre a global.
    #
    # Sup_Global:
    #   maior vantagem observada da CDF global sobre a local.
    #
    # Se Sup_Global = 0, há dominância local exata.
    # Se Sup_Local  = 0, há dominância global exata.
    # Se ambos > 0, as CDFs cruzam.
    # ------------------------------------------------------------------------

    df[
        "Sup_Local"
    ] = np.maximum(
        0.0,
        df[
            "CDF_Diferenca_Max_Flocal_menos_Fglobal"
        ].to_numpy(
            dtype=float
        ),
    )

    df[
        "Sup_Global"
    ] = np.maximum(
        0.0,
        -
        df[
            "CDF_Diferenca_Min_Flocal_menos_Fglobal"
        ].to_numpy(
            dtype=float
        ),
    )

    eps_marginais = []
    tolerancias = []

    for n in df[
        "N_Estados_Elegiveis"
    ].astype(
        int
    ):

        eps_marginal, tol_diff = (
            tolerancia_dkw_diferenca_duas_cdfs(
                n,
                ALPHA_REFERENCIA,
            )
        )

        eps_marginais.append(
            eps_marginal
        )

        tolerancias.append(
            tol_diff
        )

    df[
        "DKW_Epsilon_Marginal_Referencia_95"
    ] = eps_marginais

    df[
        "DKW_Tolerancia_Diferenca_Referencia_95"
    ] = tolerancias

    # Quantas "degraus" de 1/n representam as vantagens observadas.
    df[
        "Sup_Local_Em_Passos_1_sobre_n"
    ] = (
        df[
            "Sup_Local"
        ]
        *
        df[
            "N_Estados_Elegiveis"
        ]
    )

    df[
        "Sup_Global_Em_Passos_1_sobre_n"
    ] = (
        df[
            "Sup_Global"
        ]
        *
        df[
            "N_Estados_Elegiveis"
        ]
    )

    # Relação com a banda DKW de referência.
    df[
        "Razao_Sup_Local_sobre_Tolerancia_DKW"
    ] = (
        df[
            "Sup_Local"
        ]
        /
        df[
            "DKW_Tolerancia_Diferenca_Referencia_95"
        ]
    )

    df[
        "Razao_Sup_Global_sobre_Tolerancia_DKW"
    ] = (
        df[
            "Sup_Global"
        ]
        /
        df[
            "DKW_Tolerancia_Diferenca_Referencia_95"
        ]
    )

    # Classificação apenas DESCRITIVA em relação à escala DKW.
    # Não é a classificação final do diagnóstico.
    cond_local_forte = (
        (
            df[
                "Sup_Local"
            ]
            >
            df[
                "DKW_Tolerancia_Diferenca_Referencia_95"
            ]
        )
        &
        (
            df[
                "Sup_Global"
            ]
            <=
            df[
                "DKW_Tolerancia_Diferenca_Referencia_95"
            ]
        )
    )

    cond_global_forte = (
        (
            df[
                "Sup_Global"
            ]
            >
            df[
                "DKW_Tolerancia_Diferenca_Referencia_95"
            ]
        )
        &
        (
            df[
                "Sup_Local"
            ]
            <=
            df[
                "DKW_Tolerancia_Diferenca_Referencia_95"
            ]
        )
    )

    cond_indist = (
        (
            df[
                "Sup_Local"
            ]
            <=
            df[
                "DKW_Tolerancia_Diferenca_Referencia_95"
            ]
        )
        &
        (
            df[
                "Sup_Global"
            ]
            <=
            df[
                "DKW_Tolerancia_Diferenca_Referencia_95"
            ]
        )
    )

    classe_ref = np.full(
        len(
            df
        ),
        "bidirecional_acima_tolerancia",
        dtype=object,
    )

    classe_ref[
        cond_local_forte
    ] = "local_forte_na_escala_DKW"

    classe_ref[
        cond_global_forte
    ] = "global_forte_na_escala_DKW"

    classe_ref[
        cond_indist
    ] = "indistinguivel_na_escala_DKW"

    df[
        "Classe_Descritiva_Escala_DKW"
    ] = classe_ref

    df.to_csv(
        ARQUIVO_GRUPOS,
        index=False,
    )

    # ------------------------------------------------------------------------
    # RESUMO POR CENÁRIO
    # ------------------------------------------------------------------------

    linhas_cenarios = []

    for cenario, parte in df.groupby(
        "Cenario",
        sort=True,
    ):

        n = int(
            parte[
                "N_Estados_Elegiveis"
            ].iloc[
                0
            ]
        )

        if not (
            parte[
                "N_Estados_Elegiveis"
            ]
            ==
            n
        ).all():

            raise RuntimeError(
                f"{cenario}: N_Estados_Elegiveis varia entre grupos."
            )

        tol = float(
            parte[
                "DKW_Tolerancia_Diferenca_Referencia_95"
            ].iloc[
                0
            ]
        )

        classes_ref = (
            parte[
                "Classe_Descritiva_Escala_DKW"
            ]
            .value_counts()
            .to_dict()
        )

        linhas_cenarios.append(
            {
                "Cenario":
                    cenario,

                "N_Grupos":
                    len(
                        parte
                    ),

                "N_Estados_Elegiveis":
                    n,

                "DKW_Tolerancia_Diferenca_Referencia_95":
                    tol,

                # pior violação contra uma hipótese de dominância local
                "Pior_Sup_Global_Entre_Grupos":
                    float(
                        parte[
                            "Sup_Global"
                        ].max()
                    ),

                # grupo no qual a vantagem local é mais fraca
                "Menor_Sup_Local_Entre_Grupos":
                    float(
                        parte[
                            "Sup_Local"
                        ].min()
                    ),

                # maior vantagem local observada em qualquer grupo
                "Maior_Sup_Local_Entre_Grupos":
                    float(
                        parte[
                            "Sup_Local"
                        ].max()
                    ),

                "Pior_Razao_Sup_Global_sobre_DKW":
                    float(
                        parte[
                            "Razao_Sup_Global_sobre_Tolerancia_DKW"
                        ].max()
                    ),

                "Menor_Razao_Sup_Local_sobre_DKW":
                    float(
                        parte[
                            "Razao_Sup_Local_sobre_Tolerancia_DKW"
                        ].min()
                    ),

                "N_Local_Forte_Escala_DKW":
                    int(
                        classes_ref.get(
                            "local_forte_na_escala_DKW",
                            0,
                        )
                    ),

                "N_Global_Forte_Escala_DKW":
                    int(
                        classes_ref.get(
                            "global_forte_na_escala_DKW",
                            0,
                        )
                    ),

                "N_Indistinguivel_Escala_DKW":
                    int(
                        classes_ref.get(
                            "indistinguivel_na_escala_DKW",
                            0,
                        )
                    ),

                "N_Bidirecional_Acima_DKW":
                    int(
                        classes_ref.get(
                            "bidirecional_acima_tolerancia",
                            0,
                        )
                    ),
            }
        )

    cenarios = pd.DataFrame(
        linhas_cenarios
    )

    cenarios.to_csv(
        ARQUIVO_CENARIOS,
        index=False,
    )

    # ------------------------------------------------------------------------
    # TERMINAL
    # ------------------------------------------------------------------------

    print()
    print("=" * 124)
    print("RESUMO POR CENÁRIO")
    print("=" * 124)
    print()

    colunas_resumo = [
        "Cenario",
        "N_Grupos",
        "N_Estados_Elegiveis",
        "DKW_Tolerancia_Diferenca_Referencia_95",
        "Pior_Sup_Global_Entre_Grupos",
        "Menor_Sup_Local_Entre_Grupos",
        "Maior_Sup_Local_Entre_Grupos",
        "N_Local_Forte_Escala_DKW",
        "N_Global_Forte_Escala_DKW",
        "N_Indistinguivel_Escala_DKW",
        "N_Bidirecional_Acima_DKW",
    ]

    print(
        cenarios[
            colunas_resumo
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 124)
    print("LEITURA DA AUDITORIA")
    print("=" * 124)

    print()
    print(
        "Sup_Local  = max_q [F_local(q) - F_global(q)]."
    )

    print(
        "Sup_Global = max_q [F_global(q) - F_local(q)]."
    )

    print(
        "A V3 exata exige Sup_Global=0 para dominância local "
        "e Sup_Local=0 para dominância global."
    )

    print()
    print(
        "A coluna DKW é SOMENTE uma escala de referência sob hipótese i.i.d."
    )

    print(
        "O gerador possui dependência temporal; portanto, NÃO transformar "
        "automaticamente a classe descritiva DKW em regra final da tese."
    )

    print()
    print(
        "Nenhum ataque, P01/P02, m, delta, direção ou rótulo de teste foi usado."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)


if __name__ == "__main__":
    main()
