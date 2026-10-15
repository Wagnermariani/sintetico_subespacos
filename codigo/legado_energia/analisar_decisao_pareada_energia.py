import os

import numpy as np
import pandas as pd


# ============================================================================
# AUDITORIA PAREADA DA DECISÃO — ENERGIA — P01/P02
#
# Objetivo:
# explicar por que um cenário pode apresentar F_dec > 1 na maior parte dos
# estados normais e, ainda assim, mostrar recall bruto global maior.
#
# O programa NÃO recalibra limiares nem modifica ataques.
# Ele apenas analisa, ponto a ponto, o arquivo já produzido pela auditoria
# de incrementos de Energia.
# ============================================================================


PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_PONTOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_incrementos_energia",
    "incrementos_energia_por_ponto.csv",
)

ARQUIVO_DIAGNOSTICO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "diagnostico_energia_normal_only",
    "diagnostico_energia_por_grupo.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_decisao_pareada_energia",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_RESUMO = os.path.join(
    DIRETORIO_SAIDA,
    "resumo_decisao_pareada_energia.csv",
)

ARQUIVO_CRUZAMENTOS = os.path.join(
    DIRETORIO_SAIDA,
    "cruzamentos_decisao_pareada_energia.csv",
)

EPS = 1e-12
D_GLOBAL = 400


def proporcao(mask):
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0:
        return np.nan

    return float(np.mean(mask))


def main():

    print("=" * 118)
    print("AUDITORIA PAREADA DA DECISÃO — ENERGIA — P01/P02")
    print("=" * 118)

    pontos = pd.read_csv(
        ARQUIVO_PONTOS
    )

    diag = pd.read_csv(
        ARQUIVO_DIAGNOSTICO_GRUPOS
    )

    # ------------------------------------------------------------------------
    # Tamanho do grupo descoberto
    # ------------------------------------------------------------------------

    mapa_tamanho = (
        diag[
            [
                "Cenario",
                "Grupo",
                "Tamanho_Grupo",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "Grupo":
                    "Grupo_Descoberto_Alvo",
            }
        )
    )

    pontos = pontos.merge(
        mapa_tamanho,
        on=[
            "Cenario",
            "Grupo_Descoberto_Alvo",
        ],
        how="left",
        validate="many_to_one",
    )

    if pontos[
        "Tamanho_Grupo"
    ].isna().any():

        faltantes = (
            pontos.loc[
                pontos[
                    "Tamanho_Grupo"
                ].isna(),
                [
                    "Cenario",
                    "Grupo_Descoberto_Alvo",
                ],
            ]
            .drop_duplicates()
        )

        raise RuntimeError(
            "Não foi possível localizar o tamanho de alguns grupos:\n"
            +
            faltantes.to_string(
                index=False
            )
        )

    if len(
        pontos
    ) != 48000:

        raise RuntimeError(
            f"Esperados 48000 pontos; obtidos {len(pontos)}."
        )

    # ------------------------------------------------------------------------
    # Quantidade total de evidência de Energia adicionada.
    #
    # Quando h(A)=1:
    #
    #   q(t) = D * DeltaE_global
    #        = d_k * DeltaE_local_alvo
    #
    # ------------------------------------------------------------------------

    pontos[
        "Q_Real_Global"
    ] = (
        D_GLOBAL
        *
        pontos[
            "Delta_E_Global"
        ]
    )

    pontos[
        "Q_Real_Local"
    ] = (
        pontos[
            "Tamanho_Grupo"
        ]
        *
        pontos[
            "Delta_E_Local_Alvo"
        ]
    )

    erro_q = np.max(
        np.abs(
            pontos[
                "Q_Real_Global"
            ]
            -
            pontos[
                "Q_Real_Local"
            ]
        )
    )

    print(
        f"Máximo erro |q_global - q_local|: {erro_q:.6e}"
    )

    # ------------------------------------------------------------------------
    # Estado inicial e transições
    # ------------------------------------------------------------------------

    g_base = (
        pontos[
            "Alarme_Global_Base"
        ]
        .astype(bool)
    )

    g_att = (
        pontos[
            "Alarme_Global_Perturbado"
        ]
        .astype(bool)
    )

    l_base = (
        pontos[
            "Alarme_Local_Base"
        ]
        .astype(bool)
    )

    l_att = (
        pontos[
            "Alarme_Local_Perturbado"
        ]
        .astype(bool)
    )

    pontos[
        "Novo_Global"
    ] = (
        (~g_base)
        &
        g_att
    ).astype(int)

    pontos[
        "Novo_Local"
    ] = (
        (~l_base)
        &
        l_att
    ).astype(int)

    pontos[
        "Perda_Global"
    ] = (
        g_base
        &
        (~g_att)
    ).astype(int)

    pontos[
        "Perda_Local"
    ] = (
        l_base
        &
        (~l_att)
    ).astype(int)

    # ------------------------------------------------------------------------
    # Q necessário para cruzar cada limiar no MESMO estado-base.
    #
    # Para a comparação da transição causada por evidência positiva,
    # consideramos pontos em que AMBAS as decisões estavam abaixo do limiar.
    # ------------------------------------------------------------------------

    ambos_negativos = (
        (~g_base)
        &
        (~l_base)
    )

    pontos[
        "Ambos_Base_Negativos"
    ] = ambos_negativos.astype(
        int
    )

    pontos[
        "Q_Necessario_Global"
    ] = np.where(
        ambos_negativos,
        D_GLOBAL
        *
        (
            pontos[
                "Tau_Global"
            ]
            -
            pontos[
                "Score_Global_Base"
            ]
        ),
        np.nan,
    )

    pontos[
        "Q_Necessario_Local"
    ] = np.where(
        ambos_negativos,
        pontos[
            "Tamanho_Grupo"
        ]
        *
        (
            pontos[
                "Tau_Local"
            ]
            -
            pontos[
                "Score_Local_Alvo_Base"
            ]
        ),
        np.nan,
    )

    validos_q = (
        ambos_negativos
        &
        (
            pontos[
                "Q_Necessario_Global"
            ]
            >
            EPS
        )
        &
        (
            pontos[
                "Q_Necessario_Local"
            ]
            >
            EPS
        )
    )

    pontos[
        "F_Dec_Test"
    ] = np.nan

    pontos.loc[
        validos_q,
        "F_Dec_Test",
    ] = (
        pontos.loc[
            validos_q,
            "Q_Necessario_Global",
        ]
        /
        pontos.loc[
            validos_q,
            "Q_Necessario_Local",
        ]
    )

    pontos[
        "Local_Requer_Menos_Evidencia"
    ] = (
        pontos[
            "F_Dec_Test"
        ]
        >
        1.0
    ).astype(int)

    # ------------------------------------------------------------------------
    # Para q positivo, a comparação acima deve prever exatamente qual decisão
    # cruza primeiro no estado-base.
    # ------------------------------------------------------------------------

    q_positivo = (
        pontos[
            "Q_Real_Global"
        ]
        >
        EPS
    )

    universo_positivo = (
        validos_q
        &
        q_positivo
    )

    pontos[
        "Q_Suficiente_Global"
    ] = (
        universo_positivo
        &
        (
            pontos[
                "Q_Real_Global"
            ]
            >
            pontos[
                "Q_Necessario_Global"
            ]
        )
    ).astype(int)

    pontos[
        "Q_Suficiente_Local"
    ] = (
        universo_positivo
        &
        (
            pontos[
                "Q_Real_Global"
            ]
            >
            pontos[
                "Q_Necessario_Local"
            ]
        )
    ).astype(int)

    # ------------------------------------------------------------------------
    # Resumo
    # ------------------------------------------------------------------------

    linhas_resumo = []
    linhas_cruzamentos = []

    for (
        cenario,
        probe
    ), parte in pontos.groupby(
        [
            "Cenario",
            "Probe",
        ],
        sort=True,
    ):

        n = len(
            parte
        )

        if n != 2000:

            raise RuntimeError(
                f"{cenario} {probe}: esperados 2000 pontos, obtidos {n}."
            )

        gb = parte[
            "Alarme_Global_Base"
        ].astype(bool)

        ga = parte[
            "Alarme_Global_Perturbado"
        ].astype(bool)

        lb = parte[
            "Alarme_Local_Base"
        ].astype(bool)

        la = parte[
            "Alarme_Local_Perturbado"
        ].astype(bool)

        novo_g = (
            (~gb)
            &
            ga
        )

        novo_l = (
            (~lb)
            &
            la
        )

        perda_g = (
            gb
            &
            (~ga)
        )

        perda_l = (
            lb
            &
            (~la)
        )

        neg_g = ~gb
        neg_l = ~lb

        ambos_neg = (
            (~gb)
            &
            (~lb)
        )

        qpos = (
            parte[
                "Q_Real_Global"
            ]
            >
            EPS
        )

        universo = (
            ambos_neg
            &
            qpos
            &
            parte[
                "F_Dec_Test"
            ].notna()
        )

        fdec = parte.loc[
            universo,
            "F_Dec_Test",
        ]

        # --------------------------------------------------------------
        # Quatro tipos de cruzamento entre pontos inicialmente negativos
        # em ambas as decisões.
        # --------------------------------------------------------------

        novo_g_ambos = (
            universo
            &
            ga
        )

        novo_l_ambos = (
            universo
            &
            la
        )

        ambos_cruzam = (
            novo_g_ambos
            &
            novo_l_ambos
        )

        so_global = (
            novo_g_ambos
            &
            (~novo_l_ambos)
        )

        so_local = (
            novo_l_ambos
            &
            (~novo_g_ambos)
        )

        nenhum = (
            universo
            &
            (~novo_g_ambos)
            &
            (~novo_l_ambos)
        )

        # --------------------------------------------------------------
        # Checagem lógica:
        # global-only sob q positivo deve ocorrer em região F_dec < 1;
        # local-only sob q positivo deve ocorrer em região F_dec > 1.
        # --------------------------------------------------------------

        violacao_global_only = int(
            np.sum(
                so_global
                &
                (
                    parte[
                        "F_Dec_Test"
                    ]
                    >=
                    1.0
                )
            )
        )

        violacao_local_only = int(
            np.sum(
                so_local
                &
                (
                    parte[
                        "F_Dec_Test"
                    ]
                    <=
                    1.0
                )
            )
        )

        linhas_resumo.append(
            {
                "Cenario":
                    cenario,

                "Probe":
                    probe,

                "N_Pontos":
                    n,

                "Alarme_Base_Global":
                    proporcao(
                        gb
                    ),

                "Alarme_Base_Local":
                    proporcao(
                        lb
                    ),

                "Recall_Bruto_Global":
                    proporcao(
                        ga
                    ),

                "Recall_Bruto_Local":
                    proporcao(
                        la
                    ),

                "Novos_Alarmes_Global":
                    int(
                        np.sum(
                            novo_g
                        )
                    ),

                "Novos_Alarmes_Local":
                    int(
                        np.sum(
                            novo_l
                        )
                    ),

                "Perdas_Alarmes_Global":
                    int(
                        np.sum(
                            perda_g
                        )
                    ),

                "Perdas_Alarmes_Local":
                    int(
                        np.sum(
                            perda_l
                        )
                    ),

                "Taxa_Novo_Alarme_Global_Cond_Base_Neg":
                    (
                        float(
                            np.sum(
                                novo_g
                            )
                            /
                            np.sum(
                                neg_g
                            )
                        )
                        if np.sum(
                            neg_g
                        )
                        >
                        0
                        else
                        np.nan
                    ),

                "Taxa_Novo_Alarme_Local_Cond_Base_Neg":
                    (
                        float(
                            np.sum(
                                novo_l
                            )
                            /
                            np.sum(
                                neg_l
                            )
                        )
                        if np.sum(
                            neg_l
                        )
                        >
                        0
                        else
                        np.nan
                    ),

                "Efeito_Liquido_Global":
                    float(
                        (
                            np.sum(
                                novo_g
                            )
                            -
                            np.sum(
                                perda_g
                            )
                        )
                        /
                        n
                    ),

                "Efeito_Liquido_Local":
                    float(
                        (
                            np.sum(
                                novo_l
                            )
                            -
                            np.sum(
                                perda_l
                            )
                        )
                        /
                        n
                    ),

                "N_Ambos_Base_Negativos_Q_Positivo":
                    int(
                        np.sum(
                            universo
                        )
                    ),

                "Frac_F_Dec_Test_Maior_1":
                    (
                        float(
                            np.mean(
                                fdec
                                >
                                1.0
                            )
                        )
                        if len(
                            fdec
                        )
                        else
                        np.nan
                    ),

                "F_Dec_Test_Mediana":
                    (
                        float(
                            np.median(
                                fdec
                            )
                        )
                        if len(
                            fdec
                        )
                        else
                        np.nan
                    ),

                "N_Cruzam_Ambos":
                    int(
                        np.sum(
                            ambos_cruzam
                        )
                    ),

                "N_Cruza_So_Global":
                    int(
                        np.sum(
                            so_global
                        )
                    ),

                "N_Cruza_So_Local":
                    int(
                        np.sum(
                            so_local
                        )
                    ),

                "N_Nenhum_Cruza":
                    int(
                        np.sum(
                            nenhum
                        )
                    ),

                "Violacao_GlobalOnly_com_Fdec_ge_1":
                    violacao_global_only,

                "Violacao_LocalOnly_com_Fdec_le_1":
                    violacao_local_only,
            }
        )

        for nome, mascara in [
            (
                "ambos",
                ambos_cruzam,
            ),
            (
                "somente_global",
                so_global,
            ),
            (
                "somente_local",
                so_local,
            ),
            (
                "nenhum",
                nenhum,
            ),
        ]:

            sub = parte.loc[
                mascara
            ]

            linhas_cruzamentos.append(
                {
                    "Cenario":
                        cenario,

                    "Probe":
                        probe,

                    "Tipo":
                        nome,

                    "N":
                        len(
                            sub
                        ),

                    "F_Dec_Mediana":
                        (
                            float(
                                sub[
                                    "F_Dec_Test"
                                ].median()
                            )
                            if len(
                                sub
                            )
                            else
                            np.nan
                        ),

                    "F_Dec_Min":
                        (
                            float(
                                sub[
                                    "F_Dec_Test"
                                ].min()
                            )
                            if len(
                                sub
                            )
                            else
                            np.nan
                        ),

                    "F_Dec_Max":
                        (
                            float(
                                sub[
                                    "F_Dec_Test"
                                ].max()
                            )
                            if len(
                                sub
                            )
                            else
                            np.nan
                        ),

                    "Q_Real_Mediana":
                        (
                            float(
                                sub[
                                    "Q_Real_Global"
                                ].median()
                            )
                            if len(
                                sub
                            )
                            else
                            np.nan
                        ),

                    "Q_Nec_Global_Mediana":
                        (
                            float(
                                sub[
                                    "Q_Necessario_Global"
                                ].median()
                            )
                            if len(
                                sub
                            )
                            else
                            np.nan
                        ),

                    "Q_Nec_Local_Mediana":
                        (
                            float(
                                sub[
                                    "Q_Necessario_Local"
                                ].median()
                            )
                            if len(
                                sub
                            )
                            else
                            np.nan
                        ),
                }
            )

    resumo = pd.DataFrame(
        linhas_resumo
    )

    cruzamentos = pd.DataFrame(
        linhas_cruzamentos
    )

    resumo.to_csv(
        ARQUIVO_RESUMO,
        index=False,
    )

    cruzamentos.to_csv(
        ARQUIVO_CRUZAMENTOS,
        index=False,
    )

    print()
    print("=" * 118)
    print("RESUMO")
    print("=" * 118)

    colunas = [
        "Cenario",
        "Probe",
        "Alarme_Base_Global",
        "Alarme_Base_Local",
        "Recall_Bruto_Global",
        "Recall_Bruto_Local",
        "Novos_Alarmes_Global",
        "Novos_Alarmes_Local",
        "Perdas_Alarmes_Global",
        "Perdas_Alarmes_Local",
        "Frac_F_Dec_Test_Maior_1",
        "N_Cruza_So_Global",
        "N_Cruza_So_Local",
        "Violacao_GlobalOnly_com_Fdec_ge_1",
        "Violacao_LocalOnly_com_Fdec_le_1",
    ]

    print(
        resumo[
            colunas
        ].to_string(
            index=False
        )
    )

    print()
    print("Resultados:")
    print(ARQUIVO_RESUMO)
    print(ARQUIVO_CRUZAMENTOS)


if __name__ == "__main__":
    main()
