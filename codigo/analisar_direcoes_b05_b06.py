import os

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf

from gerar_normal_estrutural import GANHO_TANH


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

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_confirmatorios_base.csv",
)

ARQUIVO_ORDEM = os.path.join(
    PROJETO,
    "dados",
    "ordem_features_confirmatorias.csv",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

ARQUIVO_MAPA_ORACULO = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv",
)

ARQUIVO_DIAGNOSTICO = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "diagnostico_oraculo",
    "diagnostico_por_grupo.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "analise_direcoes",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_DETALHE = os.path.join(
    DIRETORIO_SAIDA,
    "direcoes_b05_b06_detalhe.csv",
)

ARQUIVO_RESUMO = os.path.join(
    DIRETORIO_SAIDA,
    "direcoes_b05_b06_resumo.csv",
)


CENARIOS = [
    "B05",
    "B06",
]

PROBES = [
    {
        "Probe": "P01",
        "m": 8,
        "delta": 1.25,
    },
    {
        "Probe": "P02",
        "m": 8,
        "delta": 1.50,
    },
]


# ============================================================================
# NORMALIZAÇÃO FÍSICA
# ============================================================================

def normalizar_fisicamente(
    df,
    features,
    parametros,
):

    X = df[
        features
    ].copy()

    for feature in features:

        if feature not in parametros:
            continue

        minimo = float(
            parametros[
                feature
            ][
                "Min_Fisico"
            ]
        )

        maximo = float(
            parametros[
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
# MAIN
# ============================================================================

def main():

    print("=" * 108)
    print("ANÁLISE PÓS-HOC DAS DIREÇÕES REAIS — B05 E B06")
    print("=" * 108)

    print()
    print(
        "Objetivo: comparar r(v) das perturbações efetivamente aplicadas "
        "com o Burden Ratio calculado apenas no normal."
    )

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    ordem = pd.read_csv(
        ARQUIVO_ORDEM
    )

    mapa = pd.read_csv(
        ARQUIVO_MAPA_ORACULO
    )

    diagnostico = pd.read_csv(
        ARQUIVO_DIAGNOSTICO
    )

    parametros_df = pd.read_csv(
        ARQUIVO_PARAMETROS
    )

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
    # ORDEM CONGELADA DE FEATURES PARA OS ATAQUES
    # ------------------------------------------------------------------------

    ordem_por_grupo = {}

    for grupo, parte in ordem.groupby(
        "Grupo"
    ):

        parte = parte.sort_values(
            "Posicao"
        )

        ordem_por_grupo[
            str(grupo)
        ] = parte[
            "Feature"
        ].tolist()

    # ------------------------------------------------------------------------
    # MAPA ORÁCULO
    # ------------------------------------------------------------------------

    features = sorted(
        mapa[
            "Feature"
        ].tolist()
    )

    if len(features) != 400:

        raise RuntimeError(
            "Esperadas exatamente 400 features."
        )

    indice_feature = {
        feature: indice
        for indice, feature
        in enumerate(
            features
        )
    }

    features_por_grupo = {}

    for grupo, parte in mapa.groupby(
        "Grupo_Verdadeiro"
    ):

        features_por_grupo[
            str(grupo)
        ] = parte[
            "Feature"
        ].tolist()

    detalhes = []

    # =========================================================================
    # B05 E B06
    # =========================================================================

    for cenario in CENARIOS:

        print()
        print("=" * 108)
        print(cenario)
        print("=" * 108)

        linha_manifesto = manifesto.loc[
            manifesto[
                "Cenario"
            ]
            ==
            cenario
        ]

        if len(
            linha_manifesto
        ) != 1:

            raise RuntimeError(
                f"{cenario}: cenário não encontrado de forma única."
            )

        linha_manifesto = linha_manifesto.iloc[
            0
        ]

        treino = pd.read_csv(
            linha_manifesto[
                "Arquivo_Treino"
            ]
        )

        teste_base = pd.read_csv(
            linha_manifesto[
                "Arquivo_Teste"
            ]
        )

        X_train = normalizar_fisicamente(
            treino,
            features,
            parametros,
        )

        X_test_base = normalizar_fisicamente(
            teste_base,
            features,
            parametros,
        )

        # ====================================================================
        # MODELO GLOBAL
        # ====================================================================

        modelo_global = LedoitWolf(
            store_precision=True,
            assume_centered=False,
        )

        modelo_global.fit(
            X_train
        )

        # ====================================================================
        # MODELOS LOCAIS ORÁCULO
        # ====================================================================

        modelos_locais = {}

        indices_grupos = {}

        for grupo in sorted(
            features_por_grupo
        ):

            features_grupo = features_por_grupo[
                grupo
            ]

            idx_grupo = np.asarray(
                [
                    indice_feature[
                        feature
                    ]
                    for feature in features_grupo
                ],
                dtype=np.int64,
            )

            indices_grupos[
                grupo
            ] = idx_grupo

            modelo_local = LedoitWolf(
                store_precision=True,
                assume_centered=False,
            )

            modelo_local.fit(
                X_train[
                    :,
                    idx_grupo
                ]
            )

            modelos_locais[
                grupo
            ] = modelo_local

        # ====================================================================
        # PROBES
        # ====================================================================

        for probe in PROBES:

            nome_probe = str(
                probe[
                    "Probe"
                ]
            )

            m = int(
                probe[
                    "m"
                ]
            )

            delta = float(
                probe[
                    "delta"
                ]
            )

            print()
            print(
                f"{nome_probe} | m={m} | delta={delta:.2f}"
            )

            # ================================================================
            # EVENTOS
            # ================================================================

            for _, evento in eventos.iterrows():

                nome_evento = str(
                    evento[
                        "Evento"
                    ]
                )

                grupo = str(
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

                features_atacadas = (
                    ordem_por_grupo[
                        grupo
                    ][
                        :m
                    ]
                )

                # ------------------------------------------------------------
                # ÍNDICES DAS FEATURES ATACADAS NO VETOR GLOBAL
                # ------------------------------------------------------------

                idx_atacadas_global = np.asarray(
                    [
                        indice_feature[
                            feature
                        ]
                        for feature in features_atacadas
                    ],
                    dtype=np.int64,
                )

                # ------------------------------------------------------------
                # POSIÇÕES DAS FEATURES ATACADAS DENTRO DO GRUPO LOCAL
                # ------------------------------------------------------------

                features_grupo = features_por_grupo[
                    grupo
                ]

                pos_local = {
                    feature: indice
                    for indice, feature
                    in enumerate(
                        features_grupo
                    )
                }

                idx_atacadas_local = np.asarray(
                    [
                        pos_local[
                            feature
                        ]
                        for feature in features_atacadas
                    ],
                    dtype=np.int64,
                )

                linhas = np.arange(
                    inicio,
                    fim + 1,
                )

                valores_base = X_test_base[
                    np.ix_(
                        linhas,
                        idx_atacadas_global,
                    )
                ]

                # ------------------------------------------------------------
                # MESMA TRANSFORMAÇÃO UTILIZADA NOS PROBES ANTERIORES
                # ------------------------------------------------------------

                eps = 1e-12

                valores_seguros = np.clip(
                    valores_base,
                    eps,
                    1.0 - eps,
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

                V = (
                    valores_atacados
                    -
                    valores_base
                )

                # ------------------------------------------------------------
                # PRECISÕES RESTRITAS ÀS 8 FEATURES EFETIVAMENTE ALTERADAS
                #
                # Como as demais coordenadas de v são zero, usar a submatriz
                # atacada é equivalente a preencher um vetor de 20/400
                # dimensões com zeros nas coordenadas não perturbadas.
                # ------------------------------------------------------------

                P_global = modelo_global.precision_[
                    np.ix_(
                        idx_atacadas_global,
                        idx_atacadas_global,
                    )
                ]

                P_local = modelos_locais[
                    grupo
                ].precision_[
                    np.ix_(
                        idx_atacadas_local,
                        idx_atacadas_local,
                    )
                ]

                # ------------------------------------------------------------
                # BURDEN NORMAL-ONLY DESTE GRUPO
                # ------------------------------------------------------------

                linha_diag = diagnostico.loc[
                    (
                        diagnostico[
                            "Cenario"
                        ]
                        ==
                        cenario
                    )
                    &
                    (
                        diagnostico[
                            "Grupo"
                        ]
                        ==
                        grupo
                    )
                ]

                if len(
                    linha_diag
                ) != 1:

                    raise RuntimeError(
                        f"{cenario} {grupo}: diagnóstico não encontrado."
                    )

                linha_diag = linha_diag.iloc[
                    0
                ]

                burden = float(
                    linha_diag[
                        "Burden_Ratio"
                    ]
                )

                r_max = float(
                    linha_diag[
                        "Razao_Quadratica_Max"
                    ]
                )

                fator_pessimista = float(
                    linha_diag[
                        "Fator_Vantagem_Pessimista"
                    ]
                )

                # ------------------------------------------------------------
                # r(v) PARA CADA SEGUNDO DO EVENTO
                # ------------------------------------------------------------

                for posicao, v in enumerate(
                    V
                ):

                    q_global = float(
                        v
                        @
                        P_global
                        @
                        v
                    )

                    q_local = float(
                        v
                        @
                        P_local
                        @
                        v
                    )

                    if q_local <= 0:

                        raise RuntimeError(
                            f"{cenario} {nome_probe} {nome_evento}: "
                            "q_local não positivo."
                        )

                    r_v = (
                        q_global
                        /
                        q_local
                    )

                    fator_direcao = (
                        burden
                        /
                        r_v
                    )

                    detalhes.append(
                        {
                            "Cenario":
                                cenario,

                            "Probe":
                                nome_probe,

                            "m":
                                m,

                            "delta":
                                delta,

                            "Evento":
                                nome_evento,

                            "Grupo":
                                grupo,

                            "Segundo_Evento":
                                posicao,

                            "Direcao":
                                direcao,

                            "Burden_Ratio":
                                burden,

                            "R_Max":
                                r_max,

                            "Fator_Pessimista":
                                fator_pessimista,

                            "Q_Global_Direcao":
                                q_global,

                            "Q_Local_Direcao":
                                q_local,

                            "R_v":
                                r_v,

                            "Fator_Direcao":
                                fator_direcao,

                            "Condicao_Burden_Maior_Rv":
                                int(
                                    burden
                                    >
                                    r_v
                                ),
                        }
                    )

    # =========================================================================
    # RESULTADOS
    # =========================================================================

    df = pd.DataFrame(
        detalhes
    )

    df.to_csv(
        ARQUIVO_DETALHE,
        index=False,
    )

    resumo = (
        df
        .groupby(
            [
                "Cenario",
                "Probe",
                "m",
                "delta",
            ],
            as_index=False,
        )
        .agg(
            Burden_Mediano=(
                "Burden_Ratio",
                "median",
            ),

            Rv_Min=(
                "R_v",
                "min",
            ),

            Rv_Mediano=(
                "R_v",
                "median",
            ),

            Rv_Max=(
                "R_v",
                "max",
            ),

            Fator_Direcao_Min=(
                "Fator_Direcao",
                "min",
            ),

            Fator_Direcao_Mediano=(
                "Fator_Direcao",
                "median",
            ),

            Fator_Direcao_Max=(
                "Fator_Direcao",
                "max",
            ),

            Fracao_Pontos_Burden_Maior_Rv=(
                "Condicao_Burden_Maior_Rv",
                "mean",
            ),
        )
    )

    resumo.to_csv(
        ARQUIVO_RESUMO,
        index=False,
    )

    print()
    print("=" * 108)
    print("RESUMO — DIREÇÕES EFETIVAMENTE OBSERVADAS")
    print("=" * 108)

    print()

    print(
        resumo.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretação:"
    )

    print(
        "Fator_Direcao > 1 equivale a Burden_Ratio > r(v)."
    )

    print(
        "Esta é uma análise pós-hoc das direções dos probes; "
        "não é um diagnóstico operacional zero-day."
    )

    print()
    print(
        "Arquivos:"
    )

    print(
        ARQUIVO_DETALHE
    )

    print(
        ARQUIVO_RESUMO
    )


if __name__ == "__main__":
    main()