import os

import numpy as np
import pandas as pd


# ============================================================================
# BOOTSTRAP EM BLOCOS — FAVORABILIDADE NORMAL-ONLY DA ENERGIA
#
# Esta rotina NÃO usa ataques.
#
# Para cada grupo e cada estado normal elegível:
#
#   Q_G(t)   = D   * [tau_G - s_G(t)]
#   Q_L,k(t) = d_k * [tau_L - s_k(t)]
#
# Duas estatísticas pareadas são avaliadas:
#
#   p_k = P[Q_L,k < Q_G]
#
# e
#
#   ell_k = E[ log(Q_G / Q_L,k) ]
#
# p_k > 0.5 e ell_k > 0 apontam para menor evidência necessária no local.
# p_k < 0.5 e ell_k < 0 apontam para menor evidência necessária no global.
#
# A incerteza é estimada por bootstrap circular em blocos sobre a sequência
# temporal completa. O comprimento do bloco é obtido SOMENTE do normal:
# usa-se o maior lag de decorrelação |ACF|<0.05 já medido na auditoria
# temporal de cada cenário.
#
# Classificação por grupo:
#
#   local_suportado:
#       limite inferior 95% de p_k > 0.5
#       E limite inferior 95% de ell_k > 0
#
#   global_suportado:
#       limite superior 95% de p_k < 0.5
#       E limite superior 95% de ell_k < 0
#
#   equivalente:
#       Q_G e Q_L coincidem numericamente nos estados elegíveis
#
#   indefinido:
#       demais casos
#
# Classificação de cenário:
#
#   local_muito_provavel:
#       todos os grupos local_suportado ou equivalente,
#       com pelo menos um local_suportado
#
#   local_muito_improvavel:
#       todos os grupos global_suportado ou equivalente,
#       com pelo menos um global_suportado
#
#   equivalente:
#       todos os grupos equivalentes
#
#   indefinido:
#       demais casos
#
# Esta rotina é uma etapa de auditoria/metodologia. A decisão de incorporá-la
# como diagnóstico final da tese deve ser tomada APÓS examinar os resultados.
# ============================================================================


PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_MANIFESTO = os.path.join(
    PROJETO,
    "dados",
    "fronteira_estrutural",
    "manifesto_cenarios_estruturais.csv",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

DIRETORIO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "descoberta_grupos",
)

ARQUIVO_DEPENDENCIA_CENARIOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_dependencia_temporal_energia",
    "dependencia_temporal_energia_por_cenario.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "bootstrap_blocos_energia_normal_only",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "bootstrap_blocos_energia_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "bootstrap_blocos_energia_por_cenario.csv",
)


FPR_ALVO = 0.01
EPS = 1e-12

N_BOOT = 2000
SEED_BOOT = 20261008

ALPHA_IC = 0.05


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

        amplitude = (
            maximo
            -
            minimo
        )

        if abs(
            amplitude
        ) < EPS:

            raise RuntimeError(
                f"{feature}: intervalo físico degenerado."
            )

        X[
            feature
        ] = (
            X[
                feature
            ]
            -
            minimo
        ) / amplitude

    return X.to_numpy(
        dtype=np.float64
    )


def prefixo_circular(
    vetor,
):
    """
    Prefix sum da série duplicada, permitindo somas de blocos circulares
    que começam em qualquer índice 0..n-1.
    """

    vetor = np.asarray(
        vetor,
        dtype=np.float64,
    )

    ext = np.concatenate(
        [
            vetor,
            vetor,
        ]
    )

    prefixo = np.empty(
        len(
            ext
        )
        +
        1,
        dtype=np.float64,
    )

    prefixo[
        0
    ] = 0.0

    np.cumsum(
        ext,
        out=prefixo[
            1:
        ],
    )

    return prefixo


def somar_blocos(
    prefixo,
    inicios,
    comprimento,
):
    """
    Soma, em cada réplica, todos os blocos de mesmo comprimento.

    inicios: matriz [n_boot, n_blocos]
    """

    if inicios.size == 0:

        return np.zeros(
            inicios.shape[
                0
            ],
            dtype=np.float64,
        )

    somas = (
        prefixo[
            inicios
            +
            comprimento
        ]
        -
        prefixo[
            inicios
        ]
    )

    return np.sum(
        somas,
        axis=1,
    )


def gerar_plano_bootstrap(
    rng,
    n,
    bloco,
    n_boot,
):
    """
    Gera plano circular com comprimento total exatamente n em cada réplica.
    O mesmo plano é reutilizado em todos os grupos do cenário.
    """

    if bloco < 1:

        raise RuntimeError(
            "Comprimento de bloco inválido."
        )

    bloco = min(
        bloco,
        n,
    )

    n_blocos_completos = (
        n
        //
        bloco
    )

    resto = (
        n
        %
        bloco
    )

    inicios_completos = rng.integers(
        0,
        n,
        size=(
            n_boot,
            n_blocos_completos,
        ),
        endpoint=False,
    )

    if resto > 0:

        inicios_resto = rng.integers(
            0,
            n,
            size=(
                n_boot,
                1,
            ),
            endpoint=False,
        )

    else:

        inicios_resto = np.empty(
            (
                n_boot,
                0,
            ),
            dtype=np.int64,
        )

    return (
        inicios_completos,
        inicios_resto,
        resto,
    )


def bootstrap_somas(
    vetor,
    inicios_completos,
    bloco,
    inicios_resto,
    resto,
):
    """
    Soma o vetor ao longo dos índices reamostrados, usando prefixos
    circulares e sem materializar os 10.000 índices de cada réplica.
    """

    prefixo = prefixo_circular(
        vetor
    )

    total = somar_blocos(
        prefixo,
        inicios_completos,
        bloco,
    )

    if resto > 0:

        total += somar_blocos(
            prefixo,
            inicios_resto,
            resto,
        )

    return total


def intervalo_percentil(
    valores,
    alpha=0.05,
):

    return (
        float(
            np.quantile(
                valores,
                alpha
                /
                2.0,
            )
        ),
        float(
            np.quantile(
                valores,
                1.0
                -
                alpha
                /
                2.0,
            )
        ),
    )


def classificar_grupo(
    equivalente,
    p_low,
    p_high,
    log_low,
    log_high,
):

    if equivalente:

        return "equivalente"

    if (
        p_low
        >
        0.5
        and
        log_low
        >
        0.0
    ):

        return "local_suportado"

    if (
        p_high
        <
        0.5
        and
        log_high
        <
        0.0
    ):

        return "global_suportado"

    return "indefinido"


def classificar_cenario(
    classes,
):

    conjunto = set(
        classes
    )

    if conjunto == {
        "equivalente"
    }:

        return "equivalente"

    if conjunto.issubset(
        {
            "local_suportado",
            "equivalente",
        }
    ) and (
        "local_suportado"
        in
        conjunto
    ):

        return "local_muito_provavel"

    if conjunto.issubset(
        {
            "global_suportado",
            "equivalente",
        }
    ) and (
        "global_suportado"
        in
        conjunto
    ):

        return "local_muito_improvavel"

    return "indefinido"


def main():

    print("=" * 126)
    print("BOOTSTRAP CIRCULAR EM BLOCOS — FAVORABILIDADE NORMAL-ONLY DA ENERGIA")
    print("=" * 126)

    print()
    print(
        f"Réplicas bootstrap: {N_BOOT}"
    )

    print(
        f"IC percentil: {(1.0 - ALPHA_IC) * 100:.1f}%"
    )

    print(
        "Nenhum ataque, P01/P02, m, delta ou rótulo de teste é usado."
    )

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
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

    dependencia = pd.read_csv(
        ARQUIVO_DEPENDENCIA_CENARIOS
    )

    dependencia = (
        dependencia
        .set_index(
            "Cenario"
        )
    )

    rng = np.random.default_rng(
        SEED_BOOT
    )

    resultados_grupos = []
    resultados_cenarios = []

    for _, linha_manifesto in manifesto.iterrows():

        cenario = str(
            linha_manifesto[
                "Cenario"
            ]
        )

        print()
        print("=" * 126)
        print(cenario)
        print("=" * 126)

        treino = pd.read_csv(
            linha_manifesto[
                "Arquivo_Treino"
            ]
        )

        validacao = pd.read_csv(
            linha_manifesto[
                "Arquivo_Validacao"
            ]
        )

        if not (
            treino[
                "Attack"
            ]
            ==
            0
        ).all():

            raise RuntimeError(
                f"{cenario}: treino contém ataque."
            )

        if not (
            validacao[
                "Attack"
            ]
            ==
            0
        ).all():

            raise RuntimeError(
                f"{cenario}: validação contém ataque."
            )

        arquivo_mapa = os.path.join(
            DIRETORIO_GRUPOS,
            cenario,
            "mapa_grupos_descobertos.csv",
        )

        grupos_df = pd.read_csv(
            arquivo_mapa
        )

        features = sorted(
            grupos_df[
                "Feature"
            ].tolist()
        )

        if len(
            features
        ) != 400:

            raise RuntimeError(
                f"{cenario}: esperadas 400 features."
            )

        indice_feature = {
            feature: indice
            for indice, feature
            in enumerate(
                features
            )
        }

        grupos = sorted(
            grupos_df[
                "Grupo_Descoberto"
            ]
            .unique()
            .tolist()
        )

        indices_por_grupo = {}

        for grupo in grupos:

            features_grupo = (
                grupos_df.loc[
                    grupos_df[
                        "Grupo_Descoberto"
                    ]
                    ==
                    grupo,
                    "Feature",
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
                dtype=np.int64,
            )

        X_train = normalizar_fisicamente(
            treino,
            features,
            parametros,
        )

        X_val = normalizar_fisicamente(
            validacao,
            features,
            parametros,
        )

        media_train = np.mean(
            X_train,
            axis=0,
        )

        desvio_train = np.std(
            X_train,
            axis=0,
            ddof=0,
        )

        if np.any(
            desvio_train
            <
            EPS
        ):

            raise RuntimeError(
                f"{cenario}: feature com desvio padrão praticamente zero."
            )

        Z_val = (
            X_val
            -
            media_train
        ) / desvio_train

        Z2_val = (
            Z_val
            **
            2
        )

        D = int(
            Z2_val.shape[
                1
            ]
        )

        score_global = np.mean(
            Z2_val,
            axis=1,
        )

        tau_global = float(
            np.quantile(
                score_global,
                1.0
                -
                FPR_ALVO,
            )
        )

        scores_locais = {}

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            scores_locais[
                grupo
            ] = np.mean(
                Z2_val[
                    :,
                    idx
                ],
                axis=1,
            )

        matriz_local = np.column_stack(
            [
                scores_locais[
                    grupo
                ]
                for grupo
                in grupos
            ]
        )

        score_fusao = np.max(
            matriz_local,
            axis=1,
        )

        tau_local = float(
            np.quantile(
                score_fusao,
                1.0
                -
                FPR_ALVO,
            )
        )

        mascara_elegivel = (
            (score_global < tau_global)
            &
            (score_fusao < tau_local)
        )

        n_elegivel = int(
            np.sum(
                mascara_elegivel
            )
        )

        n = len(
            validacao
        )

        if cenario not in dependencia.index:

            raise RuntimeError(
                f"{cenario}: cenário ausente na auditoria temporal."
            )

        bloco = int(
            round(
                float(
                    dependencia.loc[
                        cenario,
                        "Lag_Decorrelacao_005_Max_Entre_Grupos",
                    ]
                )
            )
        )

        bloco = max(
            1,
            min(
                bloco,
                n,
            ),
        )

        (
            inicios_completos,
            inicios_resto,
            resto,
        ) = gerar_plano_bootstrap(
            rng,
            n,
            bloco,
            N_BOOT,
        )

        classes_grupos = []
        registros_cenario = []

        q_global_raw = (
            D
            *
            (
                tau_global
                -
                score_global
            )
        )

        elegivel_float = mascara_elegivel.astype(
            np.float64
        )

        boot_n_elegivel = bootstrap_somas(
            elegivel_float,
            inicios_completos,
            bloco,
            inicios_resto,
            resto,
        )

        if np.any(
            boot_n_elegivel
            <=
            0
        ):

            raise RuntimeError(
                f"{cenario}: réplica bootstrap sem estados elegíveis."
            )

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            d_k = int(
                len(
                    idx
                )
            )

            score_local = scores_locais[
                grupo
            ]

            q_local_raw = (
                d_k
                *
                (
                    tau_local
                    -
                    score_local
                )
            )

            qg = q_global_raw[
                mascara_elegivel
            ]

            ql = q_local_raw[
                mascara_elegivel
            ]

            if np.any(
                qg
                <=
                0
            ) or np.any(
                ql
                <=
                0
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: Q não positivo em elegíveis."
                )

            equivalente = bool(
                np.allclose(
                    qg,
                    ql,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )

            win_local_full = np.zeros(
                n,
                dtype=np.float64,
            )

            win_local_full[
                mascara_elegivel
            ] = (
                ql
                <
                qg
            ).astype(
                np.float64
            )

            log_ratio_full = np.zeros(
                n,
                dtype=np.float64,
            )

            log_ratio_full[
                mascara_elegivel
            ] = np.log(
                qg
                /
                ql
            )

            p_obs = float(
                np.mean(
                    ql
                    <
                    qg
                )
            )

            log_obs = float(
                np.mean(
                    np.log(
                        qg
                        /
                        ql
                    )
                )
            )

            boot_wins = bootstrap_somas(
                win_local_full,
                inicios_completos,
                bloco,
                inicios_resto,
                resto,
            )

            boot_log_sum = bootstrap_somas(
                log_ratio_full,
                inicios_completos,
                bloco,
                inicios_resto,
                resto,
            )

            p_boot = (
                boot_wins
                /
                boot_n_elegivel
            )

            log_boot = (
                boot_log_sum
                /
                boot_n_elegivel
            )

            p_low, p_high = intervalo_percentil(
                p_boot,
                ALPHA_IC,
            )

            log_low, log_high = intervalo_percentil(
                log_boot,
                ALPHA_IC,
            )

            classe = classificar_grupo(
                equivalente,
                p_low,
                p_high,
                log_low,
                log_high,
            )

            classes_grupos.append(
                classe
            )

            registro = {
                "Cenario":
                    cenario,

                "Grupo":
                    grupo,

                "Tamanho_Grupo":
                    d_k,

                "N_Validacao":
                    n,

                "N_Estados_Elegiveis":
                    n_elegivel,

                "Comprimento_Bloco":
                    bloco,

                "N_Bootstrap":
                    N_BOOT,

                "P_Qlocal_menor_Qglobal_Obs":
                    p_obs,

                "P_Qlocal_menor_Qglobal_IC95_Low":
                    p_low,

                "P_Qlocal_menor_Qglobal_IC95_High":
                    p_high,

                "Media_Log_Qglobal_sobre_Qlocal_Obs":
                    log_obs,

                "Media_Log_Qglobal_sobre_Qlocal_IC95_Low":
                    log_low,

                "Media_Log_Qglobal_sobre_Qlocal_IC95_High":
                    log_high,

                "Classificacao_Bootstrap":
                    classe,
            }

            resultados_grupos.append(
                registro
            )

            registros_cenario.append(
                registro
            )

        rc = pd.DataFrame(
            registros_cenario
        )

        classe_cenario = classificar_cenario(
            classes_grupos
        )

        contagens = (
            rc[
                "Classificacao_Bootstrap"
            ]
            .value_counts()
            .to_dict()
        )

        resultados_cenarios.append(
            {
                "Cenario":
                    cenario,

                "N_Grupos":
                    len(
                        grupos
                    ),

                "N_Estados_Elegiveis":
                    n_elegivel,

                "Comprimento_Bloco":
                    bloco,

                "N_Bootstrap":
                    N_BOOT,

                "Menor_P_Obs_Entre_Grupos":
                    float(
                        rc[
                            "P_Qlocal_menor_Qglobal_Obs"
                        ].min()
                    ),

                "Menor_P_IC95_Low_Entre_Grupos":
                    float(
                        rc[
                            "P_Qlocal_menor_Qglobal_IC95_Low"
                        ].min()
                    ),

                "Maior_P_IC95_High_Entre_Grupos":
                    float(
                        rc[
                            "P_Qlocal_menor_Qglobal_IC95_High"
                        ].max()
                    ),

                "Menor_Log_Obs_Entre_Grupos":
                    float(
                        rc[
                            "Media_Log_Qglobal_sobre_Qlocal_Obs"
                        ].min()
                    ),

                "Menor_Log_IC95_Low_Entre_Grupos":
                    float(
                        rc[
                            "Media_Log_Qglobal_sobre_Qlocal_IC95_Low"
                        ].min()
                    ),

                "Maior_Log_IC95_High_Entre_Grupos":
                    float(
                        rc[
                            "Media_Log_Qglobal_sobre_Qlocal_IC95_High"
                        ].max()
                    ),

                "N_Grupos_Local_Suportado":
                    int(
                        contagens.get(
                            "local_suportado",
                            0,
                        )
                    ),

                "N_Grupos_Global_Suportado":
                    int(
                        contagens.get(
                            "global_suportado",
                            0,
                        )
                    ),

                "N_Grupos_Equivalente":
                    int(
                        contagens.get(
                            "equivalente",
                            0,
                        )
                    ),

                "N_Grupos_Indefinido":
                    int(
                        contagens.get(
                            "indefinido",
                            0,
                        )
                    ),

                "Classificacao_Cenario_Bootstrap":
                    classe_cenario,
            }
        )

        print(
            f"Estados elegíveis:                  {n_elegivel}/{n}"
        )

        print(
            f"Comprimento de bloco:               {bloco}"
        )

        print(
            f"p observado, pior grupo:            "
            f"{rc['P_Qlocal_menor_Qglobal_Obs'].min():.4f}"
        )

        print(
            f"limite inferior 95% de p, pior:     "
            f"{rc['P_Qlocal_menor_Qglobal_IC95_Low'].min():.4f}"
        )

        print(
            f"log-ratio observado, pior grupo:    "
            f"{rc['Media_Log_Qglobal_sobre_Qlocal_Obs'].min():.4f}"
        )

        print(
            f"limite inferior 95% log, pior:      "
            f"{rc['Media_Log_Qglobal_sobre_Qlocal_IC95_Low'].min():.4f}"
        )

        print(
            "Grupos:                            "
            f"local={contagens.get('local_suportado', 0)} | "
            f"global={contagens.get('global_suportado', 0)} | "
            f"equiv={contagens.get('equivalente', 0)} | "
            f"indef={contagens.get('indefinido', 0)}"
        )

        print(
            f"Classificação do cenário:           {classe_cenario}"
        )

    df_grupos = pd.DataFrame(
        resultados_grupos
    )

    df_cenarios = pd.DataFrame(
        resultados_cenarios
    )

    df_grupos.to_csv(
        ARQUIVO_GRUPOS,
        index=False,
    )

    df_cenarios.to_csv(
        ARQUIVO_CENARIOS,
        index=False,
    )

    print()
    print("=" * 126)
    print("RESUMO FINAL")
    print("=" * 126)
    print()

    colunas = [
        "Cenario",
        "N_Grupos",
        "Comprimento_Bloco",
        "Menor_P_Obs_Entre_Grupos",
        "Menor_P_IC95_Low_Entre_Grupos",
        "Menor_Log_Obs_Entre_Grupos",
        "Menor_Log_IC95_Low_Entre_Grupos",
        "N_Grupos_Local_Suportado",
        "N_Grupos_Global_Suportado",
        "N_Grupos_Equivalente",
        "N_Grupos_Indefinido",
        "Classificacao_Cenario_Bootstrap",
    ]

    print(
        df_cenarios[
            colunas
        ].to_string(
            index=False
        )
    )

    print()
    print("IMPORTANTE:")
    print(
        "1) O bootstrap é temporal e pareado: Q_G e Q_L são reamostrados pelos mesmos blocos."
    )
    print(
        "2) O comprimento do bloco vem exclusivamente da auditoria normal-only de autocorrelação."
    )
    print(
        "3) Nenhum ataque ou resultado P01/P02 participa da classificação."
    )
    print(
        "4) A classificação combina duas evidências: probabilidade pareada Q_L<Q_G e média do log-ratio Q_G/Q_L."
    )
    print(
        "5) Esta execução ainda deve ser auditada antes de substituir formalmente o diagnóstico anterior."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)


if __name__ == "__main__":
    main()
