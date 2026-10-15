import os

import numpy as np
import pandas as pd


# ============================================================================
# DIAGNÓSTICO ENERGIA NORMAL-ONLY — V2
#
# Mantém o fator original:
#
#   F_k^E = (D/d_k) * gap_G / gap_L,k
#
# e acrescenta um diagnóstico amostra a amostra da quantidade de evidência
# positiva localizada necessária para atingir os limiares global e local.
#
# Nenhum ataque, m, delta, mapa verdadeiro ou rótulo de teste é utilizado.
# ============================================================================


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

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "diagnostico_energia_normal_only_v2",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_energia_v2_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_energia_v2_por_cenario.csv",
)

ARQUIVO_DISTRIBUICAO = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_energia_v2_distribuicao.csv",
)


FPR_ALVO = 0.01
EPS = 1e-12


# ============================================================================
# FUNÇÕES
# ============================================================================

def normalizar_fisicamente(df, features, parametros):

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


def classificar_fator_original(
    fator_min,
    fator_max,
):

    if fator_min > 1.0:
        return "favoravel_local_todos_grupos"

    if fator_max < 1.0:
        return "favoravel_global_todos_grupos"

    return "dependente_localizacao"


def classificar_distribuicao_estrita(
    f_min,
    f_max,
):

    if f_min > 1.0:
        return "local_estrito"

    if f_max < 1.0:
        return "global_estrito"

    return "dependente_estado"


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 118)
    print("DIAGNÓSTICO ENERGIA NORMAL-ONLY V2 — MARGENS AMOSTRA A AMOSTRA")
    print("=" * 118)

    print()
    print(
        "O fator original é preservado. "
        "A nova camada usa somente treino normal, validação normal "
        "e a partição descoberta no normal."
    )

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    if len(
        manifesto
    ) != 12:

        raise RuntimeError(
            "Esperados exatamente 12 cenários."
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

    resultados_grupos = []
    resultados_cenarios = []
    resultados_distribuicao = []

    for _, linha_manifesto in manifesto.iterrows():

        cenario = str(
            linha_manifesto[
                "Cenario"
            ]
        )

        alpha = float(
            linha_manifesto[
                "Alpha_Grupo"
            ]
        )

        beta = float(
            linha_manifesto[
                "Beta_Global"
            ]
        )

        print()
        print("=" * 118)
        print(
            f"{cenario} | "
            f"alpha={alpha:.2f} | "
            f"beta={beta:.2f}"
        )
        print("=" * 118)

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

        if len(
            set(
                features
            )
        ) != len(
            features
        ):

            raise RuntimeError(
                f"{cenario}: existem features duplicadas."
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
        tamanhos_grupos = []

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

            idx = np.asarray(
                [
                    indice_feature[
                        feature
                    ]
                    for feature
                    in features_grupo
                ],
                dtype=np.int64,
            )

            if len(
                idx
            ) == 0:

                raise RuntimeError(
                    f"{cenario} {grupo}: grupo vazio."
                )

            indices_por_grupo[
                grupo
            ] = idx

            tamanhos_grupos.append(
                len(
                    idx
                )
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

        if D != 400:

            raise RuntimeError(
                f"{cenario}: D={D}, esperado 400."
            )

        score_global_val = np.mean(
            Z2_val,
            axis=1,
        )

        tau_global = float(
            np.quantile(
                score_global_val,
                1.0
                -
                FPR_ALVO,
            )
        )

        media_global = float(
            np.mean(
                score_global_val
            )
        )

        gap_global = (
            tau_global
            -
            media_global
        )

        if gap_global <= 0:

            raise RuntimeError(
                f"{cenario}: gap global não positivo."
            )

        scores_locais_val = {}

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            scores_locais_val[
                grupo
            ] = np.mean(
                Z2_val[
                    :,
                    idx
                ],
                axis=1,
            )

        matriz_scores_locais = np.column_stack(
            [
                scores_locais_val[
                    grupo
                ]
                for grupo
                in grupos
            ]
        )

        score_fusao_local = np.max(
            matriz_scores_locais,
            axis=1,
        )

        tau_local = float(
            np.quantile(
                score_fusao_local,
                1.0
                -
                FPR_ALVO,
            )
        )

        mascara_elegivel = (
            (score_global_val < tau_global)
            &
            (score_fusao_local < tau_local)
        )

        n_elegiveis = int(
            np.sum(
                mascara_elegivel
            )
        )

        if n_elegiveis == 0:

            raise RuntimeError(
                f"{cenario}: nenhum estado normal elegível."
            )

        fatores_originais = []

        fracoes_dec_maior_1 = []
        q05_dec_cenario = []
        medianas_dec_cenario = []
        q95_dec_cenario = []
        mins_dec_cenario = []
        maxs_dec_cenario = []

        classificacoes_grupo = []

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            d_k = int(
                len(
                    idx
                )
            )

            score_grupo = scores_locais_val[
                grupo
            ]

            media_local = float(
                np.mean(
                    score_grupo
                )
            )

            gap_local = (
                tau_local
                -
                media_local
            )

            if gap_local <= 0:

                raise RuntimeError(
                    f"{cenario} {grupo}: gap local não positivo."
                )

            burden_ratio = (
                gap_global
                /
                gap_local
            )

            fator_original = (
                D
                /
                d_k
            ) * burden_ratio

            evidencia_global_media = (
                D
                *
                gap_global
            )

            evidencia_local_media = (
                d_k
                *
                gap_local
            )

            razao_evidencias_media = (
                evidencia_global_media
                /
                evidencia_local_media
            )

            if not np.isclose(
                fator_original,
                razao_evidencias_media,
                rtol=1e-10,
                atol=1e-12,
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: inconsistência no fator original."
                )

            sG = score_global_val[
                mascara_elegivel
            ]

            sK = score_grupo[
                mascara_elegivel
            ]

            Q_global = (
                D
                *
                (
                    tau_global
                    -
                    sG
                )
            )

            Q_local = (
                d_k
                *
                (
                    tau_local
                    -
                    sK
                )
            )

            if np.any(
                Q_global
                <=
                0
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: Q_global não positivo em elegíveis."
                )

            if np.any(
                Q_local
                <=
                0
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: Q_local não positivo em elegíveis."
                )

            F_dec = (
                Q_global
                /
                Q_local
            )

            if np.any(
                ~np.isfinite(
                    F_dec
                )
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: F_dec contém valor não finito."
                )

            f_dec_min = float(
                np.min(
                    F_dec
                )
            )

            f_dec_q01 = float(
                np.quantile(
                    F_dec,
                    0.01,
                )
            )

            f_dec_q05 = float(
                np.quantile(
                    F_dec,
                    0.05,
                )
            )

            f_dec_q25 = float(
                np.quantile(
                    F_dec,
                    0.25,
                )
            )

            f_dec_med = float(
                np.median(
                    F_dec
                )
            )

            f_dec_q75 = float(
                np.quantile(
                    F_dec,
                    0.75,
                )
            )

            f_dec_q95 = float(
                np.quantile(
                    F_dec,
                    0.95,
                )
            )

            f_dec_q99 = float(
                np.quantile(
                    F_dec,
                    0.99,
                )
            )

            f_dec_max = float(
                np.max(
                    F_dec
                )
            )

            fracao_maior_1 = float(
                np.mean(
                    F_dec
                    >
                    1.0
                )
            )

            classificacao_estrita = (
                classificar_distribuicao_estrita(
                    f_dec_min,
                    f_dec_max,
                )
            )

            classificacoes_grupo.append(
                classificacao_estrita
            )

            resultados_grupos.append(
                {
                    "Cenario":
                        cenario,

                    "Alpha_Grupo":
                        alpha,

                    "Beta_Global":
                        beta,

                    "Grupo":
                        grupo,

                    "D_Global":
                        D,

                    "Tamanho_Grupo":
                        d_k,

                    "N_Grupos_Descobertos":
                        len(
                            grupos
                        ),

                    "N_Validacao":
                        len(
                            validacao
                        ),

                    "N_Estados_Elegiveis":
                        n_elegiveis,

                    "Fracao_Estados_Elegiveis":
                        n_elegiveis
                        /
                        len(
                            validacao
                        ),

                    "Tau_Global_Energia":
                        tau_global,

                    "Media_Global_Energia":
                        media_global,

                    "Gap_Global_Energia":
                        gap_global,

                    "Tau_Local_Fusao_Energia":
                        tau_local,

                    "Media_Local_Energia":
                        media_local,

                    "Gap_Local_Energia":
                        gap_local,

                    "B_Energia":
                        burden_ratio,

                    "Amplificacao_D_por_dk":
                        D
                        /
                        d_k,

                    "Fator_Energia_Original":
                        fator_original,

                    "Evidencia_Global_Media":
                        evidencia_global_media,

                    "Evidencia_Local_Media":
                        evidencia_local_media,

                    "F_Dec_Min":
                        f_dec_min,

                    "F_Dec_Q01":
                        f_dec_q01,

                    "F_Dec_Q05":
                        f_dec_q05,

                    "F_Dec_Q25":
                        f_dec_q25,

                    "F_Dec_Mediana":
                        f_dec_med,

                    "F_Dec_Q75":
                        f_dec_q75,

                    "F_Dec_Q95":
                        f_dec_q95,

                    "F_Dec_Q99":
                        f_dec_q99,

                    "F_Dec_Max":
                        f_dec_max,

                    "Fracao_F_Dec_Maior_1":
                        fracao_maior_1,

                    "Classificacao_Estrita_F_Dec":
                        classificacao_estrita,
                }
            )

            indices_validacao = np.flatnonzero(
                mascara_elegivel
            )

            for pos, indice_val in enumerate(
                indices_validacao
            ):

                resultados_distribuicao.append(
                    {
                        "Cenario":
                            cenario,

                        "Grupo":
                            grupo,

                        "Indice_Validacao":
                            int(
                                indice_val
                            ),

                        "Score_Global":
                            float(
                                sG[
                                    pos
                                ]
                            ),

                        "Score_Local_Grupo":
                            float(
                                sK[
                                    pos
                                ]
                            ),

                        "Tau_Global":
                            tau_global,

                        "Tau_Local_Fusao":
                            tau_local,

                        "Q_Global":
                            float(
                                Q_global[
                                    pos
                                ]
                            ),

                        "Q_Local":
                            float(
                                Q_local[
                                    pos
                                ]
                            ),

                        "F_Dec":
                            float(
                                F_dec[
                                    pos
                                ]
                            ),
                    }
                )

            fatores_originais.append(
                fator_original
            )

            fracoes_dec_maior_1.append(
                fracao_maior_1
            )

            q05_dec_cenario.append(
                f_dec_q05
            )

            medianas_dec_cenario.append(
                f_dec_med
            )

            q95_dec_cenario.append(
                f_dec_q95
            )

            mins_dec_cenario.append(
                f_dec_min
            )

            maxs_dec_cenario.append(
                f_dec_max
            )

        fatores_originais = np.asarray(
            fatores_originais,
            dtype=np.float64,
        )

        fracoes_dec_maior_1 = np.asarray(
            fracoes_dec_maior_1,
            dtype=np.float64,
        )

        q05_dec_cenario = np.asarray(
            q05_dec_cenario,
            dtype=np.float64,
        )

        medianas_dec_cenario = np.asarray(
            medianas_dec_cenario,
            dtype=np.float64,
        )

        q95_dec_cenario = np.asarray(
            q95_dec_cenario,
            dtype=np.float64,
        )

        mins_dec_cenario = np.asarray(
            mins_dec_cenario,
            dtype=np.float64,
        )

        maxs_dec_cenario = np.asarray(
            maxs_dec_cenario,
            dtype=np.float64,
        )

        f_original_min = float(
            np.min(
                fatores_originais
            )
        )

        f_original_max = float(
            np.max(
                fatores_originais
            )
        )

        classificacao_original = (
            classificar_fator_original(
                f_original_min,
                f_original_max,
            )
        )

        if all(
            x
            ==
            "local_estrito"
            for x
            in classificacoes_grupo
        ):

            classificacao_dec_cenario = (
                "local_estrito_todos_grupos"
            )

        elif all(
            x
            ==
            "global_estrito"
            for x
            in classificacoes_grupo
        ):

            classificacao_dec_cenario = (
                "global_estrito_todos_grupos"
            )

        else:

            classificacao_dec_cenario = (
                "dependente_estado_ou_grupo"
            )

        resultados_cenarios.append(
            {
                "Cenario":
                    cenario,

                "Alpha_Grupo":
                    alpha,

                "Beta_Global":
                    beta,

                "D_Global":
                    D,

                "N_Grupos_Descobertos":
                    len(
                        grupos
                    ),

                "Tamanho_Min":
                    min(
                        tamanhos_grupos
                    ),

                "Tamanho_Mediana":
                    float(
                        np.median(
                            tamanhos_grupos
                        )
                    ),

                "Tamanho_Max":
                    max(
                        tamanhos_grupos
                    ),

                "N_Validacao":
                    len(
                        validacao
                    ),

                "N_Estados_Elegiveis":
                    n_elegiveis,

                "Fracao_Estados_Elegiveis":
                    n_elegiveis
                    /
                    len(
                        validacao
                    ),

                "Tau_Global_Energia":
                    tau_global,

                "Tau_Local_Fusao_Energia":
                    tau_local,

                "F_Original_Min":
                    f_original_min,

                "F_Original_Mediana":
                    float(
                        np.median(
                            fatores_originais
                        )
                    ),

                "F_Original_Max":
                    f_original_max,

                "Classificacao_Original":
                    classificacao_original,

                "Frac_F_Dec_Maior_1_Min_Entre_Grupos":
                    float(
                        np.min(
                            fracoes_dec_maior_1
                        )
                    ),

                "Frac_F_Dec_Maior_1_Mediana_Entre_Grupos":
                    float(
                        np.median(
                            fracoes_dec_maior_1
                        )
                    ),

                "Frac_F_Dec_Maior_1_Max_Entre_Grupos":
                    float(
                        np.max(
                            fracoes_dec_maior_1
                        )
                    ),

                "F_Dec_Q05_Min_Entre_Grupos":
                    float(
                        np.min(
                            q05_dec_cenario
                        )
                    ),

                "F_Dec_Mediana_Min_Entre_Grupos":
                    float(
                        np.min(
                            medianas_dec_cenario
                        )
                    ),

                "F_Dec_Q95_Min_Entre_Grupos":
                    float(
                        np.min(
                            q95_dec_cenario
                        )
                    ),

                "F_Dec_Min_Absoluto":
                    float(
                        np.min(
                            mins_dec_cenario
                        )
                    ),

                "F_Dec_Max_Absoluto":
                    float(
                        np.max(
                            maxs_dec_cenario
                        )
                    ),

                "Classificacao_Estrita_Decisao":
                    classificacao_dec_cenario,
            }
        )

        print(
            f"Grupos descobertos:                  {len(grupos)}"
        )

        print(
            f"Estados normais elegíveis:           "
            f"{n_elegiveis}/{len(validacao)} "
            f"({100.0 * n_elegiveis / len(validacao):.2f}%)"
        )

        print(
            f"F original:                           "
            f"min={f_original_min:.6f} | "
            f"med={np.median(fatores_originais):.6f} | "
            f"max={f_original_max:.6f}"
        )

        print(
            f"F_dec > 1 (entre grupos):             "
            f"min={np.min(fracoes_dec_maior_1):.4f} | "
            f"med={np.median(fracoes_dec_maior_1):.4f} | "
            f"max={np.max(fracoes_dec_maior_1):.4f}"
        )

        print(
            f"Q05(F_dec), pior grupo:               "
            f"{np.min(q05_dec_cenario):.6f}"
        )

        print(
            f"Mediana(F_dec), pior grupo:           "
            f"{np.min(medianas_dec_cenario):.6f}"
        )

        print(
            f"Q95(F_dec), pior grupo:               "
            f"{np.min(q95_dec_cenario):.6f}"
        )

        print(
            f"Classificação estrita decisão:        "
            f"{classificacao_dec_cenario}"
        )

    df_grupos = pd.DataFrame(
        resultados_grupos
    )

    df_cenarios = pd.DataFrame(
        resultados_cenarios
    )

    df_distribuicao = pd.DataFrame(
        resultados_distribuicao
    )

    df_grupos.to_csv(
        ARQUIVO_GRUPOS,
        index=False,
    )

    df_cenarios.to_csv(
        ARQUIVO_CENARIOS,
        index=False,
    )

    df_distribuicao.to_csv(
        ARQUIVO_DISTRIBUICAO,
        index=False,
    )

    print()
    print("=" * 118)
    print("RESUMO — DIAGNÓSTICO ENERGIA NORMAL-ONLY V2")
    print("=" * 118)
    print()

    colunas_resumo = [
        "Cenario",
        "N_Grupos_Descobertos",
        "F_Original_Min",
        "Frac_F_Dec_Maior_1_Min_Entre_Grupos",
        "F_Dec_Q05_Min_Entre_Grupos",
        "F_Dec_Mediana_Min_Entre_Grupos",
        "F_Dec_Q95_Min_Entre_Grupos",
        "Classificacao_Estrita_Decisao",
    ]

    print(
        df_cenarios[
            colunas_resumo
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("IMPORTANTE:")

    print(
        "1) Nenhum ataque, m, delta, mapa verdadeiro ou rótulo de teste "
        "foi utilizado."
    )

    print(
        "2) O fator original foi preservado para comparação."
    )

    print(
        "3) F_dec compara, em cada estado normal elegível, a quantidade "
        "de evidência positiva localizada necessária para cruzar os "
        "limiares global e local."
    )

    print(
        "4) A classificação estrita não usa percentis arbitrários: "
        "somente declara domínio local/global quando todos os estados "
        "normais elegíveis do grupo ficam do mesmo lado de 1."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)
    print(ARQUIVO_DISTRIBUICAO)


if __name__ == "__main__":
    main()
