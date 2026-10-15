import os

import numpy as np
import pandas as pd


# ============================================================================
# DIAGNÓSTICO ENERGIA NORMAL-ONLY — V3
# DOMINÂNCIA DAS DISTRIBUIÇÕES DE EVIDÊNCIA NECESSÁRIA
#
# Objetivo:
# classificar a favorabilidade da avaliação local SEM usar ataques.
#
# Para cada estado normal elegível t e grupo descoberto G_k:
#
#   Q_G(t)   = D   * [tau_G - s_G(t)]
#   Q_L,k(t) = d_k * [tau_L - s_k(t)]
#
# Q representa a quantidade de evidência POSITIVA localizada que ainda seria
# necessária para a decisão cruzar o respectivo limiar.
#
# Para cada grupo são comparadas as distribuições empíricas de Q_G e Q_L,k.
#
# - LOCAL_DOMINA:
#       F_L(q) >= F_G(q) para todo q, com desigualdade estrita em algum q.
#
# - GLOBAL_DOMINA:
#       F_G(q) >= F_L(q) para todo q, com desigualdade estrita em algum q.
#
# - EQUIVALENTE:
#       as CDFs empíricas coincidem.
#
# - INDEFINIDO:
#       as CDFs se cruzam.
#
# Classificação do cenário:
#
# - local_muito_provavel:
#       todos os grupos são LOCAL_DOMINA ou EQUIVALENTE e pelo menos um grupo
#       é LOCAL_DOMINA.
#
# - local_muito_improvavel:
#       todos os grupos são GLOBAL_DOMINA ou EQUIVALENTE e pelo menos um grupo
#       é GLOBAL_DOMINA.
#
# - equivalente:
#       todos os grupos são EQUIVALENTE.
#
# - indefinido:
#       qualquer outro caso.
#
# Nenhum ataque, m, delta, direção, mapa verdadeiro ou rótulo de teste é usado.
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
    "diagnostico_energia_normal_only_v3",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_energia_v3_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_energia_v3_por_cenario.csv",
)


FPR_ALVO = 0.01
EPS = 1e-12

# A comparação das CDFs é feita em proporções de amostras.
# O valor abaixo serve apenas para absorver erro de ponto flutuante.
TOL_CDF = 1e-12


# ============================================================================
# FUNÇÕES
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


def comparar_dominancia_empirica(
    q_global,
    q_local,
):

    q_global = np.asarray(
        q_global,
        dtype=np.float64,
    )

    q_local = np.asarray(
        q_local,
        dtype=np.float64,
    )

    if q_global.ndim != 1:
        raise RuntimeError(
            "q_global deve ser vetor."
        )

    if q_local.ndim != 1:
        raise RuntimeError(
            "q_local deve ser vetor."
        )

    if len(
        q_global
    ) != len(
        q_local
    ):

        raise RuntimeError(
            "q_global e q_local devem ter o mesmo número de estados."
        )

    if len(
        q_global
    ) == 0:

        raise RuntimeError(
            "Vetores Q vazios."
        )

    if np.any(
        q_global
        <=
        0
    ):

        raise RuntimeError(
            "q_global contém valor não positivo."
        )

    if np.any(
        q_local
        <=
        0
    ):

        raise RuntimeError(
            "q_local contém valor não positivo."
        )

    qg = np.sort(
        q_global
    )

    ql = np.sort(
        q_local
    )

    suporte = np.unique(
        np.concatenate(
            [
                qg,
                ql,
            ]
        )
    )

    n = len(
        qg
    )

    # CDF empírica avaliada em todos os pontos onde alguma das duas CDFs muda.
    Fg = (
        np.searchsorted(
            qg,
            suporte,
            side="right",
        )
        /
        n
    )

    Fl = (
        np.searchsorted(
            ql,
            suporte,
            side="right",
        )
        /
        n
    )

    diferenca = (
        Fl
        -
        Fg
    )

    min_diff = float(
        np.min(
            diferenca
        )
    )

    max_diff = float(
        np.max(
            diferenca
        )
    )

    equivalente = (
        abs(
            min_diff
        )
        <=
        TOL_CDF
        and
        abs(
            max_diff
        )
        <=
        TOL_CDF
    )

    local_domina = (
        min_diff
        >=
        -TOL_CDF
        and
        max_diff
        >
        TOL_CDF
    )

    global_domina = (
        max_diff
        <=
        TOL_CDF
        and
        min_diff
        <
        -TOL_CDF
    )

    if equivalente:

        classe = "equivalente"

    elif local_domina:

        classe = "local_domina"

    elif global_domina:

        classe = "global_domina"

    else:

        classe = "indefinido"

    # Medidas auxiliares. Não entram na regra de classificação.
    frac_q_local_menor = float(
        np.mean(
            q_local
            <
            q_global
        )
    )

    frac_q_iguais = float(
        np.mean(
            np.isclose(
                q_local,
                q_global,
                rtol=1e-10,
                atol=1e-12,
            )
        )
    )

    frac_q_global_menor = float(
        np.mean(
            q_global
            <
            q_local
        )
    )

    return {
        "Classificacao_Dominancia":
            classe,

        "CDF_Diferenca_Min_Flocal_menos_Fglobal":
            min_diff,

        "CDF_Diferenca_Max_Flocal_menos_Fglobal":
            max_diff,

        "Frac_Estados_Qlocal_menor_Qglobal":
            frac_q_local_menor,

        "Frac_Estados_Qiguais":
            frac_q_iguais,

        "Frac_Estados_Qglobal_menor_Qlocal":
            frac_q_global_menor,
    }


def classificar_cenario(
    classes_grupos,
):

    classes_grupos = list(
        classes_grupos
    )

    conjunto = set(
        classes_grupos
    )

    if conjunto == {
        "equivalente"
    }:

        return "equivalente"

    if conjunto.issubset(
        {
            "local_domina",
            "equivalente",
        }
    ) and (
        "local_domina"
        in
        conjunto
    ):

        return "local_muito_provavel"

    if conjunto.issubset(
        {
            "global_domina",
            "equivalente",
        }
    ) and (
        "global_domina"
        in
        conjunto
    ):

        return "local_muito_improvavel"

    return "indefinido"


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 122)
    print("DIAGNÓSTICO ENERGIA NORMAL-ONLY V3 — DOMINÂNCIA DE EVIDÊNCIA NECESSÁRIA")
    print("=" * 122)

    print()
    print(
        "Somente treino normal + validação normal + partição descoberta."
    )

    print(
        "Ataques e parâmetros P01/P02 NÃO são usados."
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
        print("=" * 122)
        print(
            f"{cenario} | "
            f"alpha={alpha:.2f} | "
            f"beta={beta:.2f}"
        )
        print("=" * 122)

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

        # --------------------------------------------------------------------
        # NORMALIZAÇÃO FÍSICA
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # PADRONIZAÇÃO APRENDIDA SOMENTE NO TREINO NORMAL
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # GLOBAL
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # LOCAL + FUSÃO
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # UNIVERSO ELEGÍVEL
        #
        # Isolamos estados corretamente classificados como normais por AMBAS
        # as decisões. Assim Q mede somente a evidência adicional necessária
        # para provocar um novo cruzamento de limiar.
        # --------------------------------------------------------------------

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

        score_global_elegivel = (
            score_global_val[
                mascara_elegivel
            ]
        )

        q_global = (
            D
            *
            (
                tau_global
                -
                score_global_elegivel
            )
        )

        if np.any(
            q_global
            <=
            0
        ):

            raise RuntimeError(
                f"{cenario}: Q global não positivo em estado elegível."
            )

        classes_grupos = []
        fatores_originais = []

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

            # ---------------------------------------------------------------
            # FATOR ORIGINAL — PRESERVADO COMO MEDIDA DE AMPLIFICAÇÃO MÉDIA
            # ---------------------------------------------------------------

            fator_original = (
                D
                /
                d_k
            ) * (
                gap_global
                /
                gap_local
            )

            fatores_originais.append(
                fator_original
            )

            # ---------------------------------------------------------------
            # EVIDÊNCIA NECESSÁRIA LOCAL
            # ---------------------------------------------------------------

            score_local_elegivel = (
                score_grupo[
                    mascara_elegivel
                ]
            )

            q_local = (
                d_k
                *
                (
                    tau_local
                    -
                    score_local_elegivel
                )
            )

            if np.any(
                q_local
                <=
                0
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: Q local não positivo em estado elegível."
                )

            dominancia = (
                comparar_dominancia_empirica(
                    q_global,
                    q_local,
                )
            )

            classe_grupo = (
                dominancia[
                    "Classificacao_Dominancia"
                ]
            )

            classes_grupos.append(
                classe_grupo
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

                    "Tau_Local_Fusao_Energia":
                        tau_local,

                    "Media_Global_Energia":
                        media_global,

                    "Media_Local_Energia":
                        media_local,

                    "Gap_Global_Energia":
                        gap_global,

                    "Gap_Local_Energia":
                        gap_local,

                    "Amplificacao_D_por_dk":
                        D
                        /
                        d_k,

                    "Fator_Energia_Original":
                        fator_original,

                    "Q_Global_Min":
                        float(
                            np.min(
                                q_global
                            )
                        ),

                    "Q_Global_Q05":
                        float(
                            np.quantile(
                                q_global,
                                0.05,
                            )
                        ),

                    "Q_Global_Mediana":
                        float(
                            np.median(
                                q_global
                            )
                        ),

                    "Q_Global_Q95":
                        float(
                            np.quantile(
                                q_global,
                                0.95,
                            )
                        ),

                    "Q_Global_Max":
                        float(
                            np.max(
                                q_global
                            )
                        ),

                    "Q_Local_Min":
                        float(
                            np.min(
                                q_local
                            )
                        ),

                    "Q_Local_Q05":
                        float(
                            np.quantile(
                                q_local,
                                0.05,
                            )
                        ),

                    "Q_Local_Mediana":
                        float(
                            np.median(
                                q_local
                            )
                        ),

                    "Q_Local_Q95":
                        float(
                            np.quantile(
                                q_local,
                                0.95,
                            )
                        ),

                    "Q_Local_Max":
                        float(
                            np.max(
                                q_local
                            )
                        ),

                    **dominancia,
                }
            )

        # --------------------------------------------------------------------
        # CLASSIFICAÇÃO DO CENÁRIO
        # --------------------------------------------------------------------

        classificacao_cenario = (
            classificar_cenario(
                classes_grupos
            )
        )

        fatores_originais = np.asarray(
            fatores_originais,
            dtype=np.float64,
        )

        contagens = {
            classe: classes_grupos.count(
                classe
            )
            for classe in [
                "local_domina",
                "global_domina",
                "equivalente",
                "indefinido",
            ]
        }

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
                    float(
                        np.min(
                            fatores_originais
                        )
                    ),

                "F_Original_Mediana":
                    float(
                        np.median(
                            fatores_originais
                        )
                    ),

                "F_Original_Max":
                    float(
                        np.max(
                            fatores_originais
                        )
                    ),

                "N_Grupos_Local_Domina":
                    contagens[
                        "local_domina"
                    ],

                "N_Grupos_Global_Domina":
                    contagens[
                        "global_domina"
                    ],

                "N_Grupos_Equivalente":
                    contagens[
                        "equivalente"
                    ],

                "N_Grupos_Indefinido":
                    contagens[
                        "indefinido"
                    ],

                "Classificacao_Energia_V3":
                    classificacao_cenario,
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
            f"min={np.min(fatores_originais):.6f} | "
            f"med={np.median(fatores_originais):.6f} | "
            f"max={np.max(fatores_originais):.6f}"
        )

        print(
            "Dominância por grupo:                "
            f"local={contagens['local_domina']} | "
            f"global={contagens['global_domina']} | "
            f"equiv={contagens['equivalente']} | "
            f"indef={contagens['indefinido']}"
        )

        print(
            f"Classificação Energia V3:             "
            f"{classificacao_cenario}"
        )

    # =========================================================================
    # SALVAMENTO
    # =========================================================================

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

    # =========================================================================
    # CHECAGENS FINAIS
    # =========================================================================

    if len(
        df_cenarios
    ) != 12:

        raise RuntimeError(
            "Resumo final não contém 12 cenários."
        )

    if len(
        df_grupos
    ) != int(
        df_cenarios[
            "N_Grupos_Descobertos"
        ].sum()
    ):

        raise RuntimeError(
            "Número de linhas por grupo inconsistente."
        )

    classes_validas_grupo = {
        "local_domina",
        "global_domina",
        "equivalente",
        "indefinido",
    }

    if not set(
        df_grupos[
            "Classificacao_Dominancia"
        ].unique()
    ).issubset(
        classes_validas_grupo
    ):

        raise RuntimeError(
            "Classe de grupo inesperada."
        )

    classes_validas_cenario = {
        "local_muito_provavel",
        "local_muito_improvavel",
        "equivalente",
        "indefinido",
    }

    if not set(
        df_cenarios[
            "Classificacao_Energia_V3"
        ].unique()
    ).issubset(
        classes_validas_cenario
    ):

        raise RuntimeError(
            "Classe de cenário inesperada."
        )

    # =========================================================================
    # RESUMO
    # =========================================================================

    print()
    print("=" * 122)
    print("RESUMO — DIAGNÓSTICO ENERGIA NORMAL-ONLY V3")
    print("=" * 122)
    print()

    colunas = [
        "Cenario",
        "N_Grupos_Descobertos",
        "F_Original_Min",
        "N_Grupos_Local_Domina",
        "N_Grupos_Global_Domina",
        "N_Grupos_Equivalente",
        "N_Grupos_Indefinido",
        "Classificacao_Energia_V3",
    ]

    print(
        df_cenarios[
            colunas
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("INTERPRETAÇÃO:")

    print(
        "local_muito_provavel   = local domina global em todos os grupos "
        "(admitindo equivalência em alguns)."
    )

    print(
        "local_muito_improvavel = global domina local em todos os grupos "
        "(admitindo equivalência em alguns)."
    )

    print(
        "indefinido             = há cruzamento de CDFs ou comportamento "
        "heterogêneo entre grupos."
    )

    print(
        "equivalente            = distribuições empíricas de evidência "
        "necessária coincidem em todos os grupos."
    )

    print()
    print("IMPORTANTE:")

    print(
        "1) A classificação usa somente dados normais."
    )

    print(
        "2) O fator F_E original é preservado apenas como medida de "
        "amplificação média; ele não decide a classe V3."
    )

    print(
        "3) A V3 é condicionada a evidência positiva localizada."
    )

    print(
        "4) Não existe corte percentual escolhido após observar P01/P02: "
        "a regra é dominância empírica das CDFs em todo o suporte observado."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)


if __name__ == "__main__":
    main()
