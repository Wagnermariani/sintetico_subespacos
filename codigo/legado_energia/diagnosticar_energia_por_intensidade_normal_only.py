import os

import numpy as np
import pandas as pd


# ============================================================================
# ENERGIA NORMAL-ONLY — FAVORABILIDADE CONDICIONAL À INTENSIDADE DA EVIDÊNCIA
#
# Para um grupo G_k, considere estados normais elegíveis t:
#
#   Q_G(t)   = D   [tau_G - s_G(t)]
#   Q_L,k(t) = d_k [tau_L - s_L,k(t)]
#
# Se uma manifestação futura, localizada em G_k, produzir uma quantidade
# POSITIVA de evidência quadrática total q, então:
#
#   global cruza o limiar <=> q >= Q_G(t)
#   local  cruza o limiar <=> q >= Q_L,k(t)
#
# Definimos, usando SOMENTE a validação normal:
#
#   H_G(q)   = fração empírica de estados com Q_G <= q
#   H_L,k(q) = fração empírica de estados com Q_L,k <= q
#   Delta_k(q) = H_L,k(q) - H_G(q)
#
# Interpretação condicional:
#   Delta > 0 : para essa intensidade q, mais estados normais favorecem
#               cruzamento local;
#   Delta < 0 : para essa intensidade q, mais estados normais favorecem
#               cruzamento global;
#   Delta = 0 : equivalência empírica nessa intensidade.
#
# A curva é avaliada EXATAMENTE nos pontos de salto das duas ECDFs.
# Não há percentil, nível de confiança ou margem escolhida para forçar classe.
#
# Saídas:
#   1) resumo por grupo;
#   2) resumo por cenário;
#   3) intervalos contíguos de sinal da curva Delta(q);
#   4) curva amostrada apenas para visualização (não define a classificação).
#
# LIMITES:
# - diagnóstico condicionado a q > 0;
# - a evidência deve permanecer localizada no grupo analisado;
# - sem informação adicional sobre a intensidade futura q, curvas que cruzam
#   são classificadas como dependentes da intensidade;
# - isto caracteriza suscetibilidade de cruzamento, não prevê recall futuro.
#
# Nenhum ataque, P01/P02, m, delta de ataque ou rótulo de teste é utilizado.
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
    "energia_intensidade_normal_only",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "energia_intensidade_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "energia_intensidade_por_cenario.csv",
)

ARQUIVO_INTERVALOS = os.path.join(
    DIRETORIO_SAIDA,
    "energia_intensidade_intervalos.csv",
)

ARQUIVO_CURVA_AMOSTRADA = os.path.join(
    DIRETORIO_SAIDA,
    "energia_intensidade_curva_amostrada.csv",
)


FPR_ALVO = 0.01
EPS = 1e-12
TOL_EQ = 1e-10

# Apenas para arquivo de visualização. A classificação usa TODOS os pontos.
QUANTIS_VISUALIZACAO = np.linspace(
    0.0,
    1.0,
    101,
)


# ============================================================================
# NORMALIZAÇÃO
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


# ============================================================================
# ECDF E INTERVALOS
# ============================================================================

def ecdf_em_pontos(
    valores_ordenados,
    pontos,
):

    n = len(
        valores_ordenados
    )

    return (
        np.searchsorted(
            valores_ordenados,
            pontos,
            side="right",
        )
        /
        n
    )


def sinal_delta(
    delta,
):

    s = np.zeros(
        len(
            delta
        ),
        dtype=np.int8,
    )

    s[
        delta
        >
        TOL_EQ
    ] = 1

    s[
        delta
        <
        -TOL_EQ
    ] = -1

    return s


def nome_sinal(
    s,
):

    if s > 0:
        return "local"

    if s < 0:
        return "global"

    return "equivalente"


def construir_intervalos(
    cenario,
    grupo,
    q,
    delta,
):
    """
    A ECDF é constante em [q_i, q_{i+1}).
    O último ponto não possui largura à direita dentro do domínio observado.
    """

    registros = []

    if len(
        q
    ) < 2:

        return registros

    s = sinal_delta(
        delta
    )

    inicio = 0

    for i in range(
        1,
        len(
            q
        )
        -
        1,
    ):

        if s[
            i
        ] != s[
            inicio
        ]:

            fim = i

            largura = float(
                q[
                    fim
                ]
                -
                q[
                    inicio
                ]
            )

            if largura > 0:

                fatia_delta = delta[
                    inicio:fim
                ]

                registros.append(
                    {
                        "Cenario":
                            cenario,

                        "Grupo":
                            grupo,

                        "Favorabilidade":
                            nome_sinal(
                                int(
                                    s[
                                        inicio
                                    ]
                                )
                            ),

                        "Q_Inicio_Inclusivo":
                            float(
                                q[
                                    inicio
                                ]
                            ),

                        "Q_Fim_Exclusivo":
                            float(
                                q[
                                    fim
                                ]
                            ),

                        "Largura_Q":
                            largura,

                        "Delta_Min_Intervalo":
                            float(
                                np.min(
                                    fatia_delta
                                )
                            ),

                        "Delta_Max_Intervalo":
                            float(
                                np.max(
                                    fatia_delta
                                )
                            ),
                    }
                )

            inicio = i

    fim = len(
        q
    ) - 1

    if fim > inicio:

        largura = float(
            q[
                fim
            ]
            -
            q[
                inicio
            ]
        )

        if largura > 0:

            fatia_delta = delta[
                inicio:fim
            ]

            registros.append(
                {
                    "Cenario":
                        cenario,

                    "Grupo":
                        grupo,

                    "Favorabilidade":
                        nome_sinal(
                            int(
                                s[
                                    inicio
                                ]
                            )
                        ),

                    "Q_Inicio_Inclusivo":
                        float(
                            q[
                                inicio
                            ]
                        ),

                    "Q_Fim_Exclusivo":
                        float(
                            q[
                                fim
                            ]
                        ),

                    "Largura_Q":
                        largura,

                    "Delta_Min_Intervalo":
                        float(
                            np.min(
                                fatia_delta
                            )
                        ),

                    "Delta_Max_Intervalo":
                        float(
                            np.max(
                                fatia_delta
                            )
                        ),
                }
            )

    return registros


def classificar_curva(
    equivalente,
    minimo,
    maximo,
):

    if equivalente:
        return "equivalente"

    tem_global = (
        minimo
        <
        -TOL_EQ
    )

    tem_local = (
        maximo
        >
        TOL_EQ
    )

    if tem_local and not tem_global:
        return "favorabilidade_local_toda_faixa"

    if tem_global and not tem_local:
        return "favorabilidade_global_toda_faixa"

    if tem_local and tem_global:
        return "dependente_intensidade"

    return "equivalente"


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
            "favorabilidade_local_toda_faixa",
            "equivalente",
        }
    ):
        return "favorabilidade_local_toda_faixa"

    if conjunto.issubset(
        {
            "favorabilidade_global_toda_faixa",
            "equivalente",
        }
    ):
        return "favorabilidade_global_toda_faixa"

    return "dependente_grupo_ou_intensidade"


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 138)
    print("ENERGIA NORMAL-ONLY — FAVORABILIDADE CONDICIONAL À INTENSIDADE DA EVIDÊNCIA")
    print("=" * 138)

    print()
    print(
        "Nenhum ataque, P01/P02, m, delta de ataque ou rótulo de teste é usado."
    )

    print(
        "A classificação usa as ECDFs exatas de Q_G e Q_L nos estados normais elegíveis."
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

    resultados_grupos = []
    resultados_cenarios = []
    resultados_intervalos = []
    resultados_curva_amostrada = []

    for _, linha_manifesto in manifesto.iterrows():

        cenario = str(
            linha_manifesto[
                "Cenario"
            ]
        )

        alpha = float(
            linha_manifesto.get(
                "Alpha_Grupo",
                np.nan,
            )
        )

        beta = float(
            linha_manifesto.get(
                "Beta_Global",
                np.nan,
            )
        )

        print()
        print("=" * 138)
        print(
            f"{cenario} | alpha={alpha:.2f} | beta={beta:.2f}"
        )
        print("=" * 138)

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

        n_total = len(
            validacao
        )

        n_elegivel = int(
            np.sum(
                mascara_elegivel
            )
        )

        if n_elegivel == 0:

            raise RuntimeError(
                f"{cenario}: nenhum estado elegível."
            )

        q_global = (
            D
            *
            (
                tau_global
                -
                score_global[
                    mascara_elegivel
                ]
            )
        )

        if np.any(
            q_global
            <=
            0
        ):

            raise RuntimeError(
                f"{cenario}: Q_G não positivo entre elegíveis."
            )

        classes_cenario = []
        deltas_min_cenario = []
        deltas_max_cenario = []

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
            ][
                mascara_elegivel
            ]

            q_local = (
                d_k
                *
                (
                    tau_local
                    -
                    score_local
                )
            )

            if np.any(
                q_local
                <=
                0
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: Q_L não positivo entre elegíveis."
                )

            qg_sort = np.sort(
                q_global
            )

            ql_sort = np.sort(
                q_local
            )

            equivalente = bool(
                np.allclose(
                    qg_sort,
                    ql_sort,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )

            q_pontos = np.unique(
                np.concatenate(
                    [
                        qg_sort,
                        ql_sort,
                    ]
                )
            )

            hg = ecdf_em_pontos(
                qg_sort,
                q_pontos,
            )

            hl = ecdf_em_pontos(
                ql_sort,
                q_pontos,
            )

            delta = (
                hl
                -
                hg
            )

            delta_min = float(
                np.min(
                    delta
                )
            )

            delta_max = float(
                np.max(
                    delta
                )
            )

            idx_min = int(
                np.argmin(
                    delta
                )
            )

            idx_max = int(
                np.argmax(
                    delta
                )
            )

            q_delta_min = float(
                q_pontos[
                    idx_min
                ]
            )

            q_delta_max = float(
                q_pontos[
                    idx_max
                ]
            )

            classe = classificar_curva(
                equivalente,
                delta_min,
                delta_max,
            )

            classes_cenario.append(
                classe
            )

            deltas_min_cenario.append(
                delta_min
            )

            deltas_max_cenario.append(
                delta_max
            )

            sinais = sinal_delta(
                delta
            )

            indices_global = np.where(
                sinais
                <
                0
            )[0]

            indices_local = np.where(
                sinais
                >
                0
            )[0]

            q_primeiro_global = (
                float(
                    q_pontos[
                        indices_global[
                            0
                        ]
                    ]
                )
                if len(
                    indices_global
                )
                >
                0
                else np.nan
            )

            q_ultimo_global = (
                float(
                    q_pontos[
                        indices_global[
                            -1
                        ]
                    ]
                )
                if len(
                    indices_global
                )
                >
                0
                else np.nan
            )

            q_primeiro_local = (
                float(
                    q_pontos[
                        indices_local[
                            0
                        ]
                    ]
                )
                if len(
                    indices_local
                )
                >
                0
                else np.nan
            )

            q_ultimo_local = (
                float(
                    q_pontos[
                        indices_local[
                            -1
                        ]
                    ]
                )
                if len(
                    indices_local
                )
                >
                0
                else np.nan
            )

            # ---------------------------------------------------------------
            # Área entre curvas: resumo descritivo, NÃO critério de classe.
            # Para ECDFs, integral(F_L - F_G)dq = E[Q_G] - E[Q_L].
            # ---------------------------------------------------------------

            if len(
                q_pontos
            ) >= 2:

                larguras = np.diff(
                    q_pontos
                )

                delta_esquerda = delta[
                    :-1
                ]

                area_assinada = float(
                    np.sum(
                        delta_esquerda
                        *
                        larguras
                    )
                )

                area_local = float(
                    np.sum(
                        np.maximum(
                            delta_esquerda,
                            0.0,
                        )
                        *
                        larguras
                    )
                )

                area_global = float(
                    np.sum(
                        np.maximum(
                            -delta_esquerda,
                            0.0,
                        )
                        *
                        larguras
                    )
                )

                amplitude_q = float(
                    q_pontos[
                        -1
                    ]
                    -
                    q_pontos[
                        0
                    ]
                )

            else:

                area_assinada = 0.0
                area_local = 0.0
                area_global = 0.0
                amplitude_q = 0.0

            # Checagem algébrica da identidade da área.
            diferenca_medias = float(
                np.mean(
                    q_global
                )
                -
                np.mean(
                    q_local
                )
            )

            erro_area = abs(
                area_assinada
                -
                diferenca_medias
            )

            intervalos = construir_intervalos(
                cenario,
                grupo,
                q_pontos,
                delta,
            )

            resultados_intervalos.extend(
                intervalos
            )

            n_intervalos_global = sum(
                1
                for r in intervalos
                if r[
                    "Favorabilidade"
                ]
                ==
                "global"
            )

            n_intervalos_local = sum(
                1
                for r in intervalos
                if r[
                    "Favorabilidade"
                ]
                ==
                "local"
            )

            # Curva amostrada apenas para inspeção/plot.
            pooled = np.concatenate(
                [
                    q_global,
                    q_local,
                ]
            )

            q_vis = np.quantile(
                pooled,
                QUANTIS_VISUALIZACAO,
            )

            q_vis = np.unique(
                q_vis
            )

            hg_vis = ecdf_em_pontos(
                qg_sort,
                q_vis,
            )

            hl_vis = ecdf_em_pontos(
                ql_sort,
                q_vis,
            )

            for qv, hgv, hlv in zip(
                q_vis,
                hg_vis,
                hl_vis,
            ):

                resultados_curva_amostrada.append(
                    {
                        "Cenario":
                            cenario,

                        "Grupo":
                            grupo,

                        "q":
                            float(
                                qv
                            ),

                        "H_Global":
                            float(
                                hgv
                            ),

                        "H_Local":
                            float(
                                hlv
                            ),

                        "Delta_Local_Menos_Global":
                            float(
                                hlv
                                -
                                hgv
                            ),
                    }
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

                    "Tamanho_Grupo":
                        d_k,

                    "D_Total":
                        D,

                    "Amplificacao_D_sobre_d":
                        float(
                            D
                            /
                            d_k
                        ),

                    "N_Validacao":
                        n_total,

                    "N_Estados_Elegiveis":
                        n_elegivel,

                    "Tau_Global":
                        tau_global,

                    "Tau_Local_Fusao":
                        tau_local,

                    "Q_Global_Media":
                        float(
                            np.mean(
                                q_global
                            )
                        ),

                    "Q_Local_Media":
                        float(
                            np.mean(
                                q_local
                            )
                        ),

                    "Delta_CDF_Min":
                        delta_min,

                    "Q_no_Delta_CDF_Min":
                        q_delta_min,

                    "Delta_CDF_Max":
                        delta_max,

                    "Q_no_Delta_CDF_Max":
                        q_delta_max,

                    "Q_Primeiro_Global":
                        q_primeiro_global,

                    "Q_Ultimo_Global":
                        q_ultimo_global,

                    "Q_Primeiro_Local":
                        q_primeiro_local,

                    "Q_Ultimo_Local":
                        q_ultimo_local,

                    "N_Intervalos_Global":
                        n_intervalos_global,

                    "N_Intervalos_Local":
                        n_intervalos_local,

                    "Area_Assinada_Delta":
                        area_assinada,

                    "Area_Favoravel_Local":
                        area_local,

                    "Area_Favoravel_Global":
                        area_global,

                    "Amplitude_Q_Observada":
                        amplitude_q,

                    "Diferenca_Medias_QG_Menos_QL":
                        diferenca_medias,

                    "Erro_Identidade_Area":
                        erro_area,

                    "Equivalente_Distribuicao":
                        equivalente,

                    "Classificacao_Curva":
                        classe,
                }
            )

        contagens = pd.Series(
            classes_cenario
        ).value_counts()

        classe_cenario = classificar_cenario(
            classes_cenario
        )

        resultados_cenarios.append(
            {
                "Cenario":
                    cenario,

                "Alpha_Grupo":
                    alpha,

                "Beta_Global":
                    beta,

                "N_Grupos":
                    len(
                        grupos
                    ),

                "N_Estados_Elegiveis":
                    n_elegivel,

                "N_Grupos_Local_Toda_Faixa":
                    int(
                        contagens.get(
                            "favorabilidade_local_toda_faixa",
                            0,
                        )
                    ),

                "N_Grupos_Global_Toda_Faixa":
                    int(
                        contagens.get(
                            "favorabilidade_global_toda_faixa",
                            0,
                        )
                    ),

                "N_Grupos_Dependentes_Intensidade":
                    int(
                        contagens.get(
                            "dependente_intensidade",
                            0,
                        )
                    ),

                "N_Grupos_Equivalentes":
                    int(
                        contagens.get(
                            "equivalente",
                            0,
                        )
                    ),

                "Menor_Delta_CDF_Entre_Grupos":
                    float(
                        np.min(
                            deltas_min_cenario
                        )
                    ),

                "Maior_Delta_CDF_Entre_Grupos":
                    float(
                        np.max(
                            deltas_max_cenario
                        )
                    ),

                "Classificacao_Cenario":
                    classe_cenario,
            }
        )

        print(
            f"Estados elegíveis:                    {n_elegivel}/{n_total}"
        )

        print(
            f"Grupos:                               {len(grupos)}"
        )

        print(
            f"Curvas local em toda faixa:           "
            f"{sum(c == 'favorabilidade_local_toda_faixa' for c in classes_cenario)}"
        )

        print(
            f"Curvas global em toda faixa:          "
            f"{sum(c == 'favorabilidade_global_toda_faixa' for c in classes_cenario)}"
        )

        print(
            f"Curvas dependentes da intensidade:    "
            f"{sum(c == 'dependente_intensidade' for c in classes_cenario)}"
        )

        print(
            f"Curvas equivalentes:                  "
            f"{sum(c == 'equivalente' for c in classes_cenario)}"
        )

        print(
            f"Delta CDF mínimo entre grupos:         {np.min(deltas_min_cenario):.6f}"
        )

        print(
            f"Delta CDF máximo entre grupos:         {np.max(deltas_max_cenario):.6f}"
        )

        print(
            f"Classificação do cenário:              {classe_cenario}"
        )

    df_grupos = pd.DataFrame(
        resultados_grupos
    )

    df_cenarios = pd.DataFrame(
        resultados_cenarios
    )

    df_intervalos = pd.DataFrame(
        resultados_intervalos
    )

    df_curva = pd.DataFrame(
        resultados_curva_amostrada
    )

    df_grupos.to_csv(
        ARQUIVO_GRUPOS,
        index=False,
    )

    df_cenarios.to_csv(
        ARQUIVO_CENARIOS,
        index=False,
    )

    df_intervalos.to_csv(
        ARQUIVO_INTERVALOS,
        index=False,
    )

    df_curva.to_csv(
        ARQUIVO_CURVA_AMOSTRADA,
        index=False,
    )

    print()
    print("=" * 138)
    print("RESUMO FINAL")
    print("=" * 138)
    print()

    print(
        df_cenarios[
            [
                "Cenario",
                "N_Grupos",
                "N_Grupos_Local_Toda_Faixa",
                "N_Grupos_Global_Toda_Faixa",
                "N_Grupos_Dependentes_Intensidade",
                "N_Grupos_Equivalentes",
                "Menor_Delta_CDF_Entre_Grupos",
                "Maior_Delta_CDF_Entre_Grupos",
                "Classificacao_Cenario",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("CHECAGEM:")
    print(
        f"Maior erro da identidade integral(F_L-F_G)dq = E[Q_G]-E[Q_L]: "
        f"{df_grupos['Erro_Identidade_Area'].max():.6e}"
    )

    print()
    print("IMPORTANTE:")
    print(
        "1) Delta(q)>0 significa favorabilidade local CONDICIONAL à quantidade positiva de evidência q."
    )
    print(
        "2) Delta(q)<0 significa favorabilidade global CONDICIONAL à quantidade positiva de evidência q."
    )
    print(
        "3) Se a curva muda de sinal, o normal-only identifica dependência da intensidade; não força um vencedor."
    )
    print(
        "4) A classificação usa todos os pontos exatos das ECDFs; o arquivo de curva amostrada é apenas para visualização."
    )
    print(
        "5) Nenhum resultado de ataque participa deste diagnóstico."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)
    print(ARQUIVO_INTERVALOS)
    print(ARQUIVO_CURVA_AMOSTRADA)


if __name__ == "__main__":
    main()
