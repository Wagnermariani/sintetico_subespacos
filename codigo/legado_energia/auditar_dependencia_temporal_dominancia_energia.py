import os
import math

import numpy as np
import pandas as pd


# ============================================================================
# AUDITORIA DE DEPENDÊNCIA TEMPORAL — DOMINÂNCIA DE ENERGIA
#
# Objetivo:
# medir quanto a dependência temporal reduz a quantidade efetiva de
# informação na validação normal antes de transformar a banda i.i.d. da
# auditoria anterior em qualquer regra de classificação.
#
# IMPORTANTE:
# - NÃO usa ataques, P01/P02, m, delta, direção ou rótulos de teste;
# - NÃO define ainda a V4 final;
# - estima a dependência diretamente das séries normais de escores;
# - calcula tamanho efetivo aproximado n_eff via tempo integrado de
#   autocorrelação (IAT), apenas como diagnóstico;
# - compara a escala DKW i.i.d. com uma escala conservadora baseada em n_eff.
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

ARQUIVO_MARGEM_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_margem_dominancia_energia_v3",
    "margem_dominancia_energia_v3_por_grupo.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_dependencia_temporal_energia",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "dependencia_temporal_energia_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "dependencia_temporal_energia_por_cenario.csv",
)


FPR_ALVO = 0.01
EPS = 1e-12

MAX_LAG = 3000
LIMIAR_ACF_DECORRELACAO = 0.05
JANELA_ESTAVEL = 20

ALPHA_REFERENCIA = 0.05


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


def autocorrelacao_fft(
    x,
    max_lag,
):
    """
    ACF normalizada calculada por FFT.

    Usa denominador aproximadamente não viesado por lag:
        cov(h) / (n-h)
    e normaliza pela covariância em lag zero.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    n = len(
        x
    )

    if n < 3:

        raise RuntimeError(
            "Série curta demais para ACF."
        )

    x = (
        x
        -
        np.mean(
            x
        )
    )

    var0 = float(
        np.dot(
            x,
            x
        )
        /
        n
    )

    if var0 <= EPS:

        return np.ones(
            min(
                max_lag,
                n - 1
            )
            +
            1,
            dtype=np.float64,
        )

    tamanho_fft = 1

    while tamanho_fft < (
        2
        *
        n
    ):
        tamanho_fft *= 2

    fft = np.fft.rfft(
        x,
        n=tamanho_fft,
    )

    acov = np.fft.irfft(
        fft
        *
        np.conjugate(
            fft
        ),
        n=tamanho_fft,
    )[
        :
        n
    ]

    denominadores = np.arange(
        n,
        0,
        -1,
        dtype=np.float64,
    )

    acov = (
        acov
        /
        denominadores
    )

    limite = min(
        max_lag,
        n - 1,
    )

    acf = (
        acov[
            :
            limite
            +
            1
        ]
        /
        acov[
            0
        ]
    )

    acf[
        0
    ] = 1.0

    return acf


def tempo_integrado_autocorrelacao(
    acf,
):
    """
    Estimativa simples/conservadora por sequência positiva inicial:
    soma os lags enquanto a ACF permanecer positiva.

    IAT = 1 + 2 * sum rho(h)

    O objetivo aqui é diagnóstico de escala, não inferência final.
    """

    soma = 0.0
    ultimo_lag = 0

    for lag in range(
        1,
        len(
            acf
        )
    ):

        rho = float(
            acf[
                lag
            ]
        )

        if not np.isfinite(
            rho
        ):
            break

        if rho <= 0.0:
            break

        soma += rho
        ultimo_lag = lag

    iat = (
        1.0
        +
        2.0
        *
        soma
    )

    iat = max(
        1.0,
        float(
            iat
        ),
    )

    return (
        iat,
        ultimo_lag,
    )


def lag_decorrelacao(
    acf,
    limiar=0.05,
    janela=20,
):
    """
    Primeiro lag a partir do qual |ACF| permanece abaixo do limiar
    durante 'janela' lags consecutivos.

    Se não encontrar no horizonte analisado, retorna o maior lag.
    """

    abs_acf = np.abs(
        acf
    )

    if len(
        abs_acf
    ) <= 1:

        return 0

    ultimo_inicio = (
        len(
            abs_acf
        )
        -
        janela
    )

    for lag in range(
        1,
        max(
            1,
            ultimo_inicio
            +
            1
        ),
    ):

        trecho = abs_acf[
            lag:
            lag
            +
            janela
        ]

        if len(
            trecho
        ) < janela:
            break

        if np.all(
            trecho
            <
            limiar
        ):

            return lag

    return len(
        abs_acf
    ) - 1


def tolerancia_dkw_referencia(
    n,
    alpha=0.05,
):
    """
    Mesma escala usada na auditoria anterior:
    duas bandas DKW marginais + união de eventos.

    ATENÇÃO:
    DKW clássico é i.i.d. Aqui o valor baseado em n_eff é apenas
    uma referência conservadora de ordem de grandeza, não um IC formal.
    """

    n = float(
        n
    )

    if n <= 0:

        raise ValueError(
            "n deve ser positivo."
        )

    eps = math.sqrt(
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
        eps,
        2.0
        *
        eps,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 126)
    print("AUDITORIA DE DEPENDÊNCIA TEMPORAL — DOMINÂNCIA DE ENERGIA")
    print("=" * 126)

    print()
    print(
        "Somente dados normais. "
        "Nenhum ataque ou parâmetro de P01/P02 é utilizado."
    )

    print(
        "Esta etapa mede dependência temporal antes de definir qualquer V4."
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

    margens = pd.read_csv(
        ARQUIVO_MARGEM_GRUPOS
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
                f"{cenario}: desvio padrão praticamente zero."
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

        D = Z2_val.shape[
            1
        ]

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

        # ---------------------------------------------------------------
        # A dependência é estimada nas séries COMPLETAS de escores normais,
        # sem remover os 1-2% de estados acima dos limiares. Isso preserva
        # a ordem temporal regular.
        # ---------------------------------------------------------------

        acf_global = autocorrelacao_fft(
            score_global,
            MAX_LAG,
        )

        iat_global, lag_pos_global = (
            tempo_integrado_autocorrelacao(
                acf_global
            )
        )

        lag05_global = lag_decorrelacao(
            acf_global,
            LIMIAR_ACF_DECORRELACAO,
            JANELA_ESTAVEL,
        )

        n_total = len(
            score_global
        )

        n_eff_global = (
            n_total
            /
            iat_global
        )

        registros_cenario = []

        margem_cenario = margens[
            margens[
                "Cenario"
            ]
            ==
            cenario
        ].copy()

        if len(
            margem_cenario
        ) != len(
            grupos
        ):

            raise RuntimeError(
                f"{cenario}: número de grupos não coincide com auditoria V3."
            )

        margem_por_grupo = (
            margem_cenario
            .set_index(
                "Grupo"
            )
        )

        for grupo in grupos:

            score_local = scores_locais[
                grupo
            ]

            acf_local = autocorrelacao_fft(
                score_local,
                MAX_LAG,
            )

            iat_local, lag_pos_local = (
                tempo_integrado_autocorrelacao(
                    acf_local
                )
            )

            lag05_local = lag_decorrelacao(
                acf_local,
                LIMIAR_ACF_DECORRELACAO,
                JANELA_ESTAVEL,
            )

            n_eff_local = (
                n_total
                /
                iat_local
            )

            # Critério conservador: usar o menor n_eff entre global e local.
            n_eff_conservador = max(
                1.0,
                min(
                    n_eff_global,
                    n_eff_local,
                ),
            )

            _, tol_iid = (
                tolerancia_dkw_referencia(
                    n_total,
                    ALPHA_REFERENCIA,
                )
            )

            _, tol_neff = (
                tolerancia_dkw_referencia(
                    n_eff_conservador,
                    ALPHA_REFERENCIA,
                )
            )

            linha_margem = margem_por_grupo.loc[
                grupo
            ]

            sup_local = float(
                linha_margem[
                    "Sup_Local"
                ]
            )

            sup_global = float(
                linha_margem[
                    "Sup_Global"
                ]
            )

            registro = {
                "Cenario":
                    cenario,

                "Grupo":
                    grupo,

                "N_Validacao":
                    n_total,

                "IAT_Global":
                    float(
                        iat_global
                    ),

                "IAT_Local":
                    float(
                        iat_local
                    ),

                "N_Efetivo_Global_Aprox":
                    float(
                        n_eff_global
                    ),

                "N_Efetivo_Local_Aprox":
                    float(
                        n_eff_local
                    ),

                "N_Efetivo_Conservador_Aprox":
                    float(
                        n_eff_conservador
                    ),

                "Lag_Ultimo_ACF_Positivo_Global":
                    int(
                        lag_pos_global
                    ),

                "Lag_Ultimo_ACF_Positivo_Local":
                    int(
                        lag_pos_local
                    ),

                "Lag_Decorrelacao_005_Global":
                    int(
                        lag05_global
                    ),

                "Lag_Decorrelacao_005_Local":
                    int(
                        lag05_local
                    ),

                "Tolerancia_DKW_iid_Referencia":
                    float(
                        tol_iid
                    ),

                "Tolerancia_DKW_n_eff_Referencia":
                    float(
                        tol_neff
                    ),

                "Sup_Local":
                    sup_local,

                "Sup_Global":
                    sup_global,

                "Razao_Sup_Local_sobre_Tol_n_eff":
                    (
                        sup_local
                        /
                        tol_neff
                    ),

                "Razao_Sup_Global_sobre_Tol_n_eff":
                    (
                        sup_global
                        /
                        tol_neff
                    ),
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

        resultados_cenarios.append(
            {
                "Cenario":
                    cenario,

                "N_Grupos":
                    len(
                        grupos
                    ),

                "IAT_Global":
                    float(
                        iat_global
                    ),

                "N_Efetivo_Global_Aprox":
                    float(
                        n_eff_global
                    ),

                "Lag_Decorrelacao_005_Global":
                    int(
                        lag05_global
                    ),

                "IAT_Local_Max_Entre_Grupos":
                    float(
                        rc[
                            "IAT_Local"
                        ].max()
                    ),

                "N_Efetivo_Conservador_Min_Entre_Grupos":
                    float(
                        rc[
                            "N_Efetivo_Conservador_Aprox"
                        ].min()
                    ),

                "Lag_Decorrelacao_005_Max_Entre_Grupos":
                    int(
                        max(
                            lag05_global,
                            int(
                                rc[
                                    "Lag_Decorrelacao_005_Local"
                                ].max()
                            ),
                        )
                    ),

                "Tolerancia_DKW_iid_Referencia":
                    float(
                        rc[
                            "Tolerancia_DKW_iid_Referencia"
                        ].iloc[
                            0
                        ]
                    ),

                "Tolerancia_DKW_n_eff_Max_Entre_Grupos":
                    float(
                        rc[
                            "Tolerancia_DKW_n_eff_Referencia"
                        ].max()
                    ),

                "Pior_Sup_Global_Entre_Grupos":
                    float(
                        rc[
                            "Sup_Global"
                        ].max()
                    ),

                "Menor_Sup_Local_Entre_Grupos":
                    float(
                        rc[
                            "Sup_Local"
                        ].min()
                    ),

                "Pior_Razao_Sup_Global_sobre_Tol_n_eff":
                    float(
                        rc[
                            "Razao_Sup_Global_sobre_Tol_n_eff"
                        ].max()
                    ),

                "Menor_Razao_Sup_Local_sobre_Tol_n_eff":
                    float(
                        rc[
                            "Razao_Sup_Local_sobre_Tol_n_eff"
                        ].min()
                    ),
            }
        )

        print(
            f"IAT global:                         {iat_global:.3f}"
        )

        print(
            f"n_eff global aproximado:            {n_eff_global:.2f}"
        )

        print(
            f"lag |ACF|<0.05 global:              {lag05_global}"
        )

        print(
            f"IAT local máximo:                   {rc['IAT_Local'].max():.3f}"
        )

        print(
            f"n_eff conservador mínimo:           {rc['N_Efetivo_Conservador_Aprox'].min():.2f}"
        )

        print(
            f"tolerância iid referência:          {rc['Tolerancia_DKW_iid_Referencia'].iloc[0]:.4f}"
        )

        print(
            f"tolerância n_eff máxima referência: {rc['Tolerancia_DKW_n_eff_Referencia'].max():.4f}"
        )

        print(
            f"pior Sup_Global:                    {rc['Sup_Global'].max():.4f}"
        )

        print(
            f"menor Sup_Local:                    {rc['Sup_Local'].min():.4f}"
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
    print("RESUMO POR CENÁRIO")
    print("=" * 126)
    print()

    colunas = [
        "Cenario",
        "N_Grupos",
        "IAT_Global",
        "N_Efetivo_Global_Aprox",
        "IAT_Local_Max_Entre_Grupos",
        "N_Efetivo_Conservador_Min_Entre_Grupos",
        "Lag_Decorrelacao_005_Max_Entre_Grupos",
        "Tolerancia_DKW_iid_Referencia",
        "Tolerancia_DKW_n_eff_Max_Entre_Grupos",
        "Pior_Sup_Global_Entre_Grupos",
        "Menor_Sup_Local_Entre_Grupos",
        "Pior_Razao_Sup_Global_sobre_Tol_n_eff",
        "Menor_Razao_Sup_Local_sobre_Tol_n_eff",
    ]

    print(
        df_cenarios[
            colunas
        ].to_string(
            index=False
        )
    )

    print()
    print("INTERPRETAÇÃO:")
    print(
        "- IAT alto / n_eff baixo confirma que a validação não deve ser tratada como 10.000 observações independentes."
    )
    print(
        "- A tolerância baseada em n_eff é somente uma escala diagnóstica; não é ainda a regra final."
    )
    print(
        "- Se a vantagem local continuar muito maior que essa escala enquanto a violação global permanecer muito menor, "
        "teremos justificativa para a próxima etapa: bootstrap em blocos."
    )

    print()
    print("Resultados:")
    print(ARQUIVO_GRUPOS)
    print(ARQUIVO_CENARIOS)


if __name__ == "__main__":
    main()
