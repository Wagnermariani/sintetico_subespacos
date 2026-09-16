import os

import numpy as np
import pandas as pd

from scipy.linalg import eigvalsh
from sklearn.covariance import LedoitWolf


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
    "diagnostico_normal_only",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_GRUPOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_por_grupo.csv",
)

ARQUIVO_CENARIOS = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_por_cenario.csv",
)


FPR_ALVO = 0.01


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
    print("DIAGNÓSTICO NORMAL-ONLY — EXPERIMENTO 4: FRONTEIRA ESTRUTURAL")
    print("=" * 108)

    print()
    print(
        "Nenhum ataque, m, delta ou rótulo de teste é utilizado."
    )

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    if len(manifesto) != 12:

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

    # ========================================================================
    # CENÁRIOS
    # ========================================================================

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
        print("=" * 108)
        print(
            f"{cenario} | "
            f"alpha={alpha:.2f} | "
            f"beta={beta:.2f}"
        )
        print("=" * 108)

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

        # --------------------------------------------------------------------
        # GARANTIA NORMAL-ONLY
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # GRUPOS DESCOBERTOS SOMENTE NO NORMAL
        # --------------------------------------------------------------------

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
            ]
            .tolist()
        )

        if len(features) != 400:

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

        tamanhos_grupos = []

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

            idx = np.asarray(
                [
                    indice_feature[
                        feature
                    ]
                    for feature in features_grupo
                ],
                dtype=np.int64,
            )

            indices_por_grupo[
                grupo
            ] = idx

            tamanhos_grupos.append(
                len(idx)
            )

        # --------------------------------------------------------------------
        # NORMALIZAÇÃO
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

        # ====================================================================
        # MAHALANOBIS GLOBAL
        # ====================================================================

        modelo_global = LedoitWolf(
            store_precision=True,
            assume_centered=False,
        )

        modelo_global.fit(
            X_train
        )

        score_global_val = modelo_global.mahalanobis(
            X_val
        )

        tau_global = float(
            np.quantile(
                score_global_val,
                1.0 - FPR_ALVO,
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

        # ====================================================================
        # MAHALANOBIS LOCAIS
        # ====================================================================

        modelos_locais = {}

        scores_locais_val = {}

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            modelo_local = LedoitWolf(
                store_precision=True,
                assume_centered=False,
            )

            modelo_local.fit(
                X_train[
                    :,
                    idx
                ]
            )

            modelos_locais[
                grupo
            ] = modelo_local

            scores_locais_val[
                grupo
            ] = modelo_local.mahalanobis(
                X_val[
                    :,
                    idx
                ]
            )

        # --------------------------------------------------------------------
        # FUSÃO LOCAL = MÁXIMO
        # --------------------------------------------------------------------

        matriz_scores_locais = np.column_stack(
            [
                scores_locais_val[
                    grupo
                ]
                for grupo in grupos
            ]
        )

        score_fusao_local = np.max(
            matriz_scores_locais,
            axis=1,
        )

        tau_local = float(
            np.quantile(
                score_fusao_local,
                1.0 - FPR_ALVO,
            )
        )

        # ====================================================================
        # DIAGNÓSTICO POR GRUPO
        # ====================================================================

        burdens = []
        fatores_pessimistas = []
        fatores_medianos = []
        fatores_otimistas = []

        features_fator_maior_1 = 0

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            tamanho_grupo = len(
                idx
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
            # BURDEN RATIO
            # ---------------------------------------------------------------

            burden_ratio = (
                gap_global
                /
                gap_local
            )

            # ---------------------------------------------------------------
            # RELAÇÃO ENTRE MÉTRICAS QUADRÁTICAS
            # ---------------------------------------------------------------

            precisao_global_bloco = (
                modelo_global
                .precision_[
                    np.ix_(
                        idx,
                        idx,
                    )
                ]
            )

            precisao_local = (
                modelos_locais[
                    grupo
                ]
                .precision_
            )

            precisao_global_bloco = (
                precisao_global_bloco
                +
                precisao_global_bloco.T
            ) / 2.0

            precisao_local = (
                precisao_local
                +
                precisao_local.T
            ) / 2.0

            autovalores = eigvalsh(
                precisao_global_bloco,
                precisao_local,
            )

            if np.any(
                autovalores
                <=
                0
            ):

                raise RuntimeError(
                    f"{cenario} {grupo}: autovalor não positivo."
                )

            r_min = float(
                np.min(
                    autovalores
                )
            )

            r_med = float(
                np.median(
                    autovalores
                )
            )

            r_max = float(
                np.max(
                    autovalores
                )
            )

            # ---------------------------------------------------------------
            # FATOR DE VANTAGEM
            # ---------------------------------------------------------------

            fator_pessimista = (
                burden_ratio
                /
                r_max
            )

            fator_mediano = (
                burden_ratio
                /
                r_med
            )

            fator_otimista = (
                burden_ratio
                /
                r_min
            )

            burdens.append(
                burden_ratio
            )

            fatores_pessimistas.append(
                fator_pessimista
            )

            fatores_medianos.append(
                fator_mediano
            )

            fatores_otimistas.append(
                fator_otimista
            )

            if fator_pessimista > 1.0:

                features_fator_maior_1 += (
                    tamanho_grupo
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
                        tamanho_grupo,

                    "N_Grupos_Descobertos":
                        len(grupos),

                    "Tau_Global":
                        tau_global,

                    "Media_Global":
                        media_global,

                    "Gap_Global":
                        gap_global,

                    "Tau_Local_Fusao":
                        tau_local,

                    "Media_Local":
                        media_local,

                    "Gap_Local":
                        gap_local,

                    "Burden_Ratio":
                        burden_ratio,

                    "Razao_Quadratica_Min":
                        r_min,

                    "Razao_Quadratica_Mediana":
                        r_med,

                    "Razao_Quadratica_Max":
                        r_max,

                    "Fator_Vantagem_Pessimista":
                        fator_pessimista,

                    "Fator_Vantagem_Mediano":
                        fator_mediano,

                    "Fator_Vantagem_Otimista":
                        fator_otimista,
                }
            )

        # ====================================================================
        # RESUMO DO CENÁRIO
        # ====================================================================

        burdens_arr = np.asarray(
            burdens,
            dtype=np.float64,
        )

        fatores_pess_arr = np.asarray(
            fatores_pessimistas,
            dtype=np.float64,
        )

        fatores_med_arr = np.asarray(
            fatores_medianos,
            dtype=np.float64,
        )

        fatores_ot_arr = np.asarray(
            fatores_otimistas,
            dtype=np.float64,
        )

        n_grupos_fator_maior_1 = int(
            np.sum(
                fatores_pess_arr
                >
                1.0
            )
        )

        fracao_features_fator_maior_1 = (
            features_fator_maior_1
            /
            len(features)
        )

        resultados_cenarios.append(
            {
                "Cenario":
                    cenario,

                "Alpha_Grupo":
                    alpha,

                "Beta_Global":
                    beta,

                "N_Grupos_Descobertos":
                    len(grupos),

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

                "Tau_Global":
                    tau_global,

                "Media_Global":
                    media_global,

                "Gap_Global":
                    gap_global,

                "Tau_Local_Fusao":
                    tau_local,

                "Burden_Min":
                    float(
                        np.min(
                            burdens_arr
                        )
                    ),

                "Burden_Mediana":
                    float(
                        np.median(
                            burdens_arr
                        )
                    ),

                "Burden_Max":
                    float(
                        np.max(
                            burdens_arr
                        )
                    ),

                "Fator_Pessimista_Min":
                    float(
                        np.min(
                            fatores_pess_arr
                        )
                    ),

                "Fator_Pessimista_Mediana":
                    float(
                        np.median(
                            fatores_pess_arr
                        )
                    ),

                "Fator_Pessimista_Max":
                    float(
                        np.max(
                            fatores_pess_arr
                        )
                    ),

                "Fator_Mediano_Mediana":
                    float(
                        np.median(
                            fatores_med_arr
                        )
                    ),

                "Fator_Otimista_Mediana":
                    float(
                        np.median(
                            fatores_ot_arr
                        )
                    ),

                "Grupos_Fator_Pessimista_Maior_1":
                    n_grupos_fator_maior_1,

                "Fracao_Grupos_Fator_Maior_1":
                    (
                        n_grupos_fator_maior_1
                        /
                        len(grupos)
                    ),

                "Fracao_Features_Fator_Maior_1":
                    fracao_features_fator_maior_1,
            }
        )

        print(
            f"Grupos descobertos:                {len(grupos)}"
        )

        print(
            f"Tamanhos:                          "
            f"min={min(tamanhos_grupos)} | "
            f"med={np.median(tamanhos_grupos):.1f} | "
            f"max={max(tamanhos_grupos)}"
        )

        print(
            f"Gap global:                        {gap_global:.6f}"
        )

        print(
            f"Tau local fusão:                   {tau_local:.6f}"
        )

        print(
            f"Burden mediano:                    "
            f"{np.median(burdens_arr):.6f}"
        )

        print(
            f"Fator pessimista:                  "
            f"min={np.min(fatores_pess_arr):.6f} | "
            f"med={np.median(fatores_pess_arr):.6f} | "
            f"max={np.max(fatores_pess_arr):.6f}"
        )

        print(
            "Grupos com fator pessimista > 1:   "
            f"{n_grupos_fator_maior_1} / {len(grupos)}"
        )

        print(
            "Features em grupos com fator > 1:  "
            f"{100.0 * fracao_features_fator_maior_1:.2f}%"
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
    # RESUMO
    # =========================================================================

    print()
    print("=" * 108)
    print("RESUMO NORMAL-ONLY DA FRONTEIRA ESTRUTURAL")
    print("=" * 108)

    print()

    print(
        df_cenarios[
            [
                "Cenario",
                "Alpha_Grupo",
                "Beta_Global",
                "N_Grupos_Descobertos",
                "Tamanho_Mediana",
                "Burden_Mediana",
                "Fator_Pessimista_Min",
                "Fator_Pessimista_Mediana",
                "Fracao_Features_Fator_Maior_1",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Este diagnóstico usa somente treino normal, validação normal "
        "e os grupos descobertos no normal."
    )

    print(
        "Nenhum mapa verdadeiro, ataque, m, delta ou rótulo de teste "
        "é utilizado no cálculo dos indicadores."
    )

    print()
    print(
        "Resultados:"
    )

    print(
        ARQUIVO_GRUPOS
    )

    print(
        ARQUIVO_CENARIOS
    )


if __name__ == "__main__":
    main()