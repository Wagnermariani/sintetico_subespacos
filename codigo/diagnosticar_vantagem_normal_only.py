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
    "confirmacao",
    "manifesto_realizacoes.csv",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

DIRETORIO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "grupos_confirmatorios",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "diagnostico_normal_only",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
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

    print("=" * 100)
    print("DIAGNÓSTICO NORMAL-ONLY — POSSÍVEL VANTAGEM DA ANÁLISE LOCAL")
    print("=" * 100)

    print()
    print(
        "IMPORTANTE: este programa não lê ataques e não usa rótulos "
        "de ataque para calcular nenhuma medida."
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

    resultados_realizacoes = []

    # ========================================================================
    # REALIZAÇÕES
    # ========================================================================

    for _, linha_manifesto in manifesto.iterrows():

        realizacao = str(
            linha_manifesto[
                "Realizacao"
            ]
        )

        print()
        print("=" * 100)
        print(realizacao)
        print("=" * 100)

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
                f"{realizacao}: treino contém ataque."
            )

        if not (
            validacao[
                "Attack"
            ]
            ==
            0
        ).all():

            raise RuntimeError(
                f"{realizacao}: validação contém ataque."
            )

        arquivo_grupos = os.path.join(
            DIRETORIO_GRUPOS,
            realizacao,
            "mapa_grupos_descobertos.csv",
        )

        grupos_df = pd.read_csv(
            arquivo_grupos
        )

        features = sorted(
            grupos_df[
                "Feature"
            ]
            .tolist()
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

        if len(features) != 400:

            raise RuntimeError(
                f"{realizacao}: esperadas 400 features."
            )

        if len(grupos) != 20:

            raise RuntimeError(
                f"{realizacao}: esperados 20 grupos."
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
                f"{realizacao}: gap global não positivo."
            )

        # ====================================================================
        # MODELOS LOCAIS
        # ====================================================================

        modelos_locais = {}

        indices_por_grupo = {}

        scores_locais_val = {}

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
        # FUSÃO LOCAL = MÁXIMO ENTRE OS GRUPOS
        # LIMIAR TAMBÉM DEFINIDO APENAS NA VALIDAÇÃO NORMAL
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

        fatores_pessimistas = []

        fatores_medianos = []

        fatores_otimistas = []

        burdens = []

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

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
                    f"{realizacao} {grupo}: gap local não positivo."
                )

            # ================================================================
            # BURDEN RATIO
            #
            # > 1:
            # o score global precisa percorrer uma distância maior desde
            # seu comportamento normal médio até o limiar.
            # ================================================================

            burden_ratio = (
                gap_global
                /
                gap_local
            )

            # ================================================================
            # RELAÇÃO ENTRE AS MÉTRICAS QUADRÁTICAS
            #
            # Para uma perturbação confinada ao grupo:
            #
            # q_global / q_local
            #
            # varia conforme a direção da perturbação.
            #
            # Os autovalores generalizados dão os extremos possíveis.
            # ================================================================

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
                    f"{realizacao} {grupo}: autovalor não positivo."
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

            # ================================================================
            # FATOR DE VANTAGEM ESTRUTURAL
            #
            # Aproximação:
            #
            # amplitude² necessária no global
            # --------------------------------
            # amplitude² necessária no local
            #
            # = burden_ratio / r
            #
            # > 1 favorece o local.
            #
            # "Pessimista" usa r_max, isto é, a direção mais favorável
            # ao score global.
            # ================================================================

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

            resultados_grupos.append(
                {
                    "Realizacao":
                        realizacao,

                    "Grupo":
                        grupo,

                    "D_Global":
                        len(features),

                    "D_Local":
                        len(idx),

                    "Tau_Global":
                        tau_global,

                    "Media_Global":
                        media_global,

                    "Gap_Global":
                        gap_global,

                    "Tau_Local_Fusao":
                        tau_local,

                    "Media_Score_Grupo":
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
        # RESUMO DA REALIZAÇÃO
        # ====================================================================

        resultados_realizacoes.append(
            {
                "Realizacao":
                    realizacao,

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
                            burdens
                        )
                    ),

                "Burden_Mediana":
                    float(
                        np.median(
                            burdens
                        )
                    ),

                "Burden_Max":
                    float(
                        np.max(
                            burdens
                        )
                    ),

                "Fator_Pessimista_Min":
                    float(
                        np.min(
                            fatores_pessimistas
                        )
                    ),

                "Fator_Pessimista_Mediana":
                    float(
                        np.median(
                            fatores_pessimistas
                        )
                    ),

                "Fator_Pessimista_Max":
                    float(
                        np.max(
                            fatores_pessimistas
                        )
                    ),

                "Fator_Mediano_Mediana":
                    float(
                        np.median(
                            fatores_medianos
                        )
                    ),

                "Fator_Otimista_Mediana":
                    float(
                        np.median(
                            fatores_otimistas
                        )
                    ),

                "Grupos_Pessimista_Maior_1":
                    int(
                        np.sum(
                            np.asarray(
                                fatores_pessimistas
                            )
                            >
                            1.0
                        )
                    ),
            }
        )

        print()
        print(
            f"Tau global:       {tau_global:.6f}"
        )

        print(
            f"Média global:     {media_global:.6f}"
        )

        print(
            f"Gap global:       {gap_global:.6f}"
        )

        print(
            f"Tau local fusão:  {tau_local:.6f}"
        )

        print(
            "Burden mediano:    "
            f"{np.median(burdens):.6f}"
        )

        print(
            "Fator pessimista:  "
            f"min={np.min(fatores_pessimistas):.6f} | "
            f"med={np.median(fatores_pessimistas):.6f} | "
            f"max={np.max(fatores_pessimistas):.6f}"
        )

        print(
            "Grupos com fator pessimista > 1: "
            f"{np.sum(np.asarray(fatores_pessimistas) > 1.0)} / "
            f"{len(grupos)}"
        )

    # =========================================================================
    # SALVAMENTO
    # =========================================================================

    df_grupos = pd.DataFrame(
        resultados_grupos
    )

    df_realizacoes = pd.DataFrame(
        resultados_realizacoes
    )

    arquivo_grupos = os.path.join(
        DIRETORIO_SAIDA,
        "diagnostico_por_grupo.csv",
    )

    arquivo_realizacoes = os.path.join(
        DIRETORIO_SAIDA,
        "diagnostico_por_realizacao.csv",
    )

    df_grupos.to_csv(
        arquivo_grupos,
        index=False,
    )

    df_realizacoes.to_csv(
        arquivo_realizacoes,
        index=False,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 100)
    print("RESUMO NORMAL-ONLY")
    print("=" * 100)

    print()

    print(
        df_realizacoes[
            [
                "Realizacao",
                "Gap_Global",
                "Tau_Local_Fusao",
                "Burden_Mediana",
                "Fator_Pessimista_Min",
                "Fator_Pessimista_Mediana",
                "Grupos_Pessimista_Maior_1",
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
        "Este diagnóstico utilizou apenas treino normal, "
        "validação normal e os grupos descobertos no normal."
    )

    print(
        "Nenhum ataque, m, delta ou rótulo de teste foi utilizado."
    )

    print()
    print(
        "Resultados:"
    )

    print(
        arquivo_grupos
    )

    print(
        arquivo_realizacoes
    )


if __name__ == "__main__":
    main()