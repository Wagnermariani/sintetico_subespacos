import os
import sys

import numpy as np
import pandas as pd

from scipy.linalg import eigvalsh
from sklearn.covariance import LedoitWolf


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

CENARIOS_PERMITIDOS = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
]

if len(sys.argv) != 2:

    raise SystemExit(
        "Uso: python codigo/diagnosticar_rnn_fronteira_residuos_normal_only.py CENARIO\n"
        "Exemplo: python codigo/diagnosticar_rnn_fronteira_residuos_normal_only.py A01"
    )

CENARIO = sys.argv[1].strip().upper()

if CENARIO not in CENARIOS_PERMITIDOS:

    raise SystemExit(
        f"Cenário inválido: {CENARIO}. "
        f"Permitidos: {', '.join(CENARIOS_PERMITIDOS)}"
    )


DIRETORIO_RNN = os.path.join(
    PROJETO,
    "resultado",
    "rnn_fronteira",
    CENARIO,
    "treino_monolitico",
)

ARQUIVO_RESIDUOS_TREINO = os.path.join(
    DIRETORIO_RNN,
    "residuos_treino.npy",
)

ARQUIVO_RESIDUOS_VALIDACAO = os.path.join(
    DIRETORIO_RNN,
    "residuos_validacao.npy",
)

ARQUIVO_FEATURES = os.path.join(
    DIRETORIO_RNN,
    "ordem_features.csv",
)

ARQUIVO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "descoberta_grupos",
    CENARIO,
    "mapa_grupos_descobertos.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "rnn_fronteira",
    CENARIO,
    "diagnostico_residuos_normal_only",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_POR_GRUPO = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_residuos_por_grupo.csv",
)

ARQUIVO_RESUMO = os.path.join(
    DIRETORIO_SAIDA,
    "diagnostico_residuos_resumo.csv",
)


FPR_ALVO = 0.01


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 108)
    print(f"DIAGNÓSTICO NORMAL-ONLY DOS RESÍDUOS DA RNN — {CENARIO}")
    print("=" * 108)

    print()
    print(
        "Nenhum teste ou ataque é utilizado nesta análise."
    )

    # ------------------------------------------------------------------------
    # RESÍDUOS
    # ------------------------------------------------------------------------

    residuos_train = np.load(
        ARQUIVO_RESIDUOS_TREINO
    ).astype(
        np.float64,
        copy=False,
    )

    residuos_val = np.load(
        ARQUIVO_RESIDUOS_VALIDACAO
    ).astype(
        np.float64,
        copy=False,
    )

    if residuos_train.shape != (
        19941,
        400,
    ):

        raise RuntimeError(
            f"Shape treino inesperado: {residuos_train.shape}"
        )

    if residuos_val.shape != (
        9941,
        400,
    ):

        raise RuntimeError(
            f"Shape validação inesperado: {residuos_val.shape}"
        )

    if not np.isfinite(
        residuos_train
    ).all():

        raise RuntimeError(
            "Resíduos de treino contêm NaN ou Inf."
        )

    if not np.isfinite(
        residuos_val
    ).all():

        raise RuntimeError(
            "Resíduos de validação contêm NaN ou Inf."
        )

    # ------------------------------------------------------------------------
    # ORDEM DAS FEATURES
    # ------------------------------------------------------------------------

    ordem_df = pd.read_csv(
        ARQUIVO_FEATURES
    )

    ordem_df = ordem_df.sort_values(
        "Posicao"
    )

    features = ordem_df[
        "Feature"
    ].tolist()

    if len(
        features
    ) != 400:

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

    # ------------------------------------------------------------------------
    # GRUPOS DESCOBERTOS NO NORMAL ORIGINAL
    # ------------------------------------------------------------------------

    grupos_df = pd.read_csv(
        ARQUIVO_GRUPOS
    )

    grupos = sorted(
        grupos_df[
            "Grupo_Descoberto"
        ]
        .unique()
        .tolist()
    )

    if len(
        grupos
    ) < 1:

        raise RuntimeError(
            "Nenhum grupo descoberto."
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
                for feature in features_grupo
            ],
            dtype=np.int64,
        )

    # ========================================================================
    # MAHALANOBIS GLOBAL DOS RESÍDUOS
    # ========================================================================

    print()
    print(
        "Ajustando Mahalanobis global nos resíduos normais de treino..."
    )

    modelo_global = LedoitWolf(
        store_precision=True,
        assume_centered=False,
    )

    modelo_global.fit(
        residuos_train
    )

    score_global_val = modelo_global.mahalanobis(
        residuos_val
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
            "Gap global não positivo."
        )

    # ========================================================================
    # MAHALANOBIS LOCAL DOS MESMOS RESÍDUOS
    # ========================================================================

    print(
        "Ajustando Mahalanobis local nos mesmos resíduos..."
    )

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
            residuos_train[
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
            residuos_val[
                :,
                idx
            ]
        )

    matriz_scores_locais = np.column_stack(
        [
            scores_locais_val[
                grupo
            ]
            for grupo in grupos
        ]
    )

    score_local_fusao_val = np.max(
        matriz_scores_locais,
        axis=1,
    )

    tau_local = float(
        np.quantile(
            score_local_fusao_val,
            1.0 - FPR_ALVO,
        )
    )

    # ========================================================================
    # DIAGNÓSTICO POR GRUPO
    # ========================================================================

    resultados = []

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
                f"{grupo}: gap local não positivo."
            )

        burden_ratio = (
            gap_global
            /
            gap_local
        )

        # --------------------------------------------------------------------
        # GEOMETRIA GLOBAL × LOCAL NO ESPAÇO RESIDUAL
        # --------------------------------------------------------------------

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
                f"{grupo}: autovalor não positivo."
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

        resultados.append(
            {
                "Grupo":
                    grupo,

                "Tamanho":
                    len(
                        idx
                    ),

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

                "R_Min":
                    r_min,

                "R_Mediano":
                    r_med,

                "R_Max":
                    r_max,

                "Fator_Pessimista":
                    fator_pessimista,

                "Fator_Mediano":
                    fator_mediano,

                "Fator_Otimista":
                    fator_otimista,
            }
        )

    df = pd.DataFrame(
        resultados
    )

    df.to_csv(
        ARQUIVO_POR_GRUPO,
        index=False,
    )

    # ========================================================================
    # RESUMO
    # ========================================================================

    fatores_pess = df[
        "Fator_Pessimista"
    ].to_numpy()

    burdens = df[
        "Burden_Ratio"
    ].to_numpy()

    resumo = pd.DataFrame(
        [
            {
                "Cenario":
                    CENARIO,

                "Espaco":
                    "Residuos_RNN",

                "N_Grupos":
                    len(
                        grupos
                    ),

                "Tau_Global":
                    tau_global,

                "Tau_Local":
                    tau_local,

                "Gap_Global":
                    gap_global,

                "Burden_Min":
                    float(
                        burdens.min()
                    ),

                "Burden_Mediana":
                    float(
                        np.median(
                            burdens
                        )
                    ),

                "Burden_Max":
                    float(
                        burdens.max()
                    ),

                "Fator_Pessimista_Min":
                    float(
                        fatores_pess.min()
                    ),

                "Fator_Pessimista_Mediana":
                    float(
                        np.median(
                            fatores_pess
                        )
                    ),

                "Fator_Pessimista_Max":
                    float(
                        fatores_pess.max()
                    ),

                "Grupos_Fator_Pessimista_Maior_1":
                    int(
                        np.sum(
                            fatores_pess
                            >
                            1.0
                        )
                    ),
            }
        ]
    )

    resumo.to_csv(
        ARQUIVO_RESUMO,
        index=False,
    )

    print()
    print("=" * 108)
    print("RESULTADO NORMAL-ONLY NO ESPAÇO RESIDUAL")
    print("=" * 108)

    print()
    print(
        resumo.to_string(
            index=False
        )
    )

    print()
    print(
        "Fatores pessimistas por grupo:"
    )

    print(
        df[
            [
                "Grupo",
                "Burden_Ratio",
                "R_Max",
                "Fator_Pessimista",
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
        "Este resultado foi obtido somente com resíduos normais "
        "de treino e validação da RNN."
    )

    print(
        "Nenhum teste, probe ou rótulo de ataque foi consultado."
    )

    print()
    print(
        f"Detalhe: {ARQUIVO_POR_GRUPO}"
    )

    print(
        f"Resumo:  {ARQUIVO_RESUMO}"
    )


if __name__ == "__main__":
    main()