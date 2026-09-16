import hashlib
import os

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from gerar_normal_estrutural import GANHO_TANH


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

DIRETORIO_EXP = os.path.join(
    PROJETO,
    "dados",
    "fronteira_estrutural",
)

ARQUIVO_GRADE = os.path.join(
    DIRETORIO_EXP,
    "grade_estrutural.csv",
)

ARQUIVO_PROTOCOLO = os.path.join(
    DIRETORIO_EXP,
    "protocolo_estrutural.json",
)

ARQUIVO_MANIFESTO = os.path.join(
    DIRETORIO_EXP,
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
    "probes",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_RESULTADOS = os.path.join(
    DIRETORIO_SAIDA,
    "resultados_probes.csv",
)

ARQUIVO_EVENTOS_RESULTADO = os.path.join(
    DIRETORIO_SAIDA,
    "dispersao_eventos_grupos_descobertos.csv",
)

ARQUIVO_RESUMO = os.path.join(
    DIRETORIO_SAIDA,
    "resumo_probes.csv",
)


# ============================================================================
# HASHES CONGELADOS
# ============================================================================

HASH_GRADE = (
    "a7519f91f907317a3dfea789ca3d490956b4e21938c6beb1ca418bf520268daf"
)

HASH_PROTOCOLO = (
    "8e2431c96288e55d823e7a7b6d2fd124ea67b173cecba94655872b4010ca7949"
)

HASH_EVENTOS = (
    "2083da00312e44eb04855c35349f3aef5defc4e7ce0b66311de047883bf19f7f"
)

HASH_ORDEM = (
    "7999f410767be37251ffe3f3c38679a94c8830b5ec29ae64f029e4256bb77d56"
)


# ============================================================================
# PROBES JÁ REGISTRADOS NO PROTOCOLO
# ============================================================================

PROBES = [
    {
        "Probe":
            "P01",
        "m":
            8,
        "delta":
            1.25,
    },
    {
        "Probe":
            "P02",
        "m":
            8,
        "delta":
            1.50,
    },
]

FPR_ALVO = 0.01


# ============================================================================
# HASH
# ============================================================================

def sha256_arquivo(caminho):

    sha = hashlib.sha256()

    with open(
        caminho,
        "rb",
    ) as arquivo:

        while True:

            bloco = arquivo.read(
                1024 * 1024
            )

            if not bloco:
                break

            sha.update(
                bloco
            )

    return sha.hexdigest()


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
# AVALIAÇÃO
# ============================================================================

def avaliar(
    scores,
    y_true,
    limiar,
):

    y_pred = (
        scores
        >
        limiar
    ).astype(
        np.int8
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    fpr = (
        fp
        /
        (
            fp
            +
            tn
        )
    )

    return {
        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "FPR":
            float(fpr),

        "Precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y_true,
                    y_pred,
                )
            ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 104)
    print("PROBES — EXPERIMENTO 4: FRONTEIRA ESTRUTURAL")
    print("=" * 104)

    print()
    print(
        "Regra: treino aprende os modelos; validação normal define os limiares."
    )

    print(
        "Os rótulos do teste são usados somente depois para calcular métricas."
    )

    # ========================================================================
    # AUDITORIA DOS ARQUIVOS CONGELADOS
    # ========================================================================

    arquivos_hash = [
        (
            ARQUIVO_GRADE,
            HASH_GRADE,
            "grade",
        ),
        (
            ARQUIVO_PROTOCOLO,
            HASH_PROTOCOLO,
            "protocolo",
        ),
        (
            ARQUIVO_EVENTOS,
            HASH_EVENTOS,
            "eventos",
        ),
        (
            ARQUIVO_ORDEM,
            HASH_ORDEM,
            "ordem",
        ),
    ]

    print()
    print(
        "Verificando arquivos congelados..."
    )

    for caminho, esperado, nome in arquivos_hash:

        observado = sha256_arquivo(
            caminho
        )

        print(
            f"{nome:12s}: {observado}"
        )

        if observado != esperado:

            raise RuntimeError(
                f"Hash alterado: {nome}"
            )

    print(
        "Arquivos congelados: OK"
    )

    # ========================================================================
    # LEITURA
    # ========================================================================

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    ordem = pd.read_csv(
        ARQUIVO_ORDEM
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

    if len(manifesto) != 12:

        raise RuntimeError(
            "Esperados exatamente 12 cenários."
        )

    if len(eventos) != 20:

        raise RuntimeError(
            "Esperados exatamente 20 eventos."
        )

    # ------------------------------------------------------------------------
    # ORDEM CONGELADA DAS FEATURES
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

    resultados = []

    dispersao_eventos = []

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
        print("=" * 104)
        print(
            f"{cenario} | alpha={alpha:.2f} | beta={beta:.2f}"
        )
        print("=" * 104)

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

        teste_base = pd.read_csv(
            linha_manifesto[
                "Arquivo_Teste"
            ]
        )

        # --------------------------------------------------------------------
        # GRUPOS DESCOBERTOS SOMENTE PELO NORMAL
        # --------------------------------------------------------------------

        arquivo_grupos = os.path.join(
            DIRETORIO_GRUPOS,
            cenario,
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

        feature_para_grupo = (
            grupos_df
            .set_index(
                "Feature"
            )[
                "Grupo_Descoberto"
            ]
            .to_dict()
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

        score_global_val = modelo_global.mahalanobis(
            X_val
        )

        tau_global = float(
            np.quantile(
                score_global_val,
                1.0 - FPR_ALVO,
            )
        )

        # ====================================================================
        # MODELOS LOCAIS
        # ====================================================================

        modelos_locais = {}

        scores_locais_val = []

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

            scores_locais_val.append(
                modelo_local.mahalanobis(
                    X_val[
                        :,
                        idx
                    ]
                )
            )

        score_local_val = np.max(
            np.column_stack(
                scores_locais_val
            ),
            axis=1,
        )

        tau_local = float(
            np.quantile(
                score_local_val,
                1.0 - FPR_ALVO,
            )
        )

        positivos_global_val = int(
            np.sum(
                score_global_val
                >
                tau_global
            )
        )

        positivos_local_val = int(
            np.sum(
                score_local_val
                >
                tau_local
            )
        )

        print(
            f"Grupos descobertos: {len(grupos)}"
        )

        print(
            f"Global val: {positivos_global_val} positivos | "
            f"tau={tau_global:.6f}"
        )

        print(
            f"Local val:  {positivos_local_val} positivos | "
            f"tau={tau_local:.6f}"
        )

        # ====================================================================
        # LABELS DO TESTE
        # ====================================================================

        y_test = np.zeros(
            len(
                teste_base
            ),
            dtype=np.int8,
        )

        for _, evento in eventos.iterrows():

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

            y_test[
                inicio:
                fim + 1
            ] = 1

        if int(
            y_test.sum()
        ) != 2000:

            raise RuntimeError(
                f"{cenario}: esperados 2000 pontos de ataque."
            )

        # ====================================================================
        # DOIS PROBES CONGELADOS
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

            X_test = X_test_base.copy()

            # ================================================================
            # INSERÇÃO DOS EVENTOS
            # ================================================================

            for _, evento in eventos.iterrows():

                nome_evento = str(
                    evento[
                        "Evento"
                    ]
                )

                grupo_verdadeiro = str(
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
                        grupo_verdadeiro
                    ][
                        :m
                    ]
                )

                grupos_descobertos_evento = sorted(
                    {
                        feature_para_grupo[
                            feature
                        ]
                        for feature in features_atacadas
                    }
                )

                dispersao_eventos.append(
                    {
                        "Cenario":
                            cenario,

                        "Probe":
                            nome_probe,

                        "Evento":
                            nome_evento,

                        "Grupo_Verdadeiro":
                            grupo_verdadeiro,

                        "m":
                            m,

                        "delta_latente":
                            delta,

                        "N_Grupos_Descobertos_Atingidos":
                            len(
                                grupos_descobertos_evento
                            ),

                        "Grupos_Descobertos_Atingidos":
                            ";".join(
                                grupos_descobertos_evento
                            ),
                    }
                )

                idx = np.asarray(
                    [
                        indice_feature[
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
                        idx,
                    )
                ]

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

                if np.any(
                    valores_atacados
                    <
                    0.0
                ) or np.any(
                    valores_atacados
                    >
                    1.0
                ):

                    raise RuntimeError(
                        f"{cenario} {nome_probe}: ataque fora do range."
                    )

                X_test[
                    np.ix_(
                        linhas,
                        idx,
                    )
                ] = valores_atacados

            # ================================================================
            # SCORE GLOBAL
            # ================================================================

            score_global_test = modelo_global.mahalanobis(
                X_test
            )

            # ================================================================
            # SCORE LOCAL
            # ================================================================

            scores_locais_test = []

            for grupo in grupos:

                idx = indices_por_grupo[
                    grupo
                ]

                modelo_local = modelos_locais[
                    grupo
                ]

                scores_locais_test.append(
                    modelo_local.mahalanobis(
                        X_test[
                            :,
                            idx
                        ]
                    )
                )

            score_local_test = np.max(
                np.column_stack(
                    scores_locais_test
                ),
                axis=1,
            )

            # ================================================================
            # AVALIAÇÃO
            # ================================================================

            resultado_global = avaliar(
                scores=
                    score_global_test,

                y_true=
                    y_test,

                limiar=
                    tau_global,
            )

            resultado_local = avaliar(
                scores=
                    score_local_test,

                y_true=
                    y_test,

                limiar=
                    tau_local,
            )

            delta_recall = (
                resultado_local[
                    "Recall"
                ]
                -
                resultado_global[
                    "Recall"
                ]
            )

            resultados.append(
                {
                    "Cenario":
                        cenario,

                    "Alpha_Grupo":
                        alpha,

                    "Beta_Global":
                        beta,

                    "N_Grupos_Descobertos":
                        len(grupos),

                    "Probe":
                        nome_probe,

                    "m":
                        m,

                    "delta_latente":
                        delta,

                    "Recall_Global":
                        resultado_global[
                            "Recall"
                        ],

                    "Recall_Local":
                        resultado_local[
                            "Recall"
                        ],

                    "Delta_Recall":
                        delta_recall,

                    "FPR_Global":
                        resultado_global[
                            "FPR"
                        ],

                    "FPR_Local":
                        resultado_local[
                            "FPR"
                        ],

                    "F1_Global":
                        resultado_global[
                            "F1"
                        ],

                    "F1_Local":
                        resultado_local[
                            "F1"
                        ],

                    "MCC_Global":
                        resultado_global[
                            "MCC"
                        ],

                    "MCC_Local":
                        resultado_local[
                            "MCC"
                        ],
                }
            )

            print()
            print(
                f"{cenario} {nome_probe} | "
                f"m={m} delta={delta:.2f} | "
                f"Global R={resultado_global['Recall']:.4f} "
                f"FPR={100.0 * resultado_global['FPR']:.3f}% | "
                f"Local R={resultado_local['Recall']:.4f} "
                f"FPR={100.0 * resultado_local['FPR']:.3f}% | "
                f"Delta={delta_recall:+.4f}"
            )

        # --------------------------------------------------------------------
        # CHECKPOINT
        # --------------------------------------------------------------------

        pd.DataFrame(
            resultados
        ).to_csv(
            ARQUIVO_RESULTADOS,
            index=False,
        )

        pd.DataFrame(
            dispersao_eventos
        ).to_csv(
            ARQUIVO_EVENTOS_RESULTADO,
            index=False,
        )

    # =========================================================================
    # RESUMO
    # =========================================================================

    resultados_df = pd.DataFrame(
        resultados
    )

    resumo = (
        resultados_df
        .groupby(
            [
                "Cenario",
                "Alpha_Grupo",
                "Beta_Global",
                "N_Grupos_Descobertos",
            ],
            as_index=False,
        )
        .agg(
            Recall_Global_Medio=(
                "Recall_Global",
                "mean",
            ),

            Recall_Local_Medio=(
                "Recall_Local",
                "mean",
            ),

            Delta_Recall_Medio=(
                "Delta_Recall",
                "mean",
            ),

            Delta_Recall_Min=(
                "Delta_Recall",
                "min",
            ),

            Delta_Recall_Max=(
                "Delta_Recall",
                "max",
            ),

            FPR_Global_Medio=(
                "FPR_Global",
                "mean",
            ),

            FPR_Local_Medio=(
                "FPR_Local",
                "mean",
            ),
        )
    )

    resumo.to_csv(
        ARQUIVO_RESUMO,
        index=False,
    )

    print()
    print("=" * 104)
    print("RESUMO DOS PROBES — FRONTEIRA ESTRUTURAL")
    print("=" * 104)

    print()

    print(
        resumo[
            [
                "Cenario",
                "Alpha_Grupo",
                "Beta_Global",
                "N_Grupos_Descobertos",
                "Recall_Global_Medio",
                "Recall_Local_Medio",
                "Delta_Recall_Medio",
                "FPR_Global_Medio",
                "FPR_Local_Medio",
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
        "Os grupos e os limiares foram definidos sem acesso aos ataques."
    )

    print(
        "Os ataques foram utilizados somente depois para verificar "
        "se a previsão estrutural normal-only acompanha o desempenho real."
    )

    print()
    print(
        "Resultados:"
    )

    print(
        ARQUIVO_RESULTADOS
    )

    print(
        ARQUIVO_EVENTOS_RESULTADO
    )

    print(
        ARQUIVO_RESUMO
    )


if __name__ == "__main__":
    main()