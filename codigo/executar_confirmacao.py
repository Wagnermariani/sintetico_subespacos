import hashlib
import json
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

from gerar_normal_sintetico import GANHO_TANH


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_PROTOCOLO = os.path.join(
    PROJETO,
    "dados",
    "protocolo_confirmatorio.json"
)

ARQUIVO_CONDICOES = os.path.join(
    PROJETO,
    "dados",
    "condicoes_confirmatorias.csv"
)

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_confirmatorios_base.csv"
)

ARQUIVO_ORDEM = os.path.join(
    PROJETO,
    "dados",
    "ordem_features_confirmatorias.csv"
)

ARQUIVO_MANIFESTO = os.path.join(
    PROJETO,
    "dados",
    "confirmacao",
    "manifesto_realizacoes.csv"
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv"
)

DIRETORIO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "grupos_confirmatorios"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "confirmacao"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# HASHES CONGELADOS
# ============================================================================

HASH_CONDICOES = (
    "ae7f11dc76559c2fdcbcf528b7c3bf22efef64cca4f5d9b48a38af8c89415074"
)

HASH_PROTOCOLO = (
    "a2a5891983b2cfc30fb482a8ec8a971202b93e4e5b5e750dcd9e642d7cc024aa"
)

HASH_EVENTOS = (
    "2083da00312e44eb04855c35349f3aef5defc4e7ce0b66311de047883bf19f7f"
)

HASH_ORDEM = (
    "7999f410767be37251ffe3f3c38679a94c8830b5ec29ae64f029e4256bb77d56"
)


# ============================================================================
# HASH
# ============================================================================

def sha256_arquivo(caminho):

    sha = hashlib.sha256()

    with open(
        caminho,
        "rb"
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
# NORMALIZAÇÃO FÍSICA
# ============================================================================

def normalizar_fisicamente(
    df,
    features,
    parametros
):

    X = df[
        features
    ].copy()

    for feature in features:

        if feature in parametros:

            minimo = parametros[
                feature
            ][
                "Min_Fisico"
            ]

            maximo = parametros[
                feature
            ][
                "Max_Fisico"
            ]

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
# SCORES GLOBAIS ESTÁTICOS
# ============================================================================

def score_topk(
    Z2,
    k
):

    maiores = np.partition(
        Z2,
        -k,
        axis=1
    )[
        :,
        -k:
    ]

    return np.mean(
        maiores,
        axis=1
    )


def calcular_scores_globais_estaticos(
    Z2
):

    return {
        "Global_Energia":
            np.mean(
                Z2,
                axis=1
            ),

        "Global_MAX":
            np.max(
                Z2,
                axis=1
            ),

        "Global_TOP3":
            score_topk(
                Z2,
                3
            ),

        "Global_TOP5":
            score_topk(
                Z2,
                5
            ),

        "Global_TOP10":
            score_topk(
                Z2,
                10
            ),
    }


# ============================================================================
# AVALIAÇÃO
# ============================================================================

def avaliar(
    scores,
    y_true,
    limiar
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
            labels=[0, 1]
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
                    zero_division=0
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y_true,
                    y_pred
                )
            ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 88)
    print("EXECUÇÃO CONFIRMATÓRIA")
    print("=" * 88)

    print()
    print(
        "Nenhum limiar será ajustado usando o teste."
    )

    # ------------------------------------------------------------------------
    # CONFERE OS QUATRO ARQUIVOS CONGELADOS
    # ------------------------------------------------------------------------

    arquivos_hash = [
        (
            ARQUIVO_CONDICOES,
            HASH_CONDICOES,
            "condições"
        ),
        (
            ARQUIVO_PROTOCOLO,
            HASH_PROTOCOLO,
            "protocolo"
        ),
        (
            ARQUIVO_EVENTOS,
            HASH_EVENTOS,
            "eventos"
        ),
        (
            ARQUIVO_ORDEM,
            HASH_ORDEM,
            "ordem"
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

    # ------------------------------------------------------------------------
    # LEITURA
    # ------------------------------------------------------------------------

    with open(
        ARQUIVO_PROTOCOLO,
        "r",
        encoding="utf-8"
    ) as arquivo:

        protocolo = json.load(
            arquivo
        )

    FPR_ALVO = float(
        protocolo[
            "fpr_alvo_validacao"
        ]
    )

    condicoes = pd.read_csv(
        ARQUIVO_CONDICOES
    )

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    ordem = pd.read_csv(
        ARQUIVO_ORDEM
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

    if len(condicoes) != 6:
        raise RuntimeError(
            "Esperadas 6 condições."
        )

    if len(manifesto) != 5:
        raise RuntimeError(
            "Esperadas 5 realizações."
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
            grupo
        ] = parte[
            "Feature"
        ].tolist()

    resultados = []

    # ========================================================================
    # CINCO REALIZAÇÕES
    # ========================================================================

    for _, linha_manifesto in manifesto.iterrows():

        realizacao = str(
            linha_manifesto[
                "Realizacao"
            ]
        )

        print()
        print("=" * 88)
        print(realizacao)
        print("=" * 88)

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

        arquivo_grupos = os.path.join(
            DIRETORIO_GRUPOS,
            realizacao,
            "mapa_grupos_descobertos.csv"
        )

        grupos_df = pd.read_csv(
            arquivo_grupos
        )

        features = (
            grupos_df[
                "Feature"
            ]
            .tolist()
        )

        if len(features) != 400:
            raise RuntimeError(
                f"{realizacao}: esperadas 400 features."
            )

        indice_feature = {
            feature: indice
            for indice, feature
            in enumerate(
                features
            )
        }

        # --------------------------------------------------------------------
        # NORMALIZAÇÃO
        # --------------------------------------------------------------------

        X_train = normalizar_fisicamente(
            treino,
            features,
            parametros
        )

        X_val = normalizar_fisicamente(
            validacao,
            features,
            parametros
        )

        X_test_base = normalizar_fisicamente(
            teste_base,
            features,
            parametros
        )

        # --------------------------------------------------------------------
        # LABELS — DEFINIDOS SOMENTE PELOS EVENTOS CONGELADOS
        # --------------------------------------------------------------------

        y_test = np.zeros(
            len(
                teste_base
            ),
            dtype=np.int8
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
                inicio:fim + 1
            ] = 1

        if int(
            y_test.sum()
        ) != 2000:

            raise RuntimeError(
                f"{realizacao}: esperados 2000 pontos de ataque."
            )

        # ====================================================================
        # BASELINES ESTÁTICOS — TREINO NORMAL
        # ====================================================================

        media = np.mean(
            X_train,
            axis=0
        )

        desvio = np.std(
            X_train,
            axis=0,
            ddof=0
        )

        if np.any(
            desvio
            <
            1e-12
        ):

            raise RuntimeError(
                f"{realizacao}: desvio quase zero."
            )

        Z2_val = (
            (
                X_val
                -
                media
            )
            /
            desvio
        ) ** 2

        scores_val = (
            calcular_scores_globais_estaticos(
                Z2_val
            )
        )

        # ====================================================================
        # MAHALANOBIS GLOBAL
        # ====================================================================

        print(
            "Ajustando Mahalanobis global..."
        )

        modelo_global = LedoitWolf(
            store_precision=True,
            assume_centered=False
        )

        modelo_global.fit(
            X_train
        )

        scores_val[
            "Global_Mahalanobis"
        ] = modelo_global.mahalanobis(
            X_val
        )

        # ====================================================================
        # MAHALANOBIS LOCAL
        # ====================================================================

        grupos = sorted(
            grupos_df[
                "Grupo_Descoberto"
            ]
            .unique()
            .tolist()
        )

        modelos_locais = {}

        indices_por_grupo = {}

        scores_val_locais = []

        feature_para_grupo = (
            grupos_df
            .set_index(
                "Feature"
            )[
                "Grupo_Descoberto"
            ]
            .to_dict()
        )

        for grupo in grupos:

            features_grupo = (
                grupos_df.loc[
                    grupos_df[
                        "Grupo_Descoberto"
                    ]
                    ==
                    grupo,
                    "Feature"
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
                dtype=np.int64
            )

            indices_por_grupo[
                grupo
            ] = idx

            modelo = LedoitWolf(
                store_precision=True,
                assume_centered=False
            )

            modelo.fit(
                X_train[
                    :,
                    idx
                ]
            )

            modelos_locais[
                grupo
            ] = modelo

            scores_val_locais.append(
                modelo.mahalanobis(
                    X_val[
                        :,
                        idx
                    ]
                )
            )

        scores_val[
            "Local_Mahalanobis"
        ] = np.max(
            np.column_stack(
                scores_val_locais
            ),
            axis=1
        )

        # ====================================================================
        # LIMIARES — SOMENTE VALIDAÇÃO NORMAL
        # ====================================================================

        limiares = {}

        print()
        print(
            "Limiarização pela validação normal:"
        )

        for metodo, score in scores_val.items():

            limiar = float(
                np.quantile(
                    score,
                    1.0
                    -
                    FPR_ALVO
                )
            )

            limiares[
                metodo
            ] = limiar

            positivos = int(
                np.sum(
                    score
                    >
                    limiar
                )
            )

            print(
                f"{metodo:22s} | "
                f"positivos val={positivos:3d} | "
                f"limiar={limiar:.8f}"
            )

        # ====================================================================
        # SEIS CONDIÇÕES CONGELADAS
        # ====================================================================

        for _, condicao in condicoes.iterrows():

            nome_condicao = str(
                condicao[
                    "Condicao_Confirmatoria"
                ]
            )

            m = int(
                condicao[
                    "m"
                ]
            )

            delta = float(
                condicao[
                    "delta_latente"
                ]
            )

            X_test = (
                X_test_base.copy()
            )

            # ----------------------------------------------------------------
            # INSERE OS 20 EVENTOS
            # ----------------------------------------------------------------

            for _, evento in eventos.iterrows():

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

                idx = np.asarray(
                    [
                        indice_feature[
                            feature
                        ]
                        for feature
                        in features_atacadas
                    ],
                    dtype=np.int64
                )

                # Confirma que as features atacadas pertencem a
                # um único grupo DESCOBERTO pelo detector.
                grupos_destino = {
                    feature_para_grupo[
                        feature
                    ]
                    for feature
                    in features_atacadas
                }

                if len(
                    grupos_destino
                ) != 1:

                    raise RuntimeError(
                        f"{realizacao} {nome_condicao}: "
                        "features atravessaram grupos descobertos."
                    )

                linhas = np.arange(
                    inicio,
                    fim + 1
                )

                valores_base = (
                    X_test_base[
                        np.ix_(
                            linhas,
                            idx
                        )
                    ]
                )

                eps = 1e-12

                valores_seguros = np.clip(
                    valores_base,
                    eps,
                    1.0
                    -
                    eps
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
                        f"{realizacao} {nome_condicao}: "
                        "ataque fora do range."
                    )

                X_test[
                    np.ix_(
                        linhas,
                        idx
                    )
                ] = valores_atacados

            # ================================================================
            # SCORES GLOBAIS ESTÁTICOS
            # ================================================================

            Z2_test = (
                (
                    X_test
                    -
                    media
                )
                /
                desvio
            ) ** 2

            scores_test = (
                calcular_scores_globais_estaticos(
                    Z2_test
                )
            )

            # ================================================================
            # MAHALANOBIS GLOBAL
            # ================================================================

            # ================================================================
            # SCORES ESTÁTICOS LOCAIS
            # ================================================================

            scores_test_estaticos_locais = {
                nome: []
                for nome in nomes_estaticos
            }

            for grupo in grupos:

                idx = (
                    indices_por_grupo[
                        grupo
                    ]
                )

                scores_grupo = (
                    calcular_scores_globais_estaticos(
                        Z2_test[
                            :,
                            idx
                        ]
                    )
                )

                for nome in nomes_estaticos:

                    scores_test_estaticos_locais[
                        nome
                    ].append(
                        scores_grupo[
                            f"Global_{nome}"
                        ]
                    )

            for nome in nomes_estaticos:

                scores_test[
                    f"Local_{nome}"
                ] = np.max(
                    np.column_stack(
                        scores_test_estaticos_locais[
                            nome
                        ]
                    ),
                    axis=1
                )

            # ================================================================
            # MAHALANOBIS LOCAL
            # ================================================================

            scores_locais = []

            for grupo in grupos:

                idx = (
                    indices_por_grupo[
                        grupo
                    ]
                )

                modelo = (
                    modelos_locais[
                        grupo
                    ]
                )

                scores_locais.append(
                    modelo.mahalanobis(
                        X_test[
                            :,
                            idx
                        ]
                    )
                )

            scores_test[
                "Local_Mahalanobis"
            ] = np.max(
                np.column_stack(
                    scores_locais
                ),
                axis=1
            )


            linhas_resultado = []

            for metodo, score in scores_test.items():

                resultado = avaliar(
                    scores=
                        score,

                    y_true=
                        y_test,

                    limiar=
                        limiares[
                            metodo
                        ]
                )

                registro = {
                    "Realizacao":
                        realizacao,

                    "Condicao":
                        nome_condicao,

                    "m":
                        m,

                    "delta_latente":
                        delta,

                    "Metodo":
                        metodo,

                    "Limiar_Validacao":
                        limiares[
                            metodo
                        ],

                    **resultado,
                }

                resultados.append(
                    registro
                )

                linhas_resultado.append(
                    registro
                )

            # ================================================================
            # COMPARAÇÃO PRIMÁRIA
            # ================================================================

            globais = [
                linha
                for linha
                in linhas_resultado
                if linha[
                    "Metodo"
                ].startswith(
                    "Global_"
                )
            ]

            local = next(
                linha
                for linha
                in linhas_resultado
                if linha[
                    "Metodo"
                ]
                ==
                "Local_Mahalanobis"
            )

            melhor_global = max(
                globais,
                key=lambda x:
                    x[
                        "Recall"
                    ]
            )

            delta_recall = (
                local[
                    "Recall"
                ]
                -
                melhor_global[
                    "Recall"
                ]
            )

            print()
            print(
                f"{realizacao} {nome_condicao} | "
                f"m={m:2d} delta={delta:.2f} | "
                f"best global="
                f"{melhor_global['Metodo']} "
                f"R={melhor_global['Recall']:.4f} "
                f"FPR={100.0 * melhor_global['FPR']:.3f}% | "
                f"local R={local['Recall']:.4f} "
                f"FPR={100.0 * local['FPR']:.3f}% | "
                f"Delta={delta_recall:+.4f}"
            )

            # ---------------------------------------------------------------
            # CHECKPOINT
            # ---------------------------------------------------------------

            pd.DataFrame(
                resultados
            ).to_csv(
                os.path.join(
                    DIRETORIO_SAIDA,
                    "resultados_parcial.csv"
                ),
                index=False
            )

        del treino
        del validacao
        del teste_base
        del X_train
        del X_val
        del X_test_base

    # ========================================================================
    # FINAL
    # ========================================================================

    resultados_df = pd.DataFrame(
        resultados
    )

    arquivo_resultados = os.path.join(
        DIRETORIO_SAIDA,
        "resultados_confirmatorios.csv"
    )

    resultados_df.to_csv(
        arquivo_resultados,
        index=False
    )

    print()
    print("=" * 88)
    print("EXECUÇÃO CONFIRMATÓRIA CONCLUÍDA")
    print("=" * 88)

    print(
        f"Realizações: {manifesto['Realizacao'].nunique()}"
    )

    print(
        f"Condições: {len(condicoes)}"
    )

    print(
        "Pares realização-condição: "
        f"{len(manifesto) * len(condicoes)}"
    )

    print()
    print(
        "Resultados:"
    )

    print(
        arquivo_resultados
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Todos os limiares usados nesta execução vieram "
        "exclusivamente da validação normal."
    )

    print(
        "Os rótulos do teste foram usados apenas para calcular "
        "as métricas após a decisão."
    )


if __name__ == "__main__":
    main()