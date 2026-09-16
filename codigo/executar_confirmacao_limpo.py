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
    "protocolo_confirmatorio.json",
)

ARQUIVO_CONDICOES = os.path.join(
    PROJETO,
    "dados",
    "condicoes_confirmatorias.csv",
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
    "confirmacao",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
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
# MÉTODOS
# ============================================================================

METODOS_ESTATICOS = [
    "Energia",
    "MAX",
    "TOP3",
    "TOP5",
    "TOP10",
]

METODOS_GLOBAIS = [
    "Global_Energia",
    "Global_MAX",
    "Global_TOP3",
    "Global_TOP5",
    "Global_TOP10",
    "Global_Mahalanobis",
]

METODO_PRIMARIO_LOCAL = "Local_Mahalanobis"


# ============================================================================
# HASH
# ============================================================================

def sha256_arquivo(caminho):

    sha = hashlib.sha256()

    with open(caminho, "rb") as arquivo:

        while True:

            bloco = arquivo.read(
                1024 * 1024
            )

            if not bloco:
                break

            sha.update(bloco)

    return sha.hexdigest()


# ============================================================================
# NORMALIZAÇÃO FÍSICA
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
# SCORES ESTÁTICOS
# ============================================================================

def score_topk(
    Z2,
    k,
):

    maiores = np.partition(
        Z2,
        -k,
        axis=1,
    )[
        :,
        -k:
    ]

    return np.mean(
        maiores,
        axis=1,
    )


def calcular_scores_estaticos(
    Z2,
):

    return {
        "Energia":
            np.mean(
                Z2,
                axis=1,
            ),

        "MAX":
            np.max(
                Z2,
                axis=1,
            ),

        "TOP3":
            score_topk(
                Z2,
                3,
            ),

        "TOP5":
            score_topk(
                Z2,
                5,
            ),

        "TOP10":
            score_topk(
                Z2,
                10,
            ),
    }


def calcular_scores_estaticos_locais(
    Z2,
    grupos,
    indices_por_grupo,
):

    por_metodo = {
        nome: []
        for nome in METODOS_ESTATICOS
    }

    for grupo in grupos:

        idx = indices_por_grupo[
            grupo
        ]

        scores_grupo = calcular_scores_estaticos(
            Z2[
                :,
                idx
            ]
        )

        for nome in METODOS_ESTATICOS:

            por_metodo[
                nome
            ].append(
                scores_grupo[
                    nome
                ]
            )

    resultado = {}

    for nome in METODOS_ESTATICOS:

        resultado[
            f"Local_{nome}"
        ] = np.max(
            np.column_stack(
                por_metodo[
                    nome
                ]
            ),
            axis=1,
        )

    return resultado


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

    print("=" * 94)
    print("EXECUÇÃO CONFIRMATÓRIA — VERSÃO LIMPA")
    print("=" * 94)

    print()
    print(
        "Regra: nenhum limiar será ajustado usando o teste."
    )

    # ========================================================================
    # AUDITORIA DOS ARQUIVOS CONGELADOS
    # ========================================================================

    arquivos_hash = [
        (
            ARQUIVO_CONDICOES,
            HASH_CONDICOES,
            "condições",
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
    print("Verificando hashes congelados...")

    for caminho, hash_esperado, nome in arquivos_hash:

        hash_observado = sha256_arquivo(
            caminho
        )

        print(
            f"{nome:12s}: "
            f"{hash_observado}"
        )

        if hash_observado != hash_esperado:

            raise RuntimeError(
                f"Hash alterado: {nome}"
            )

    print("Hashes congelados: OK")

    # ========================================================================
    # LEITURA
    # ========================================================================

    with open(
        ARQUIVO_PROTOCOLO,
        "r",
        encoding="utf-8",
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
            "Esperadas exatamente 6 condições confirmatórias."
        )

    if len(manifesto) != 5:

        raise RuntimeError(
            "Esperadas exatamente 5 realizações."
        )

    if len(eventos) != 20:

        raise RuntimeError(
            "Esperados exatamente 20 eventos."
        )

    # ========================================================================
    # ORDEM CONGELADA DAS FEATURES ATACADAS
    # ========================================================================

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

    comparacoes_primarias = []

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
        print("=" * 94)
        print(realizacao)
        print("=" * 94)

        # --------------------------------------------------------------------
        # DADOS
        # --------------------------------------------------------------------

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
            "mapa_grupos_descobertos.csv",
        )

        grupos_df = pd.read_csv(
            arquivo_grupos
        )

        # --------------------------------------------------------------------
        # FEATURES
        # --------------------------------------------------------------------

        features = sorted(
            grupos_df[
                "Feature"
            ]
            .tolist()
        )

        if len(features) != 400:

            raise RuntimeError(
                f"{realizacao}: esperadas 400 features."
            )

        if len(set(features)) != 400:

            raise RuntimeError(
                f"{realizacao}: existem features duplicadas."
            )

        indice_feature = {
            feature: indice
            for indice, feature
            in enumerate(
                features
            )
        }

        # --------------------------------------------------------------------
        # GRUPOS DESCOBERTOS
        # --------------------------------------------------------------------

        grupos = sorted(
            grupos_df[
                "Grupo_Descoberto"
            ]
            .unique()
            .tolist()
        )

        if len(grupos) != 20:

            raise RuntimeError(
                f"{realizacao}: esperados 20 grupos descobertos."
            )

        indices_por_grupo = {}

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

        # --------------------------------------------------------------------
        # LABELS DO TESTE
        # --------------------------------------------------------------------

        y_test = np.zeros(
            len(teste_base),
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
                f"{realizacao}: esperados exatamente 2000 pontos de ataque."
            )

        # ====================================================================
        # PADRONIZAÇÃO DOS MÉTODOS ESTÁTICOS
        # ====================================================================

        media = np.mean(
            X_train,
            axis=0,
        )

        desvio = np.std(
            X_train,
            axis=0,
            ddof=0,
        )

        if np.any(
            desvio
            <
            1e-12
        ):

            raise RuntimeError(
                f"{realizacao}: existe feature com desvio quase zero."
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

        # ====================================================================
        # VALIDAÇÃO — MÉTODOS GLOBAIS ESTÁTICOS
        # ====================================================================

        scores_val = {}

        globais_estaticos_val = calcular_scores_estaticos(
            Z2_val
        )

        for nome, score in globais_estaticos_val.items():

            scores_val[
                f"Global_{nome}"
            ] = score

        # ====================================================================
        # VALIDAÇÃO — MAHALANOBIS GLOBAL
        # ====================================================================

        print(
            "Ajustando Mahalanobis global..."
        )

        modelo_global = LedoitWolf(
            store_precision=True,
            assume_centered=False,
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
        # VALIDAÇÃO — MÉTODOS LOCAIS ESTÁTICOS
        # ====================================================================

        scores_val.update(
            calcular_scores_estaticos_locais(
                Z2_val,
                grupos,
                indices_por_grupo,
            )
        )

        # ====================================================================
        # VALIDAÇÃO — MAHALANOBIS LOCAL
        # ====================================================================

        modelos_locais = {}

        mahal_val_por_grupo = []

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            modelo = LedoitWolf(
                store_precision=True,
                assume_centered=False,
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

            mahal_val_por_grupo.append(
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
                mahal_val_por_grupo
            ),
            axis=1,
        )

        # --------------------------------------------------------------------
        # AUDITORIA MATEMÁTICA DO MAX
        # --------------------------------------------------------------------

        diferenca_max_val = float(
            np.max(
                np.abs(
                    scores_val[
                        "Global_MAX"
                    ]
                    -
                    scores_val[
                        "Local_MAX"
                    ]
                )
            )
        )

        if diferenca_max_val > 1e-10:

            raise RuntimeError(
                f"{realizacao}: Global_MAX e Local_MAX deveriam ser equivalentes."
            )

        # ====================================================================
        # LIMIARES — EXCLUSIVAMENTE VALIDAÇÃO NORMAL
        # ====================================================================

        limiares = {}

        print()
        print(
            "Limiarização usando exclusivamente validação normal:"
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

            positivos_val = int(
                np.sum(
                    score
                    >
                    limiar
                )
            )

            print(
                f"{metodo:22s} | "
                f"positivos val={positivos_val:4d} | "
                f"limiar={limiar:.8f}"
            )

        # ====================================================================
        # CONDIÇÕES CONFIRMATÓRIAS
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

            X_test = X_test_base.copy()

            # ================================================================
            # INSERÇÃO DOS 20 EVENTOS
            # ================================================================

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
                        for feature in features_atacadas
                    ],
                    dtype=np.int64,
                )

                # ------------------------------------------------------------
                # AS FEATURES DO EVENTO DEVEM CAIR EM UM ÚNICO GRUPO
                # DESCOBERTO PELO MÉTODO NORMAL-ONLY.
                # ------------------------------------------------------------

                grupos_destino = {
                    feature_para_grupo[
                        feature
                    ]
                    for feature in features_atacadas
                }

                if len(
                    grupos_destino
                ) != 1:

                    raise RuntimeError(
                        f"{realizacao} {nome_condicao}: "
                        "features atacadas atravessaram grupos descobertos."
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

                # ------------------------------------------------------------
                # RECUPERAÇÃO DO LATENTE E PERTURBAÇÃO
                # ------------------------------------------------------------

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
                        f"{realizacao} {nome_condicao}: ataque fora do range."
                    )

                X_test[
                    np.ix_(
                        linhas,
                        idx,
                    )
                ] = valores_atacados

            # =================================================================
            # TESTE — MÉTODOS GLOBAIS ESTÁTICOS
            # =================================================================

            Z2_test = (
                (
                    X_test
                    -
                    media
                )
                /
                desvio
            ) ** 2

            scores_test = {}

            globais_estaticos_test = calcular_scores_estaticos(
                Z2_test
            )

            for nome, score in globais_estaticos_test.items():

                scores_test[
                    f"Global_{nome}"
                ] = score

            # =================================================================
            # TESTE — MAHALANOBIS GLOBAL
            # =================================================================

            scores_test[
                "Global_Mahalanobis"
            ] = modelo_global.mahalanobis(
                X_test
            )

            # =================================================================
            # TESTE — MÉTODOS LOCAIS ESTÁTICOS
            # =================================================================

            scores_test.update(
                calcular_scores_estaticos_locais(
                    Z2_test,
                    grupos,
                    indices_por_grupo,
                )
            )

            # =================================================================
            # TESTE — MAHALANOBIS LOCAL
            # =================================================================

            mahal_test_por_grupo = []

            for grupo in grupos:

                idx = indices_por_grupo[
                    grupo
                ]

                modelo = modelos_locais[
                    grupo
                ]

                mahal_test_por_grupo.append(
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
                    mahal_test_por_grupo
                ),
                axis=1,
            )

            # -----------------------------------------------------------------
            # AUDITORIA DO MAX
            # -----------------------------------------------------------------

            diferenca_max_test = float(
                np.max(
                    np.abs(
                        scores_test[
                            "Global_MAX"
                        ]
                        -
                        scores_test[
                            "Local_MAX"
                        ]
                    )
                )
            )

            if diferenca_max_test > 1e-10:

                raise RuntimeError(
                    f"{realizacao} {nome_condicao}: "
                    "Global_MAX e Local_MAX deveriam ser equivalentes."
                )

            # =================================================================
            # AVALIAÇÃO
            #
            # Somente aqui y_test é consultado.
            # =================================================================

            linhas_condicao = []

            for metodo, score in scores_test.items():

                if metodo not in limiares:

                    raise RuntimeError(
                        f"Não existe limiar de validação para {metodo}."
                    )

                resultado = avaliar(
                    scores=
                        score,

                    y_true=
                        y_test,

                    limiar=
                        limiares[
                            metodo
                        ],
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

                linhas_condicao.append(
                    registro
                )

            # =================================================================
            # COMPARAÇÃO CONFIRMATÓRIA PRIMÁRIA
            #
            # Local Mahalanobis versus melhor baseline global pré-definido.
            #
            # A escolha do "melhor global" é apenas uma avaliação conservadora
            # pós-teste entre métodos que já estavam pré-definidos.
            # Nenhum limiar é alterado.
            # =================================================================

            globais = [
                linha
                for linha in linhas_condicao
                if linha[
                    "Metodo"
                ] in METODOS_GLOBAIS
            ]

            local_primario = next(
                linha
                for linha in linhas_condicao
                if linha[
                    "Metodo"
                ]
                ==
                METODO_PRIMARIO_LOCAL
            )

            melhor_global = max(
                globais,
                key=lambda linha:
                    linha[
                        "Recall"
                    ],
            )

            delta_recall = (
                local_primario[
                    "Recall"
                ]
                -
                melhor_global[
                    "Recall"
                ]
            )

            registro_primario = {
                "Realizacao":
                    realizacao,

                "Condicao":
                    nome_condicao,

                "m":
                    m,

                "delta_latente":
                    delta,

                "Melhor_Global":
                    melhor_global[
                        "Metodo"
                    ],

                "Recall_Melhor_Global":
                    melhor_global[
                        "Recall"
                    ],

                "FPR_Melhor_Global":
                    melhor_global[
                        "FPR"
                    ],

                "Recall_Local_Mahalanobis":
                    local_primario[
                        "Recall"
                    ],

                "FPR_Local_Mahalanobis":
                    local_primario[
                        "FPR"
                    ],

                "Delta_Recall_Local_Menos_Global":
                    delta_recall,
            }

            comparacoes_primarias.append(
                registro_primario
            )

            print()
            print(
                f"{realizacao} {nome_condicao} | "
                f"m={m:2d} delta={delta:.2f} | "
                f"best global={melhor_global['Metodo']} "
                f"R={melhor_global['Recall']:.4f} "
                f"FPR={100.0 * melhor_global['FPR']:.3f}% | "
                f"Local_Mah R={local_primario['Recall']:.4f} "
                f"FPR={100.0 * local_primario['FPR']:.3f}% | "
                f"Delta={delta_recall:+.4f}"
            )

            # =================================================================
            # CHECKPOINT
            # =================================================================

            pd.DataFrame(
                resultados
            ).to_csv(
                os.path.join(
                    DIRETORIO_SAIDA,
                    "resultados_parcial.csv",
                ),
                index=False,
            )

            pd.DataFrame(
                comparacoes_primarias
            ).to_csv(
                os.path.join(
                    DIRETORIO_SAIDA,
                    "comparacao_primaria_parcial.csv",
                ),
                index=False,
            )

        # --------------------------------------------------------------------
        # LIBERA MEMÓRIA DA REALIZAÇÃO
        # --------------------------------------------------------------------

        del treino
        del validacao
        del teste_base
        del X_train
        del X_val
        del X_test_base

    # =========================================================================
    # SALVAMENTO FINAL
    # =========================================================================

    resultados_df = pd.DataFrame(
        resultados
    )

    comparacao_df = pd.DataFrame(
        comparacoes_primarias
    )

    arquivo_resultados = os.path.join(
        DIRETORIO_SAIDA,
        "resultados_confirmatorios.csv",
    )

    arquivo_primario = os.path.join(
        DIRETORIO_SAIDA,
        "comparacao_primaria.csv",
    )

    resultados_df.to_csv(
        arquivo_resultados,
        index=False,
    )

    comparacao_df.to_csv(
        arquivo_primario,
        index=False,
    )

    # =========================================================================
    # RESUMO DESCRITIVO — O "MAPA" COMPLETO
    # =========================================================================

    resumo = (
        resultados_df
        .groupby(
            [
                "Condicao",
                "m",
                "delta_latente",
                "Metodo",
            ],
            as_index=False,
        )
        .agg(
            Recall_Medio=(
                "Recall",
                "mean",
            ),
            Recall_DP=(
                "Recall",
                "std",
            ),
            FPR_Medio=(
                "FPR",
                "mean",
            ),
            FPR_DP=(
                "FPR",
                "std",
            ),
            F1_Medio=(
                "F1",
                "mean",
            ),
            MCC_Medio=(
                "MCC",
                "mean",
            ),
        )
    )

    arquivo_resumo = os.path.join(
        DIRETORIO_SAIDA,
        "resumo_por_condicao_metodo.csv",
    )

    resumo.to_csv(
        arquivo_resumo,
        index=False,
    )

    mapa_recall = resumo.pivot(
        index=[
            "Condicao",
            "m",
            "delta_latente",
        ],
        columns="Metodo",
        values="Recall_Medio",
    )

    arquivo_mapa = os.path.join(
        DIRETORIO_SAIDA,
        "mapa_recall_medio.csv",
    )

    mapa_recall.to_csv(
        arquivo_mapa
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 94)
    print("EXECUÇÃO CONFIRMATÓRIA CONCLUÍDA")
    print("=" * 94)

    print()
    print(
        f"Realizações: "
        f"{manifesto['Realizacao'].nunique()}"
    )

    print(
        f"Condições: "
        f"{len(condicoes)}"
    )

    print(
        "Pares realização-condição: "
        f"{len(manifesto) * len(condicoes)}"
    )

    print(
        "Métodos por prova: "
        f"{resultados_df['Metodo'].nunique()}"
    )

    print()
    print("Resultados completos:")
    print(
        arquivo_resultados
    )

    print()
    print("Comparação primária:")
    print(
        arquivo_primario
    )

    print()
    print("Resumo global × local:")
    print(
        arquivo_resumo
    )

    print()
    print("Mapa de recall médio:")
    print(
        arquivo_mapa
    )

    print()
    print("=" * 94)
    print("REGRA METODOLÓGICA PRESERVADA")
    print("=" * 94)

    print(
        "Todos os limiares foram definidos exclusivamente "
        "na validação normal."
    )

    print(
        "Os rótulos do teste foram utilizados somente "
        "para avaliação após a decisão."
    )

    print(
        "A hipótese confirmatória primária continua sendo "
        "Local_Mahalanobis versus o melhor baseline global pré-definido."
    )

    print(
        "Os demais métodos locais constituem análise secundária/descritiva."
    )


if __name__ == "__main__":
    main()