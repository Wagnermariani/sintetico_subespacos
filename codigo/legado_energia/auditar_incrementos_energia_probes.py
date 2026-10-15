import hashlib
import os

import numpy as np
import pandas as pd

from gerar_normal_estrutural import GANHO_TANH


# ============================================================================
# OBJETIVO
# ============================================================================
#
# Auditar por que o diagnóstico normal-only de Energia (F_E > 1 em vários
# cenários) nem sempre correspondeu a maior recall local nas perturbações
# P01/P02.
#
# Esta rotina NÃO altera o protocolo nem recalibra nada com ataques.
# Ela reutiliza:
#   - os mesmos 12 cenários;
#   - os mesmos eventos congelados;
#   - a mesma ordem de features;
#   - as mesmas perturbações P01/P02;
#   - os mesmos grupos descobertos;
#   - os mesmos limiares P99 da validação normal.
#
# A auditoria mede:
#   1) incremento real de Energia causado pela perturbação:
#        Delta E = E_perturbado - E_normal_base;
#   2) sinal desse incremento (positivo ou negativo);
#   3) razão Delta E_local / Delta E_global;
#   4) razão dos incrementos normalizados pelas margens do diagnóstico;
#   5) cruzamentos e perdas de limiar causados pela perturbação;
#   6) posição do ponto normal-base em relação aos limiares.
#
# Isso permite separar:
#   - validade algébrica da amplificação D/d_k;
#   - condição de incremento positivo usada na interpretação do diagnóstico;
#   - efeito da posição de cada amostra em relação ao limiar;
#   - efeito da fusão local pelo máximo.
# ============================================================================


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

ARQUIVO_DIAGNOSTICO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "diagnostico_energia_normal_only",
    "diagnostico_energia_por_grupo.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_incrementos_energia",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_POR_EVENTO = os.path.join(
    DIRETORIO_SAIDA,
    "incrementos_energia_por_evento.csv",
)

ARQUIVO_RESUMO = os.path.join(
    DIRETORIO_SAIDA,
    "resumo_incrementos_energia.csv",
)

ARQUIVO_POR_PONTO = os.path.join(
    DIRETORIO_SAIDA,
    "incrementos_energia_por_ponto.csv",
)


# ============================================================================
# HASHES CONGELADOS — MESMOS DO EXPERIMENTO P01/P02 ORIGINAL
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
# PERTURBAÇÕES CONGELADAS
# ============================================================================

PROBES = [
    {
        "Probe": "P01",
        "m": 8,
        "delta": 1.25,
    },
    {
        "Probe": "P02",
        "m": 8,
        "delta": 1.50,
    },
]

FPR_ALVO = 0.01
EPS = 1e-12


# ============================================================================
# FUNÇÕES AUXILIARES
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


def proporcao(mask):
    return float(
        np.mean(
            np.asarray(
                mask,
                dtype=np.float64,
            )
        )
    )


def media_segura(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.size == 0:
        return np.nan

    return float(
        np.mean(x)
    )


def mediana_segura(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.size == 0:
        return np.nan

    return float(
        np.median(x)
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 112)
    print("AUDITORIA DOS INCREMENTOS DE ENERGIA — P01/P02 — FRONTEIRA ESTRUTURAL")
    print("=" * 112)

    print()
    print(
        "Objetivo: verificar o incremento real de Energia provocado pelas "
        "perturbações e sua relação com o diagnóstico normal-only."
    )

    # ========================================================================
    # INTEGRIDADE DOS ARQUIVOS CONGELADOS
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

    diagnostico = pd.read_csv(
        ARQUIVO_DIAGNOSTICO_GRUPOS
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

    linhas_eventos_saida = []
    linhas_resumo_saida = []
    linhas_ponto_saida = []

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
        print("=" * 112)
        print(
            f"{cenario} | alpha={alpha:.2f} | beta={beta:.2f}"
        )
        print("=" * 112)

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
        # PARTIÇÃO DESCOBERTA
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

        D = len(features)

        indice_feature = {
            feature: indice
            for indice, feature
            in enumerate(features)
        }

        grupos = sorted(
            grupos_df[
                "Grupo_Descoberto"
            ]
            .unique()
            .tolist()
        )

        indice_grupo_coluna = {
            grupo: indice
            for indice, grupo
            in enumerate(grupos)
        }

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
                    for feature
                    in features_grupo
                ],
                dtype=np.int64,
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

        X_test_base = normalizar_fisicamente(
            teste_base,
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
            desvio_train < EPS
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

        Z_base = (
            X_test_base
            -
            media_train
        ) / desvio_train

        Z2_base = (
            Z_base
            **
            2
        )

        # --------------------------------------------------------------------
        # SCORES E LIMIARES NORMAIS
        # --------------------------------------------------------------------

        score_global_val = np.mean(
            Z2_val,
            axis=1,
        )

        tau_global = float(
            np.quantile(
                score_global_val,
                1.0 - FPR_ALVO,
            )
        )

        score_global_base = np.mean(
            Z2_base,
            axis=1,
        )

        locais_val = []
        locais_base = []

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            locais_val.append(
                np.mean(
                    Z2_val[
                        :,
                        idx
                    ],
                    axis=1,
                )
            )

            locais_base.append(
                np.mean(
                    Z2_base[
                        :,
                        idx
                    ],
                    axis=1,
                )
            )

        matriz_local_val = np.column_stack(
            locais_val
        )

        matriz_local_base = np.column_stack(
            locais_base
        )

        score_local_val = np.max(
            matriz_local_val,
            axis=1,
        )

        score_local_base = np.max(
            matriz_local_base,
            axis=1,
        )

        tau_local = float(
            np.quantile(
                score_local_val,
                1.0 - FPR_ALVO,
            )
        )

        # --------------------------------------------------------------------
        # CHECAGEM CONTRA O DIAGNÓSTICO NORMAL-ONLY JÁ PRODUZIDO
        # --------------------------------------------------------------------

        diag_cenario = (
            diagnostico.loc[
                diagnostico[
                    "Cenario"
                ]
                ==
                cenario
            ]
            .copy()
        )

        if len(
            diag_cenario
        ) != len(
            grupos
        ):
            raise RuntimeError(
                f"{cenario}: número de grupos difere do diagnóstico."
            )

        diag_por_grupo = (
            diag_cenario
            .set_index(
                "Grupo"
            )
        )

        if not np.allclose(
            diag_cenario[
                "Tau_Global_Energia"
            ].to_numpy(),
            tau_global,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{cenario}: tau global não coincide com diagnóstico Energia."
            )

        if not np.allclose(
            diag_cenario[
                "Tau_Local_Fusao_Energia"
            ].to_numpy(),
            tau_local,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{cenario}: tau local não coincide com diagnóstico Energia."
            )

        # ====================================================================
        # P01 / P02
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

            delta_latente = float(
                probe[
                    "delta"
                ]
            )

            X_test = X_test_base.copy()

            metadados_eventos = []

            # ----------------------------------------------------------------
            # INSERÇÃO DAS MESMAS PERTURBAÇÕES
            # ----------------------------------------------------------------

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

                grupos_descobertos_atingidos = sorted(
                    {
                        feature_para_grupo[
                            feature
                        ]
                        for feature
                        in features_atacadas
                    }
                )

                idx_features = np.asarray(
                    [
                        indice_feature[
                            feature
                        ]
                        for feature
                        in features_atacadas
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
                        idx_features,
                    )
                ]

                valores_seguros = np.clip(
                    valores_base,
                    EPS,
                    1.0 - EPS,
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
                            delta_latente
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
                        f"{cenario} {nome_probe}: perturbação fora do range."
                    )

                X_test[
                    np.ix_(
                        linhas,
                        idx_features,
                    )
                ] = valores_atacados

                metadados_eventos.append(
                    {
                        "Evento":
                            nome_evento,

                        "Grupo_Verdadeiro":
                            grupo_verdadeiro,

                        "Indice_Inicio":
                            inicio,

                        "Indice_Fim":
                            fim,

                        "Direcao":
                            direcao,

                        "Features_Atacadas":
                            ";".join(
                                features_atacadas
                            ),

                        "Grupos_Descobertos_Atingidos":
                            grupos_descobertos_atingidos,
                    }
                )

            # ----------------------------------------------------------------
            # SCORES APÓS PERTURBAÇÃO
            # ----------------------------------------------------------------

            Z_test = (
                X_test
                -
                media_train
            ) / desvio_train

            Z2_test = (
                Z_test
                **
                2
            )

            score_global_test = np.mean(
                Z2_test,
                axis=1,
            )

            locais_test = []

            for grupo in grupos:

                idx = indices_por_grupo[
                    grupo
                ]

                locais_test.append(
                    np.mean(
                        Z2_test[
                            :,
                            idx
                        ],
                        axis=1,
                    )
                )

            matriz_local_test = np.column_stack(
                locais_test
            )

            score_local_test = np.max(
                matriz_local_test,
                axis=1,
            )

            # ----------------------------------------------------------------
            # ACUMULADORES DO RESUMO CENÁRIO/PROBE
            # ----------------------------------------------------------------

            todos_delta_global = []
            todos_delta_local_alvo = []
            todos_delta_local_fusao = []

            todos_base_global = []
            todos_ataque_global = []
            todos_base_local_fusao = []
            todos_ataque_local_fusao = []

            todos_fatores_evento = []
            todos_razoes_teoricas = []

            # ----------------------------------------------------------------
            # ANÁLISE EVENTO A EVENTO
            # ----------------------------------------------------------------

            for meta in metadados_eventos:

                nome_evento = meta[
                    "Evento"
                ]

                inicio = meta[
                    "Indice_Inicio"
                ]

                fim = meta[
                    "Indice_Fim"
                ]

                direcao = meta[
                    "Direcao"
                ]

                grupos_atingidos = meta[
                    "Grupos_Descobertos_Atingidos"
                ]

                linhas = np.arange(
                    inicio,
                    fim + 1,
                )

                h_evento = len(
                    grupos_atingidos
                )

                # Nesta bateria observamos h(A)=1.
                # O script continua registrando h(A), mas a comparação
                # D/d_k e F_E é feita apenas quando existe um único grupo alvo.
                grupo_alvo = (
                    grupos_atingidos[0]
                    if h_evento == 1
                    else None
                )

                base_global = score_global_base[
                    linhas
                ]

                ataque_global = score_global_test[
                    linhas
                ]

                base_local_fusao = score_local_base[
                    linhas
                ]

                ataque_local_fusao = score_local_test[
                    linhas
                ]

                delta_global = (
                    ataque_global
                    -
                    base_global
                )

                delta_local_fusao = (
                    ataque_local_fusao
                    -
                    base_local_fusao
                )

                if grupo_alvo is not None:

                    coluna_grupo = indice_grupo_coluna[
                        grupo_alvo
                    ]

                    base_local_alvo = matriz_local_base[
                        linhas,
                        coluna_grupo
                    ]

                    ataque_local_alvo = matriz_local_test[
                        linhas,
                        coluna_grupo
                    ]

                    delta_local_alvo = (
                        ataque_local_alvo
                        -
                        base_local_alvo
                    )

                    diag_grupo = diag_por_grupo.loc[
                        grupo_alvo
                    ]

                    d_k = int(
                        diag_grupo[
                            "Tamanho_Grupo"
                        ]
                    )

                    gap_global_diag = float(
                        diag_grupo[
                            "Gap_Global_Energia"
                        ]
                    )

                    gap_local_diag = float(
                        diag_grupo[
                            "Gap_Local_Energia"
                        ]
                    )

                    fator_energia = float(
                        diag_grupo[
                            "Fator_Energia"
                        ]
                    )

                    amplificacao_teorica = float(
                        D
                        /
                        d_k
                    )

                    mascara_razao = (
                        np.abs(
                            delta_global
                        )
                        >
                        EPS
                    )

                    razao_incrementos = (
                        delta_local_alvo[
                            mascara_razao
                        ]
                        /
                        delta_global[
                            mascara_razao
                        ]
                    )

                    razao_normalizada = (
                        (
                            delta_local_alvo[
                                mascara_razao
                            ]
                            /
                            gap_local_diag
                        )
                        /
                        (
                            delta_global[
                                mascara_razao
                            ]
                            /
                            gap_global_diag
                        )
                    )

                    erro_razao_D_d = (
                        np.max(
                            np.abs(
                                razao_incrementos
                                -
                                amplificacao_teorica
                            )
                        )
                        if razao_incrementos.size
                        else np.nan
                    )

                    erro_razao_F = (
                        np.max(
                            np.abs(
                                razao_normalizada
                                -
                                fator_energia
                            )
                        )
                        if razao_normalizada.size
                        else np.nan
                    )

                    # Margens específicas das amostras normais-base
                    margem_global_base = (
                        tau_global
                        -
                        base_global
                    )

                    margem_local_alvo_base = (
                        tau_local
                        -
                        base_local_alvo
                    )

                    # Entre pontos que estavam abaixo de ambos os limiares
                    # e receberam incremento global positivo, comparamos
                    # o avanço relativo até o limiar.
                    mascara_relativa = (
                        (margem_global_base > EPS)
                        &
                        (margem_local_alvo_base > EPS)
                        &
                        (delta_global > EPS)
                    )

                    if np.any(
                        mascara_relativa
                    ):

                        avanco_global = (
                            delta_global[
                                mascara_relativa
                            ]
                            /
                            margem_global_base[
                                mascara_relativa
                            ]
                        )

                        avanco_local = (
                            delta_local_alvo[
                                mascara_relativa
                            ]
                            /
                            margem_local_alvo_base[
                                mascara_relativa
                            ]
                        )

                        fracao_avanco_local_maior = proporcao(
                            avanco_local
                            >
                            avanco_global
                        )

                        media_avanco_global = media_segura(
                            avanco_global
                        )

                        media_avanco_local = media_segura(
                            avanco_local
                        )

                    else:

                        fracao_avanco_local_maior = np.nan
                        media_avanco_global = np.nan
                        media_avanco_local = np.nan

                    # O grupo alvo é o máximo da fusão?
                    argmax_base = np.argmax(
                        matriz_local_base[
                            linhas,
                            :
                        ],
                        axis=1,
                    )

                    argmax_ataque = np.argmax(
                        matriz_local_test[
                            linhas,
                            :
                        ],
                        axis=1,
                    )

                    fracao_alvo_max_base = proporcao(
                        argmax_base
                        ==
                        coluna_grupo
                    )

                    fracao_alvo_max_ataque = proporcao(
                        argmax_ataque
                        ==
                        coluna_grupo
                    )

                else:

                    d_k = np.nan
                    gap_global_diag = np.nan
                    gap_local_diag = np.nan
                    fator_energia = np.nan
                    amplificacao_teorica = np.nan
                    base_local_alvo = np.full(
                        len(linhas),
                        np.nan
                    )
                    ataque_local_alvo = np.full(
                        len(linhas),
                        np.nan
                    )
                    delta_local_alvo = np.full(
                        len(linhas),
                        np.nan
                    )
                    erro_razao_D_d = np.nan
                    erro_razao_F = np.nan
                    fracao_avanco_local_maior = np.nan
                    media_avanco_global = np.nan
                    media_avanco_local = np.nan
                    fracao_alvo_max_base = np.nan
                    fracao_alvo_max_ataque = np.nan

                # ------------------------------------------------------------
                # TRANSIÇÕES DE LIMIAR
                # ------------------------------------------------------------

                alarme_global_base = (
                    base_global
                    >
                    tau_global
                )

                alarme_global_ataque = (
                    ataque_global
                    >
                    tau_global
                )

                alarme_local_base = (
                    base_local_fusao
                    >
                    tau_local
                )

                alarme_local_ataque = (
                    ataque_local_fusao
                    >
                    tau_local
                )

                novos_global = (
                    (~alarme_global_base)
                    &
                    alarme_global_ataque
                )

                perdas_global = (
                    alarme_global_base
                    &
                    (~alarme_global_ataque)
                )

                novos_local = (
                    (~alarme_local_base)
                    &
                    alarme_local_ataque
                )

                perdas_local = (
                    alarme_local_base
                    &
                    (~alarme_local_ataque)
                )

                # ------------------------------------------------------------
                # REGISTRO EVENTO
                # ------------------------------------------------------------

                linha_evento = {
                    "Cenario":
                        cenario,

                    "Probe":
                        nome_probe,

                    "Evento":
                        nome_evento,

                    "Direcao":
                        direcao,

                    "m":
                        m,

                    "delta_latente":
                        delta_latente,

                    "Grupo_Verdadeiro":
                        meta[
                            "Grupo_Verdadeiro"
                        ],

                    "h_A":
                        h_evento,

                    "Grupo_Descoberto_Alvo":
                        grupo_alvo,

                    "Tamanho_Grupo_Alvo":
                        d_k,

                    "Fator_Energia_Diagnostico":
                        fator_energia,

                    "Amplificacao_Teorica_D_por_dk":
                        amplificacao_teorica,

                    "Media_Delta_E_Global":
                        media_segura(
                            delta_global
                        ),

                    "Mediana_Delta_E_Global":
                        mediana_segura(
                            delta_global
                        ),

                    "Media_Delta_E_Local_Alvo":
                        media_segura(
                            delta_local_alvo[
                                ~np.isnan(
                                    delta_local_alvo
                                )
                            ]
                        ),

                    "Mediana_Delta_E_Local_Alvo":
                        mediana_segura(
                            delta_local_alvo[
                                ~np.isnan(
                                    delta_local_alvo
                                )
                            ]
                        ),

                    "Media_Delta_E_Local_Fusao":
                        media_segura(
                            delta_local_fusao
                        ),

                    "Frac_Delta_Global_Positivo":
                        proporcao(
                            delta_global
                            >
                            EPS
                        ),

                    "Frac_Delta_Global_Negativo":
                        proporcao(
                            delta_global
                            <
                            -EPS
                        ),

                    "Frac_Delta_Local_Alvo_Positivo":
                        (
                            proporcao(
                                delta_local_alvo
                                >
                                EPS
                            )
                            if grupo_alvo is not None
                            else np.nan
                        ),

                    "Frac_Delta_Local_Alvo_Negativo":
                        (
                            proporcao(
                                delta_local_alvo
                                <
                                -EPS
                            )
                            if grupo_alvo is not None
                            else np.nan
                        ),

                    "Erro_Max_Razao_Delta_vs_D_por_dk":
                        erro_razao_D_d,

                    "Erro_Max_Razao_Normalizada_vs_F":
                        erro_razao_F,

                    "Recall_Global_no_Evento":
                        proporcao(
                            alarme_global_ataque
                        ),

                    "Recall_Local_no_Evento":
                        proporcao(
                            alarme_local_ataque
                        ),

                    "Alarme_Base_Global_no_Evento":
                        proporcao(
                            alarme_global_base
                        ),

                    "Alarme_Base_Local_no_Evento":
                        proporcao(
                            alarme_local_base
                        ),

                    "Novos_Alarmes_Global":
                        int(
                            np.sum(
                                novos_global
                            )
                        ),

                    "Novos_Alarmes_Local":
                        int(
                            np.sum(
                                novos_local
                            )
                        ),

                    "Perdas_Alarmes_Global":
                        int(
                            np.sum(
                                perdas_global
                            )
                        ),

                    "Perdas_Alarmes_Local":
                        int(
                            np.sum(
                                perdas_local
                            )
                        ),

                    "Frac_Grupo_Alvo_Era_Max_Base":
                        fracao_alvo_max_base,

                    "Frac_Grupo_Alvo_E_Max_Apos_Perturbacao":
                        fracao_alvo_max_ataque,

                    "Media_Avanco_Relativo_Global":
                        media_avanco_global,

                    "Media_Avanco_Relativo_Local_Alvo":
                        media_avanco_local,

                    "Frac_Avanco_Relativo_Local_Maior":
                        fracao_avanco_local_maior,
                }

                linhas_eventos_saida.append(
                    linha_evento
                )

                # ------------------------------------------------------------
                # REGISTRO POR PONTO
                # ------------------------------------------------------------

                for j, linha_abs in enumerate(
                    linhas
                ):

                    linhas_ponto_saida.append(
                        {
                            "Cenario":
                                cenario,

                            "Probe":
                                nome_probe,

                            "Evento":
                                nome_evento,

                            "Indice":
                                int(
                                    linha_abs
                                ),

                            "Direcao":
                                direcao,

                            "Grupo_Descoberto_Alvo":
                                grupo_alvo,

                            "Fator_Energia_Diagnostico":
                                fator_energia,

                            "Score_Global_Base":
                                float(
                                    base_global[
                                        j
                                    ]
                                ),

                            "Score_Global_Perturbado":
                                float(
                                    ataque_global[
                                        j
                                    ]
                                ),

                            "Delta_E_Global":
                                float(
                                    delta_global[
                                        j
                                    ]
                                ),

                            "Score_Local_Alvo_Base":
                                (
                                    float(
                                        base_local_alvo[
                                            j
                                        ]
                                    )
                                    if grupo_alvo is not None
                                    else np.nan
                                ),

                            "Score_Local_Alvo_Perturbado":
                                (
                                    float(
                                        ataque_local_alvo[
                                            j
                                        ]
                                    )
                                    if grupo_alvo is not None
                                    else np.nan
                                ),

                            "Delta_E_Local_Alvo":
                                (
                                    float(
                                        delta_local_alvo[
                                            j
                                        ]
                                    )
                                    if grupo_alvo is not None
                                    else np.nan
                                ),

                            "Score_Local_Fusao_Base":
                                float(
                                    base_local_fusao[
                                        j
                                    ]
                                ),

                            "Score_Local_Fusao_Perturbado":
                                float(
                                    ataque_local_fusao[
                                        j
                                    ]
                                ),

                            "Delta_E_Local_Fusao":
                                float(
                                    delta_local_fusao[
                                        j
                                    ]
                                ),

                            "Tau_Global":
                                tau_global,

                            "Tau_Local":
                                tau_local,

                            "Alarme_Global_Base":
                                int(
                                    alarme_global_base[
                                        j
                                    ]
                                ),

                            "Alarme_Global_Perturbado":
                                int(
                                    alarme_global_ataque[
                                        j
                                    ]
                                ),

                            "Alarme_Local_Base":
                                int(
                                    alarme_local_base[
                                        j
                                    ]
                                ),

                            "Alarme_Local_Perturbado":
                                int(
                                    alarme_local_ataque[
                                        j
                                    ]
                                ),
                        }
                    )

                todos_delta_global.append(
                    delta_global
                )

                if grupo_alvo is not None:

                    todos_delta_local_alvo.append(
                        delta_local_alvo
                    )

                    todos_fatores_evento.append(
                        fator_energia
                    )

                    todos_razoes_teoricas.append(
                        amplificacao_teorica
                    )

                todos_delta_local_fusao.append(
                    delta_local_fusao
                )

                todos_base_global.append(
                    base_global
                )

                todos_ataque_global.append(
                    ataque_global
                )

                todos_base_local_fusao.append(
                    base_local_fusao
                )

                todos_ataque_local_fusao.append(
                    ataque_local_fusao
                )

            # ----------------------------------------------------------------
            # RESUMO CENÁRIO / PROBE
            # ----------------------------------------------------------------

            delta_global_total = np.concatenate(
                todos_delta_global
            )

            delta_local_fusao_total = np.concatenate(
                todos_delta_local_fusao
            )

            base_global_total = np.concatenate(
                todos_base_global
            )

            ataque_global_total = np.concatenate(
                todos_ataque_global
            )

            base_local_fusao_total = np.concatenate(
                todos_base_local_fusao
            )

            ataque_local_fusao_total = np.concatenate(
                todos_ataque_local_fusao
            )

            if todos_delta_local_alvo:

                delta_local_alvo_total = np.concatenate(
                    todos_delta_local_alvo
                )

            else:

                delta_local_alvo_total = np.array(
                    [],
                    dtype=np.float64,
                )

            alarme_g_base_total = (
                base_global_total
                >
                tau_global
            )

            alarme_g_ataque_total = (
                ataque_global_total
                >
                tau_global
            )

            alarme_l_base_total = (
                base_local_fusao_total
                >
                tau_local
            )

            alarme_l_ataque_total = (
                ataque_local_fusao_total
                >
                tau_local
            )

            resumo_probe = {
                "Cenario":
                    cenario,

                "Probe":
                    nome_probe,

                "N_Grupos_Descobertos":
                    len(
                        grupos
                    ),

                "F_Energia_Min_Cenario":
                    float(
                        diag_cenario[
                            "Fator_Energia"
                        ].min()
                    ),

                "F_Energia_Mediana_Cenario":
                    float(
                        diag_cenario[
                            "Fator_Energia"
                        ].median()
                    ),

                "Media_F_Energia_Grupos_Atacados":
                    media_segura(
                        todos_fatores_evento
                    ),

                "Media_Delta_E_Global":
                    media_segura(
                        delta_global_total
                    ),

                "Media_Delta_E_Local_Alvo":
                    media_segura(
                        delta_local_alvo_total
                    ),

                "Media_Delta_E_Local_Fusao":
                    media_segura(
                        delta_local_fusao_total
                    ),

                "Frac_Delta_Global_Positivo":
                    proporcao(
                        delta_global_total
                        >
                        EPS
                    ),

                "Frac_Delta_Global_Negativo":
                    proporcao(
                        delta_global_total
                        <
                        -EPS
                    ),

                "Frac_Delta_Local_Alvo_Positivo":
                    (
                        proporcao(
                            delta_local_alvo_total
                            >
                            EPS
                        )
                        if delta_local_alvo_total.size
                        else np.nan
                    ),

                "Frac_Delta_Local_Alvo_Negativo":
                    (
                        proporcao(
                            delta_local_alvo_total
                            <
                            -EPS
                        )
                        if delta_local_alvo_total.size
                        else np.nan
                    ),

                "Alarme_Base_Global_nos_2000_Pontos":
                    proporcao(
                        alarme_g_base_total
                    ),

                "Alarme_Base_Local_nos_2000_Pontos":
                    proporcao(
                        alarme_l_base_total
                    ),

                "Recall_Global_Perturbado":
                    proporcao(
                        alarme_g_ataque_total
                    ),

                "Recall_Local_Perturbado":
                    proporcao(
                        alarme_l_ataque_total
                    ),

                "Delta_Recall_Local_menos_Global":
                    (
                        proporcao(
                            alarme_l_ataque_total
                        )
                        -
                        proporcao(
                            alarme_g_ataque_total
                        )
                    ),

                "Novos_Alarmes_Global":
                    int(
                        np.sum(
                            (~alarme_g_base_total)
                            &
                            alarme_g_ataque_total
                        )
                    ),

                "Novos_Alarmes_Local":
                    int(
                        np.sum(
                            (~alarme_l_base_total)
                            &
                            alarme_l_ataque_total
                        )
                    ),

                "Perdas_Alarmes_Global":
                    int(
                        np.sum(
                            alarme_g_base_total
                            &
                            (~alarme_g_ataque_total)
                        )
                    ),

                "Perdas_Alarmes_Local":
                    int(
                        np.sum(
                            alarme_l_base_total
                            &
                            (~alarme_l_ataque_total)
                        )
                    ),
            }

            linhas_resumo_saida.append(
                resumo_probe
            )

            print(
                f"{cenario} {nome_probe} | "
                f"Fmin={resumo_probe['F_Energia_Min_Cenario']:.3f} | "
                f"DeltaE+ G={100.0 * resumo_probe['Frac_Delta_Global_Positivo']:.1f}% | "
                f"DeltaE- G={100.0 * resumo_probe['Frac_Delta_Global_Negativo']:.1f}% | "
                f"R_G={100.0 * resumo_probe['Recall_Global_Perturbado']:.2f}% | "
                f"R_L={100.0 * resumo_probe['Recall_Local_Perturbado']:.2f}% | "
                f"novos G/L={resumo_probe['Novos_Alarmes_Global']}/"
                f"{resumo_probe['Novos_Alarmes_Local']} | "
                f"perdas G/L={resumo_probe['Perdas_Alarmes_Global']}/"
                f"{resumo_probe['Perdas_Alarmes_Local']}"
            )

    # =========================================================================
    # SALVAMENTO
    # =========================================================================

    eventos_df = pd.DataFrame(
        linhas_eventos_saida
    )

    resumo_df = pd.DataFrame(
        linhas_resumo_saida
    )

    pontos_df = pd.DataFrame(
        linhas_ponto_saida
    )

    eventos_df.to_csv(
        ARQUIVO_POR_EVENTO,
        index=False,
    )

    resumo_df.to_csv(
        ARQUIVO_RESUMO,
        index=False,
    )

    pontos_df.to_csv(
        ARQUIVO_POR_PONTO,
        index=False,
    )

    # =========================================================================
    # CHECAGENS FINAIS
    # =========================================================================

    if len(
        eventos_df
    ) != 480:
        raise RuntimeError(
            f"Esperados 480 registros por evento; obtidos {len(eventos_df)}."
        )

    if len(
        resumo_df
    ) != 24:
        raise RuntimeError(
            f"Esperados 24 registros no resumo; obtidos {len(resumo_df)}."
        )

    if len(
        pontos_df
    ) != 48000:
        raise RuntimeError(
            f"Esperados 48000 registros por ponto; obtidos {len(pontos_df)}."
        )

    # A razão Delta_local_alvo / Delta_global deve ser D/d_k quando h(A)=1.
    # A razão normalizada pelas margens deve coincidir com F_E.
    eventos_h1 = eventos_df.loc[
        eventos_df[
            "h_A"
        ]
        ==
        1
    ]

    max_erro_dimensional = float(
        eventos_h1[
            "Erro_Max_Razao_Delta_vs_D_por_dk"
        ]
        .dropna()
        .max()
    )

    max_erro_fator = float(
        eventos_h1[
            "Erro_Max_Razao_Normalizada_vs_F"
        ]
        .dropna()
        .max()
    )

    print()
    print("=" * 112)
    print("CHECAGENS ALGÉBRICAS")
    print("=" * 112)

    print(
        f"Máximo erro |Delta_local/Delta_global - D/d_k|: "
        f"{max_erro_dimensional:.6e}"
    )

    print(
        f"Máximo erro |razão normalizada - F_E|: "
        f"{max_erro_fator:.6e}"
    )

    print()
    print("=" * 112)
    print("RESUMO")
    print("=" * 112)

    colunas_impressao = [
        "Cenario",
        "Probe",
        "F_Energia_Min_Cenario",
        "Frac_Delta_Global_Positivo",
        "Frac_Delta_Global_Negativo",
        "Alarme_Base_Global_nos_2000_Pontos",
        "Alarme_Base_Local_nos_2000_Pontos",
        "Recall_Global_Perturbado",
        "Recall_Local_Perturbado",
        "Novos_Alarmes_Global",
        "Novos_Alarmes_Local",
        "Perdas_Alarmes_Global",
        "Perdas_Alarmes_Local",
    ]

    print(
        resumo_df[
            colunas_impressao
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("Resultados:")
    print(ARQUIVO_POR_EVENTO)
    print(ARQUIVO_RESUMO)
    print(ARQUIVO_POR_PONTO)


if __name__ == "__main__":
    main()
