import os

import numpy as np
import pandas as pd

from gerar_normal_sintetico import (
    gerar_conjunto_normal,
    GANHO_TANH,
)


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv"
)

ARQUIVO_MAPA = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv"
)

ARQUIVO_TESTE_BASE = os.path.join(
    PROJETO,
    "dados",
    "teste_normal_base_piloto.csv"
)

ARQUIVO_TESTE_ATAQUE = os.path.join(
    PROJETO,
    "dados",
    "teste_ataque_piloto_m10_delta1.csv"
)

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_ataque_piloto_m10_delta1.csv"
)


# ============================================================================
# PARÂMETROS DO PILOTO
# ============================================================================

SEED_TESTE = 44

SEED_ORDEM_VARIAVEIS = 2042

N_TESTE = 20_000

N_VARIAVEIS_ATACADAS = 10

DELTA_LATENTE = 1.0

DURACAO_EVENTO = 100

TIMESTAMP_INICIAL = (
    "2026-03-01 00:00:00"
)


# ============================================================================
# FUNÇÃO DE PERTURBAÇÃO
# ============================================================================

def aplicar_deslocamento_latente(
    valores,
    minimo_fisico,
    maximo_fisico,
    delta
):
    """
    Recupera aproximadamente a coordenada latente utilizada
    antes da transformação tanh, aplica um deslocamento
    moderado e transforma novamente para o range físico.

    Nenhum clipping físico é utilizado.
    """

    amplitude = (
        maximo_fisico
        -
        minimo_fisico
    )

    unitario = (
        valores
        -
        minimo_fisico
    ) / amplitude

    # Proteção exclusivamente numérica para arctanh.
    # Os dados originais já estão estritamente dentro de (0, 1).
    eps = 1e-12

    unitario_seguro = np.clip(
        unitario,
        eps,
        1.0 - eps
    )

    coordenada_tanh = (
        2.0
        * unitario_seguro
        -
        1.0
    )

    latente = (
        np.arctanh(
            coordenada_tanh
        )
        /
        GANHO_TANH
    )

    latente_atacado = (
        latente
        +
        delta
    )

    unitario_atacado = (
        np.tanh(
            GANHO_TANH
            *
            latente_atacado
        )
        +
        1.0
    ) / 2.0

    atacado = (
        minimo_fisico
        +
        amplitude
        *
        unitario_atacado
    )

    return atacado


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "============================================"
    )
    print(
        "GERAÇÃO DO TESTE COM ATAQUES — PILOTO"
    )
    print(
        "============================================"
    )

    print(
        f"Amostras: {N_TESTE}"
    )

    print(
        f"Variáveis atacadas por evento: "
        f"{N_VARIAVEIS_ATACADAS}"
    )

    print(
        f"Delta latente: {DELTA_LATENTE}"
    )

    print(
        f"Duração por evento: "
        f"{DURACAO_EVENTO} s"
    )

    # ------------------------------------------------------------------------
    # PARÂMETROS FÍSICOS
    # ------------------------------------------------------------------------

    parametros_df = pd.read_csv(
        ARQUIVO_PARAMETROS
    )

    parametros_sensores = (
        parametros_df
        .set_index(
            "Feature"
        )
        .to_dict(
            orient="index"
        )
    )

    mapa = pd.read_csv(
        ARQUIVO_MAPA
    )

    grupos = sorted(
        mapa[
            "Grupo_Verdadeiro"
        ]
        .unique()
        .tolist()
    )

    if len(grupos) != 20:
        raise RuntimeError(
            "Esperados exatamente 20 grupos."
        )

    # ------------------------------------------------------------------------
    # TESTE NORMAL BASE
    # ------------------------------------------------------------------------

    print(
        "\nGerando teste normal base..."
    )

    teste, _ = gerar_conjunto_normal(
        n=N_TESTE,
        seed=SEED_TESTE,
        timestamp_inicial=
            TIMESTAMP_INICIAL,
        parametros_sensores=
            parametros_sensores
    )

    teste.to_csv(
        ARQUIVO_TESTE_BASE,
        index=False
    )

    print(
        "Teste normal base salvo:"
    )

    print(
        ARQUIVO_TESTE_BASE
    )

    # ------------------------------------------------------------------------
    # CÓPIA QUE RECEBERÁ OS ATAQUES
    # ------------------------------------------------------------------------

    teste_ataque = teste.copy()

    teste_ataque[
        "Attack"
    ] = 0

    teste_ataque[
        "Tipo"
    ] = "Normal"

    # ------------------------------------------------------------------------
    # ORDEM FIXA DAS FLOATS DE CADA GRUPO
    #
    # Isso permitirá no futuro comparar m=2,4,6,8,10...
    # usando subconjuntos aninhados das mesmas variáveis.
    # ------------------------------------------------------------------------

    rng_ordem = np.random.default_rng(
        SEED_ORDEM_VARIAVEIS
    )

    ordem_por_grupo = {}

    for grupo in grupos:

        floats_grupo = (
            mapa.loc[
                (
                    mapa[
                        "Grupo_Verdadeiro"
                    ]
                    ==
                    grupo
                )
                &
                (
                    mapa[
                        "Tipo"
                    ]
                    ==
                    "float"
                ),
                "Feature"
            ]
            .tolist()
        )

        if len(floats_grupo) != 14:
            raise RuntimeError(
                f"{grupo}: esperadas 14 floats."
            )

        floats_grupo = list(
            rng_ordem.permutation(
                floats_grupo
            )
        )

        ordem_por_grupo[
            grupo
        ] = floats_grupo

    # ------------------------------------------------------------------------
    # POSIÇÕES DOS 20 EVENTOS
    # ------------------------------------------------------------------------

    inicios = np.linspace(
        500,
        N_TESTE
        -
        DURACAO_EVENTO
        -
        500,
        len(grupos),
        dtype=int
    )

    eventos = []

    # ------------------------------------------------------------------------
    # INSERÇÃO DOS ATAQUES
    # ------------------------------------------------------------------------

    print(
        "\nInserindo eventos..."
    )

    for indice_evento, (
        grupo,
        inicio
    ) in enumerate(
        zip(
            grupos,
            inicios
        ),
        start=1
    ):

        fim = (
            inicio
            +
            DURACAO_EVENTO
        )

        # Alternamos a direção para não transformar
        # todos os ataques em simples deslocamentos positivos.
        direcao = (
            1.0
            if indice_evento % 2 == 1
            else -1.0
        )

        delta_evento = (
            direcao
            *
            DELTA_LATENTE
        )

        features_atacadas = (
            ordem_por_grupo[
                grupo
            ][
                :N_VARIAVEIS_ATACADAS
            ]
        )

        for feature in features_atacadas:

            parametros = (
                parametros_sensores[
                    feature
                ]
            )

            minimo_fisico = float(
                parametros[
                    "Min_Fisico"
                ]
            )

            maximo_fisico = float(
                parametros[
                    "Max_Fisico"
                ]
            )

            valores_originais = (
                teste_ataque.loc[
                    inicio:fim - 1,
                    feature
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            valores_atacados = (
                aplicar_deslocamento_latente(
                    valores=
                        valores_originais,

                    minimo_fisico=
                        minimo_fisico,

                    maximo_fisico=
                        maximo_fisico,

                    delta=
                        delta_evento
                )
                .astype(
                    np.float32
                )
            )

            teste_ataque.loc[
                inicio:fim - 1,
                feature
            ] = valores_atacados

        teste_ataque.loc[
            inicio:fim - 1,
            "Attack"
        ] = 1

        teste_ataque.loc[
            inicio:fim - 1,
            "Tipo"
        ] = "Ataque"

        eventos.append(
            {
                "Evento":
                    f"A{indice_evento:03d}",

                "Grupo_Atacado":
                    grupo,

                "Indice_Inicio":
                    inicio,

                "Indice_Fim":
                    fim - 1,

                "Timestamp_Inicio":
                    teste_ataque.loc[
                        inicio,
                        "timestamp"
                    ],

                "Timestamp_Fim":
                    teste_ataque.loc[
                        fim - 1,
                        "timestamp"
                    ],

                "Duracao_s":
                    DURACAO_EVENTO,

                "Direcao":
                    int(
                        direcao
                    ),

                "Delta_Latente":
                    DELTA_LATENTE,

                "N_Variaveis_Atacadas":
                    len(
                        features_atacadas
                    ),

                "Variaveis_Atacadas":
                    ";".join(
                        features_atacadas
                    ),
            }
        )

        print(
            f"A{indice_evento:03d} | "
            f"{grupo} | "
            f"{inicio}:{fim - 1} | "
            f"direção {int(direcao):+d} | "
            f"{len(features_atacadas)} floats"
        )

    # ------------------------------------------------------------------------
    # AUDITORIAS MÍNIMAS
    # ------------------------------------------------------------------------

    n_attack = int(
        teste_ataque[
            "Attack"
        ].sum()
    )

    esperado = (
        len(grupos)
        *
        DURACAO_EVENTO
    )

    if n_attack != esperado:
        raise RuntimeError(
            "Quantidade de pontos de ataque inesperada."
        )

    # Confirma ranges físicos depois do ataque.
    for _, linha in parametros_df.iterrows():

        feature = linha[
            "Feature"
        ]

        minimo = float(
            linha[
                "Min_Fisico"
            ]
        )

        maximo = float(
            linha[
                "Max_Fisico"
            ]
        )

        valores = (
            teste_ataque[
                feature
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if np.any(
            valores < minimo
        ):
            raise RuntimeError(
                f"{feature}: valor abaixo do range."
            )

        if np.any(
            valores > maximo
        ):
            raise RuntimeError(
                f"{feature}: valor acima do range."
            )

    # ------------------------------------------------------------------------
    # SALVAMENTO
    # ------------------------------------------------------------------------

    eventos_df = pd.DataFrame(
        eventos
    )

    teste_ataque.to_csv(
        ARQUIVO_TESTE_ATAQUE,
        index=False
    )

    eventos_df.to_csv(
        ARQUIVO_EVENTOS,
        index=False
    )

    print()
    print(
        "============================================"
    )

    print(
        "TESTE GERADO"
    )

    print(
        "============================================"
    )

    print(
        f"Shape: {teste_ataque.shape}"
    )

    print(
        f"Pontos normais: "
        f"{len(teste_ataque) - n_attack}"
    )

    print(
        f"Pontos de ataque: {n_attack}"
    )

    print(
        f"Eventos: {len(eventos_df)}"
    )

    print(
        "\nTeste com ataques:"
    )

    print(
        ARQUIVO_TESTE_ATAQUE
    )

    print(
        "\nMetadados dos eventos:"
    )

    print(
        ARQUIVO_EVENTOS
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Este arquivo é apenas um piloto para validar "
        "o mecanismo de ataque."
    )

    print(
        "m=10 e delta=1.0 ainda NÃO são parâmetros "
        "congelados do experimento."
    )


if __name__ == "__main__":
    main()