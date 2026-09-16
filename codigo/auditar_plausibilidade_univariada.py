import os

import numpy as np
import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_TREINO = os.path.join(
    PROJETO,
    "dados",
    "treino_normal_piloto.csv"
)

ARQUIVO_BASE = os.path.join(
    PROJETO,
    "dados",
    "teste_normal_base_piloto.csv"
)

ARQUIVO_ATAQUE = os.path.join(
    PROJETO,
    "dados",
    "teste_ataque_piloto_m10_delta1.csv"
)

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_ataque_piloto_m10_delta1.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "plausibilidade_univariada"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# QUANTIS A AUDITAR
# ============================================================================

QUANTIS = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.95,
    0.99,
    0.995,
    0.999,
]


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 68)
    print("AUDITORIA DE PLAUSIBILIDADE UNIVARIADA DOS ATAQUES")
    print("=" * 68)

    print("\nCarregando arquivos...")

    treino = pd.read_csv(
        ARQUIVO_TREINO
    )

    base = pd.read_csv(
        ARQUIVO_BASE
    )

    ataque = pd.read_csv(
        ARQUIVO_ATAQUE
    )

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    print(
        f"Treino normal: {treino.shape}"
    )

    print(
        f"Teste base: {base.shape}"
    )

    print(
        f"Teste atacado: {ataque.shape}"
    )

    print(
        f"Eventos: {len(eventos)}"
    )

    # ------------------------------------------------------------------------
    # ESTATÍSTICAS DO TREINO NORMAL
    # ------------------------------------------------------------------------

    features_atacadas_unicas = sorted(
        {
            feature
            for texto in eventos[
                "Variaveis_Atacadas"
            ].astype(str)
            for feature in texto.split(";")
        }
    )

    print(
        f"Features que participam de algum ataque: "
        f"{len(features_atacadas_unicas)}"
    )

    estatisticas = {}

    for feature in features_atacadas_unicas:

        valores = treino[
            feature
        ].to_numpy(
            dtype=np.float64
        )

        q = np.quantile(
            valores,
            QUANTIS
        )

        estatisticas[
            feature
        ] = {
            "min":
                float(
                    valores.min()
                ),

            "q001":
                float(q[0]),

            "q005":
                float(q[1]),

            "q01":
                float(q[2]),

            "q05":
                float(q[3]),

            "q95":
                float(q[4]),

            "q99":
                float(q[5]),

            "q995":
                float(q[6]),

            "q999":
                float(q[7]),

            "max":
                float(
                    valores.max()
                ),
        }

    # ------------------------------------------------------------------------
    # CADA VALOR ATACADO
    # ------------------------------------------------------------------------

    registros = []

    for _, evento in eventos.iterrows():

        nome_evento = str(
            evento[
                "Evento"
            ]
        )

        grupo = str(
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

        features = (
            str(
                evento[
                    "Variaveis_Atacadas"
                ]
            )
            .split(";")
        )

        for feature in features:

            stats = estatisticas[
                feature
            ]

            valores_base = (
                base.loc[
                    inicio:fim,
                    feature
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            valores_ataque = (
                ataque.loc[
                    inicio:fim,
                    feature
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            for deslocamento, (
                valor_base,
                valor_ataque
            ) in enumerate(
                zip(
                    valores_base,
                    valores_ataque
                )
            ):

                registros.append(
                    {
                        "Evento":
                            nome_evento,

                        "Grupo":
                            grupo,

                        "Feature":
                            feature,

                        "Direcao":
                            direcao,

                        "Indice":
                            inicio
                            +
                            deslocamento,

                        "Valor_Base":
                            valor_base,

                        "Valor_Ataque":
                            valor_ataque,

                        "Fora_MinMax_Treino":
                            (
                                valor_ataque
                                <
                                stats["min"]
                                or
                                valor_ataque
                                >
                                stats["max"]
                            ),

                        "Fora_P001_P999":
                            (
                                valor_ataque
                                <
                                stats["q001"]
                                or
                                valor_ataque
                                >
                                stats["q999"]
                            ),

                        "Fora_P005_P995":
                            (
                                valor_ataque
                                <
                                stats["q005"]
                                or
                                valor_ataque
                                >
                                stats["q995"]
                            ),

                        "Fora_P01_P99":
                            (
                                valor_ataque
                                <
                                stats["q01"]
                                or
                                valor_ataque
                                >
                                stats["q99"]
                            ),

                        "Fora_P05_P95":
                            (
                                valor_ataque
                                <
                                stats["q05"]
                                or
                                valor_ataque
                                >
                                stats["q95"]
                            ),
                    }
                )

    detalhes = pd.DataFrame(
        registros
    )

    # ------------------------------------------------------------------------
    # RESUMO GLOBAL
    # ------------------------------------------------------------------------

    colunas_fora = [
        "Fora_MinMax_Treino",
        "Fora_P001_P999",
        "Fora_P005_P995",
        "Fora_P01_P99",
        "Fora_P05_P95",
    ]

    print()
    print("=" * 68)
    print("RESULTADO GLOBAL")
    print("=" * 68)

    print(
        f"Valores atacados auditados: "
        f"{len(detalhes)}"
    )

    resumo_global = []

    for coluna in colunas_fora:

        n = int(
            detalhes[
                coluna
            ].sum()
        )

        percentual = (
            100.0
            *
            n
            /
            len(detalhes)
        )

        print(
            f"{coluna:22s}: "
            f"{n:6d} / "
            f"{len(detalhes):6d} "
            f"({percentual:8.4f}%)"
        )

        resumo_global.append(
            {
                "Criterio":
                    coluna,

                "N_Fora":
                    n,

                "N_Total":
                    len(detalhes),

                "Percentual_Fora":
                    percentual,
            }
        )

    # ------------------------------------------------------------------------
    # RESULTADO POR EVENTO
    # ------------------------------------------------------------------------

    resumo_eventos = []

    print()
    print("=" * 68)
    print("POR EVENTO")
    print("=" * 68)

    for nome_evento, parte in detalhes.groupby(
        "Evento"
    ):

        registro = {
            "Evento":
                nome_evento,

            "N_Valores":
                len(parte),
        }

        texto = [
            f"{nome_evento}"
        ]

        for coluna in colunas_fora:

            percentual = (
                100.0
                *
                parte[
                    coluna
                ].mean()
            )

            registro[
                coluna
                +
                "_Percentual"
            ] = percentual

        texto.append(
            "P01-P99="
            f"{registro['Fora_P01_P99_Percentual']:.2f}%"
        )

        texto.append(
            "P005-P995="
            f"{registro['Fora_P005_P995_Percentual']:.2f}%"
        )

        texto.append(
            "MinMax="
            f"{registro['Fora_MinMax_Treino_Percentual']:.2f}%"
        )

        print(
            " | ".join(
                texto
            )
        )

        resumo_eventos.append(
            registro
        )

    # ------------------------------------------------------------------------
    # SALVAMENTO
    # ------------------------------------------------------------------------

    pd.DataFrame(
        resumo_global
    ).to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_global.csv"
        ),
        index=False
    )

    pd.DataFrame(
        resumo_eventos
    ).to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_eventos.csv"
        ),
        index=False
    )

    detalhes.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "detalhes_valores_atacados.csv"
        ),
        index=False
    )

    print()
    print(
        "Resultados salvos em:"
    )

    print(
        DIRETORIO_SAIDA
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Esta etapa CARACTERIZA o ataque; "
        "não define ainda um limite arbitrário "
        "de plausibilidade."
    )


if __name__ == "__main__":
    main()