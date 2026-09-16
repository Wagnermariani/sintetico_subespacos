import hashlib
import os

import numpy as np
import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_MAPA = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv"
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

ARQUIVO_HASH = os.path.join(
    PROJETO,
    "dados",
    "sha256_eventos_confirmatorios.txt"
)


# ============================================================================
# PARÂMETROS JÁ USADOS NO PILOTO
# ============================================================================

N_TESTE = 20_000

DURACAO_EVENTO = 100

SEED_ORDEM_VARIAVEIS = 2042


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
# MAIN
# ============================================================================

def main():

    print("=" * 76)
    print("CONGELAMENTO DOS EVENTOS CONFIRMATÓRIOS")
    print("=" * 76)

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
    # POSIÇÕES DOS EVENTOS
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

    registros_eventos = []

    for indice, (
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
            -
            1
        )

        direcao = (
            1
            if indice % 2 == 1
            else -1
        )

        registros_eventos.append(
            {
                "Evento":
                    f"A{indice:03d}",

                "Grupo_Atacado":
                    grupo,

                "Indice_Inicio":
                    int(
                        inicio
                    ),

                "Indice_Fim":
                    int(
                        fim
                    ),

                "Duracao_s":
                    DURACAO_EVENTO,

                "Direcao":
                    direcao,
            }
        )

    eventos = pd.DataFrame(
        registros_eventos
    )

    # ------------------------------------------------------------------------
    # ORDEM FIXA DAS FLOATS
    #
    # O mapa verdadeiro é usado apenas para construir o experimento.
    # Nenhum detector recebe esta informação.
    # ------------------------------------------------------------------------

    rng = np.random.default_rng(
        SEED_ORDEM_VARIAVEIS
    )

    registros_ordem = []

    for grupo in grupos:

        floats = (
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

        if len(floats) != 14:
            raise RuntimeError(
                f"{grupo}: esperadas 14 floats."
            )

        ordem = list(
            rng.permutation(
                floats
            )
        )

        for posicao, feature in enumerate(
            ordem,
            start=1
        ):

            registros_ordem.append(
                {
                    "Grupo":
                        grupo,

                    "Posicao":
                        posicao,

                    "Feature":
                        feature,
                }
            )

    ordem_df = pd.DataFrame(
        registros_ordem
    )

    # ------------------------------------------------------------------------
    # SALVAMENTO
    # ------------------------------------------------------------------------

    eventos.to_csv(
        ARQUIVO_EVENTOS,
        index=False
    )

    ordem_df.to_csv(
        ARQUIVO_ORDEM,
        index=False
    )

    hash_eventos = sha256_arquivo(
        ARQUIVO_EVENTOS
    )

    hash_ordem = sha256_arquivo(
        ARQUIVO_ORDEM
    )

    with open(
        ARQUIVO_HASH,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            f"{hash_eventos}  "
            f"dados/eventos_confirmatorios_base.csv\n"
        )

        arquivo.write(
            f"{hash_ordem}  "
            f"dados/ordem_features_confirmatorias.csv\n"
        )

    print()
    print("Eventos congelados:")
    print()

    print(
        eventos.to_string(
            index=False
        )
    )

    print()
    print(
        "Features ordenadas por grupo: "
        f"{len(ordem_df)}"
    )

    print()
    print(
        f"Hash eventos: {hash_eventos}"
    )

    print(
        f"Hash ordem:   {hash_ordem}"
    )

    print()
    print("=" * 76)
    print("GEOMETRIA DOS ATAQUES CONGELADA")
    print("=" * 76)

    print()
    print(
        ARQUIVO_EVENTOS
    )

    print(
        ARQUIVO_ORDEM
    )

    print(
        ARQUIVO_HASH
    )


if __name__ == "__main__":
    main()