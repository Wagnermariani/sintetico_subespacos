import hashlib
import json
import os

import pandas as pd

from gerar_normal_sintetico import gerar_conjunto_normal


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

DIRETORIO_CONFIRMacao = os.path.join(
    PROJETO,
    "dados",
    "confirmacao"
)

os.makedirs(
    DIRETORIO_CONFIRMacao,
    exist_ok=True
)


# ============================================================================
# HASHES CONGELADOS
# ============================================================================

HASH_CONDICOES_ESPERADO = (
    "ae7f11dc76559c2fdcbcf528b7c3bf22efef64cca4f5d9b48a38af8c89415074"
)

HASH_PROTOCOLO_ESPERADO = (
    "a2a5891983b2cfc30fb482a8ec8a971202b93e4e5b5e750dcd9e642d7cc024aa"
)


# ============================================================================
# TAMANHOS — MESMOS DO PILOTO
# ============================================================================

N_TREINO = 20_000
N_VALIDACAO = 10_000
N_TESTE = 20_000


# ============================================================================
# HASH
# ============================================================================

def sha256_arquivo(
    caminho
):

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
# AUDITORIA MÍNIMA DO NORMAL
# ============================================================================

def auditar_normal(
    nome,
    df
):

    if len(df) == 0:
        raise RuntimeError(
            f"{nome}: dataframe vazio."
        )

    if not (
        df[
            "Attack"
        ]
        ==
        0
    ).all():

        raise RuntimeError(
            f"{nome}: existem ataques no normal."
        )

    if not (
        df[
            "Tipo"
        ]
        ==
        "Normal"
    ).all():

        raise RuntimeError(
            f"{nome}: Tipo diferente de Normal."
        )

    if df.isna().any().any():

        raise RuntimeError(
            f"{nome}: existem NaN."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("GERAÇÃO DAS REALIZAÇÕES CONFIRMATÓRIAS")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # CONFERE PROTOCOLO CONGELADO
    # ------------------------------------------------------------------------

    hash_condicoes = sha256_arquivo(
        ARQUIVO_CONDICOES
    )

    hash_protocolo = sha256_arquivo(
        ARQUIVO_PROTOCOLO
    )

    print()
    print("Verificando protocolo congelado...")

    print(
        f"Hash condições: {hash_condicoes}"
    )

    print(
        f"Hash protocolo: {hash_protocolo}"
    )

    if (
        hash_condicoes
        !=
        HASH_CONDICOES_ESPERADO
    ):
        raise RuntimeError(
            "O arquivo de condições confirmatórias foi alterado."
        )

    if (
        hash_protocolo
        !=
        HASH_PROTOCOLO_ESPERADO
    ):
        raise RuntimeError(
            "O protocolo confirmatório foi alterado."
        )

    print(
        "Protocolo congelado: OK"
    )

    # ------------------------------------------------------------------------
    # PROTOCOLO
    # ------------------------------------------------------------------------

    with open(
        ARQUIVO_PROTOCOLO,
        "r",
        encoding="utf-8"
    ) as arquivo:

        protocolo = json.load(
            arquivo
        )

    seeds = protocolo[
        "seeds_realizacoes"
    ]

    if len(seeds) != 5:
        raise RuntimeError(
            "Esperadas exatamente 5 seeds."
        )

    # ------------------------------------------------------------------------
    # PARÂMETROS FÍSICOS FIXOS
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

    mapa_original = pd.read_csv(
        ARQUIVO_MAPA
    )

    registros_manifesto = []

    # ========================================================================
    # CINCO REALIZAÇÕES
    # ========================================================================

    for indice, seed_base in enumerate(
        seeds,
        start=1
    ):

        nome_realizacao = (
            f"R{indice:02d}"
        )

        seed_train = int(
            seed_base
        )

        seed_val = (
            seed_train
            +
            1
        )

        seed_test = (
            seed_train
            +
            2
        )

        diretorio_realizacao = os.path.join(
            DIRETORIO_CONFIRMacao,
            nome_realizacao
        )

        os.makedirs(
            diretorio_realizacao,
            exist_ok=True
        )

        print()
        print("=" * 80)

        print(
            f"{nome_realizacao} | "
            f"train={seed_train} | "
            f"val={seed_val} | "
            f"test={seed_test}"
        )

        print("=" * 80)

        # --------------------------------------------------------------------
        # TREINO
        # --------------------------------------------------------------------

        treino, mapa_treino = gerar_conjunto_normal(
            n=N_TREINO,
            seed=seed_train,
            timestamp_inicial=(
                "2027-01-01 00:00:00"
            ),
            parametros_sensores=
                parametros_sensores
        )

        auditar_normal(
            f"{nome_realizacao}_TREINO",
            treino
        )

        # --------------------------------------------------------------------
        # VALIDAÇÃO
        # --------------------------------------------------------------------

        validacao, mapa_validacao = gerar_conjunto_normal(
            n=N_VALIDACAO,
            seed=seed_val,
            timestamp_inicial=(
                "2027-02-01 00:00:00"
            ),
            parametros_sensores=
                parametros_sensores
        )

        auditar_normal(
            f"{nome_realizacao}_VALIDACAO",
            validacao
        )

        # --------------------------------------------------------------------
        # TESTE NORMAL BASE
        # --------------------------------------------------------------------

        teste, mapa_teste = gerar_conjunto_normal(
            n=N_TESTE,
            seed=seed_test,
            timestamp_inicial=(
                "2027-03-01 00:00:00"
            ),
            parametros_sensores=
                parametros_sensores
        )

        auditar_normal(
            f"{nome_realizacao}_TESTE",
            teste
        )

        # --------------------------------------------------------------------
        # MAPA PRECISA CONTINUAR O MESMO
        # --------------------------------------------------------------------

        colunas_mapa = [
            "Feature",
            "Grupo_Verdadeiro",
            "Tipo",
        ]

        mapa_ref = (
            mapa_original[
                colunas_mapa
            ]
            .sort_values(
                "Feature"
            )
            .reset_index(
                drop=True
            )
        )

        for nome, mapa_atual in [
            (
                "treino",
                mapa_treino
            ),
            (
                "validacao",
                mapa_validacao
            ),
            (
                "teste",
                mapa_teste
            ),
        ]:

            atual = (
                mapa_atual[
                    colunas_mapa
                ]
                .sort_values(
                    "Feature"
                )
                .reset_index(
                    drop=True
                )
            )

            if not atual.equals(
                mapa_ref
            ):
                raise RuntimeError(
                    f"{nome_realizacao}: "
                    f"mapa alterado em {nome}."
                )

        # --------------------------------------------------------------------
        # SALVAMENTO
        # --------------------------------------------------------------------

        arquivo_treino = os.path.join(
            diretorio_realizacao,
            "treino_normal.csv"
        )

        arquivo_validacao = os.path.join(
            diretorio_realizacao,
            "validacao_normal.csv"
        )

        arquivo_teste = os.path.join(
            diretorio_realizacao,
            "teste_normal_base.csv"
        )

        treino.to_csv(
            arquivo_treino,
            index=False
        )

        validacao.to_csv(
            arquivo_validacao,
            index=False
        )

        teste.to_csv(
            arquivo_teste,
            index=False
        )

        registros_manifesto.append(
            {
                "Realizacao":
                    nome_realizacao,

                "Seed_Base":
                    seed_train,

                "Seed_Treino":
                    seed_train,

                "Seed_Validacao":
                    seed_val,

                "Seed_Teste":
                    seed_test,

                "N_Treino":
                    len(treino),

                "N_Validacao":
                    len(validacao),

                "N_Teste":
                    len(teste),

                "Arquivo_Treino":
                    arquivo_treino,

                "Arquivo_Validacao":
                    arquivo_validacao,

                "Arquivo_Teste":
                    arquivo_teste,
            }
        )

        print(
            f"{nome_realizacao}: OK"
        )

        del treino
        del validacao
        del teste

    # ========================================================================
    # MANIFESTO
    # ========================================================================

    manifesto = pd.DataFrame(
        registros_manifesto
    )

    arquivo_manifesto = os.path.join(
        DIRETORIO_CONFIRMacao,
        "manifesto_realizacoes.csv"
    )

    manifesto.to_csv(
        arquivo_manifesto,
        index=False
    )

    print()
    print("=" * 80)
    print("GERAÇÃO CONFIRMATÓRIA CONCLUÍDA")
    print("=" * 80)

    print()
    print(
        manifesto[
            [
                "Realizacao",
                "Seed_Treino",
                "Seed_Validacao",
                "Seed_Teste",
                "N_Treino",
                "N_Validacao",
                "N_Teste",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Manifesto:"
    )

    print(
        arquivo_manifesto
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Nenhum ataque foi gerado nesta etapa."
    )

    print(
        "As cinco realizações contêm somente comportamento normal."
    )


if __name__ == "__main__":
    main()