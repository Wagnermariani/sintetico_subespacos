import hashlib
import os

import pandas as pd

from gerar_normal_estrutural import gerar_conjunto_normal


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

DIRETORIO_EXPERIMENTO = os.path.join(
    PROJETO,
    "dados",
    "fronteira_estrutural",
)

ARQUIVO_GRADE = os.path.join(
    DIRETORIO_EXPERIMENTO,
    "grade_estrutural.csv",
)

ARQUIVO_PROTOCOLO = os.path.join(
    DIRETORIO_EXPERIMENTO,
    "protocolo_estrutural.json",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

ARQUIVO_MAPA_REFERENCIA = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv",
)

DIRETORIO_CENARIOS = os.path.join(
    DIRETORIO_EXPERIMENTO,
    "cenarios",
)

ARQUIVO_MANIFESTO = os.path.join(
    DIRETORIO_EXPERIMENTO,
    "manifesto_cenarios_estruturais.csv",
)

os.makedirs(
    DIRETORIO_CENARIOS,
    exist_ok=True,
)


# ============================================================================
# HASHES REGISTRADOS
# ============================================================================

HASH_GRADE = (
    "a7519f91f907317a3dfea789ca3d490956b4e21938c6beb1ca418bf520268daf"
)

HASH_PROTOCOLO = (
    "8e2431c96288e55d823e7a7b6d2fd124ea67b173cecba94655872b4010ca7949"
)


# ============================================================================
# PARÂMETROS DO EXPERIMENTO
# ============================================================================

N_TREINO = 20_000
N_VALIDACAO = 10_000
N_TESTE = 20_000

SEED_TREINO = 6001
SEED_VALIDACAO = 6002
SEED_TESTE = 6003

TIMESTAMP_TREINO = "2031-01-01 00:00:00"
TIMESTAMP_VALIDACAO = "2031-02-01 00:00:00"
TIMESTAMP_TESTE = "2031-03-01 00:00:00"


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
# AUDITORIA DE NORMALIDADE
# ============================================================================

def auditar_normal(
    df,
    nome,
):

    if not (
        df[
            "Attack"
        ]
        ==
        0
    ).all():

        raise RuntimeError(
            f"{nome}: existem linhas marcadas como ataque."
        )

    if not (
        df[
            "Tipo"
        ]
        ==
        "Normal"
    ).all():

        raise RuntimeError(
            f"{nome}: existem linhas não normais."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 96)
    print("GERAÇÃO DOS CENÁRIOS — EXPERIMENTO 4: FRONTEIRA ESTRUTURAL")
    print("=" * 96)

    # ========================================================================
    # VERIFICAÇÃO DA GRADE REGISTRADA
    # ========================================================================

    hash_grade = sha256_arquivo(
        ARQUIVO_GRADE
    )

    hash_protocolo = sha256_arquivo(
        ARQUIVO_PROTOCOLO
    )

    print()
    print(
        f"Hash grade observado:     {hash_grade}"
    )

    print(
        f"Hash protocolo observado: {hash_protocolo}"
    )

    if hash_grade != HASH_GRADE:

        raise RuntimeError(
            "A grade estrutural foi alterada."
        )

    if hash_protocolo != HASH_PROTOCOLO:

        raise RuntimeError(
            "O protocolo estrutural foi alterado."
        )

    print(
        "Grade e protocolo: OK"
    )

    # ========================================================================
    # LEITURA
    # ========================================================================

    grade = pd.read_csv(
        ARQUIVO_GRADE
    )

    if len(grade) != 12:

        raise RuntimeError(
            "Esperados exatamente 12 cenários."
        )

    parametros_df = pd.read_csv(
        ARQUIVO_PARAMETROS
    )

    parametros_sensores = (
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

    mapa_referencia = pd.read_csv(
        ARQUIVO_MAPA_REFERENCIA
    )

    manifesto = []

    # ========================================================================
    # CENÁRIOS
    # ========================================================================

    for _, linha in grade.iterrows():

        cenario = str(
            linha[
                "Cenario"
            ]
        )

        alpha = float(
            linha[
                "Alpha_Grupo"
            ]
        )

        beta = float(
            linha[
                "Beta_Global"
            ]
        )

        var_ruido = float(
            linha[
                "Var_Ruido"
            ]
        )

        print()
        print("=" * 96)
        print(
            f"{cenario} | "
            f"alpha={alpha:.2f} | "
            f"beta={beta:.2f} | "
            f"ruído={var_ruido:.2f}"
        )
        print("=" * 96)

        diretorio_cenario = os.path.join(
            DIRETORIO_CENARIOS,
            cenario,
        )

        os.makedirs(
            diretorio_cenario,
            exist_ok=True,
        )

        # --------------------------------------------------------------------
        # TREINO
        # --------------------------------------------------------------------

        treino, mapa_treino = gerar_conjunto_normal(
            n=N_TREINO,
            seed=SEED_TREINO,
            timestamp_inicial=TIMESTAMP_TREINO,
            parametros_sensores=parametros_sensores,
            alpha_grupo=alpha,
            beta_global=beta,
        )

        # --------------------------------------------------------------------
        # VALIDAÇÃO
        # --------------------------------------------------------------------

        validacao, mapa_validacao = gerar_conjunto_normal(
            n=N_VALIDACAO,
            seed=SEED_VALIDACAO,
            timestamp_inicial=TIMESTAMP_VALIDACAO,
            parametros_sensores=parametros_sensores,
            alpha_grupo=alpha,
            beta_global=beta,
        )

        # --------------------------------------------------------------------
        # TESTE NORMAL-BASE
        # --------------------------------------------------------------------

        teste, mapa_teste = gerar_conjunto_normal(
            n=N_TESTE,
            seed=SEED_TESTE,
            timestamp_inicial=TIMESTAMP_TESTE,
            parametros_sensores=parametros_sensores,
            alpha_grupo=alpha,
            beta_global=beta,
        )

        # --------------------------------------------------------------------
        # AUDITORIAS
        # --------------------------------------------------------------------

        auditar_normal(
            treino,
            f"{cenario} treino",
        )

        auditar_normal(
            validacao,
            f"{cenario} validação",
        )

        auditar_normal(
            teste,
            f"{cenario} teste",
        )

        if not mapa_treino.equals(
            mapa_referencia
        ):

            raise RuntimeError(
                f"{cenario}: mapa do treino diferente da referência."
            )

        if not mapa_validacao.equals(
            mapa_referencia
        ):

            raise RuntimeError(
                f"{cenario}: mapa da validação diferente da referência."
            )

        if not mapa_teste.equals(
            mapa_referencia
        ):

            raise RuntimeError(
                f"{cenario}: mapa do teste diferente da referência."
            )

        # --------------------------------------------------------------------
        # SALVAMENTO
        # --------------------------------------------------------------------

        arquivo_treino = os.path.join(
            diretorio_cenario,
            "treino_normal.csv",
        )

        arquivo_validacao = os.path.join(
            diretorio_cenario,
            "validacao_normal.csv",
        )

        arquivo_teste = os.path.join(
            diretorio_cenario,
            "teste_normal_base.csv",
        )

        treino.to_csv(
            arquivo_treino,
            index=False,
        )

        validacao.to_csv(
            arquivo_validacao,
            index=False,
        )

        teste.to_csv(
            arquivo_teste,
            index=False,
        )

        manifesto.append(
            {
                "Cenario":
                    cenario,

                "Eixo":
                    linha[
                        "Eixo"
                    ],

                "Alpha_Grupo":
                    alpha,

                "Beta_Global":
                    beta,

                "Var_Ruido":
                    var_ruido,

                "Seed_Treino":
                    SEED_TREINO,

                "Seed_Validacao":
                    SEED_VALIDACAO,

                "Seed_Teste":
                    SEED_TESTE,

                "N_Treino":
                    len(
                        treino
                    ),

                "N_Validacao":
                    len(
                        validacao
                    ),

                "N_Teste":
                    len(
                        teste
                    ),

                "Arquivo_Treino":
                    arquivo_treino,

                "Arquivo_Validacao":
                    arquivo_validacao,

                "Arquivo_Teste":
                    arquivo_teste,
            }
        )

        print(
            f"{cenario}: "
            f"{len(treino)} treino | "
            f"{len(validacao)} validação | "
            f"{len(teste)} teste | OK"
        )

    # ========================================================================
    # MANIFESTO
    # ========================================================================

    manifesto_df = pd.DataFrame(
        manifesto
    )

    manifesto_df.to_csv(
        ARQUIVO_MANIFESTO,
        index=False,
    )

    print()
    print("=" * 96)
    print("GERAÇÃO ESTRUTURAL CONCLUÍDA")
    print("=" * 96)

    print()
    print(
        f"Cenários gerados: {len(manifesto_df)}"
    )

    print(
        f"Treinos normais: {len(manifesto_df)}"
    )

    print(
        f"Validações normais: {len(manifesto_df)}"
    )

    print(
        f"Testes normais-base: {len(manifesto_df)}"
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Nenhum ataque foi gerado nesta etapa."
    )

    print(
        "As mesmas seeds foram usadas em todos os cenários."
    )

    print(
        "A única alteração controlada entre os cenários "
        "é a estrutura alpha/beta definida na grade."
    )

    print()
    print(
        "Manifesto:"
    )

    print(
        ARQUIVO_MANIFESTO
    )


if __name__ == "__main__":
    main()