import os
import json

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

SEED = 42

N_GRUPOS = 20

N_FLOAT_POR_GRUPO = 14
N_BINARIAS_POR_GRUPO = 6

N_FEATURES_POR_GRUPO = (
    N_FLOAT_POR_GRUPO
    + N_BINARIAS_POR_GRUPO
)

N_FEATURES_TOTAL = (
    N_GRUPOS
    * N_FEATURES_POR_GRUPO
)

# Piloto inicial.
N_TREINO = 20000
N_VALIDACAO = 10000

# ---------------------------------------------------------------------------
# Estrutura estatística
#
# x = sqrt(ALPHA_GRUPO) * fator_grupo
#   + sqrt(BETA_GLOBAL) * fator_global
#   + sqrt(VAR_RUIDO) * ruido_individual
#
# Em população, para floats:
#
# mesma área:
# rho_intra ~= ALPHA_GRUPO + BETA_GLOBAL = 0.70
#
# áreas diferentes:
# rho_inter ~= BETA_GLOBAL = 0.05
# ---------------------------------------------------------------------------

ALPHA_GRUPO = 0.65
BETA_GLOBAL = 0.05

VAR_RUIDO = (
    1.0
    - ALPHA_GRUPO
    - BETA_GLOBAL
)

# Dependência temporal dos fatores latentes.
PHI_GRUPO = 0.97
PHI_GLOBAL = 0.995

# Thresholds das seis variáveis ON/OFF.
# Produzem taxas de ativação diferentes sem destruir
# a estrutura de dependência do grupo.
THRESHOLDS_BINARIOS = np.array(
    [
        -0.35,
        -0.20,
        -0.05,
         0.05,
         0.20,
         0.35,
    ],
    dtype=np.float64
)

# ---------------------------------------------------------------------------
# PARÂMETROS FÍSICOS DOS SENSORES
#
# São gerados UMA ÚNICA VEZ e reutilizados em treino, validação e teste.
# Nunca dependem da seed usada para gerar cada conjunto temporal.
# ---------------------------------------------------------------------------

SEED_PARAMETROS_SENSORES = 1042

# Controla quanto da faixa física é normalmente utilizada.
# A transformação tanh garante que nenhum valor ultrapasse o range.
GANHO_TANH = 0.85


def criar_parametros_sensores():
    """
    Cria ranges físicos fixos para todas as variáveis float.

    Estes parâmetros pertencem ao SENSOR, e não ao split.
    Portanto são idênticos em treino, validação e teste.
    """

    rng = np.random.default_rng(
        SEED_PARAMETROS_SENSORES
    )

    registros = []

    for k in range(
        1,
        N_GRUPOS + 1
    ):

        nome_grupo = (
            f"P{k:02d}"
        )

        for j in range(
            1,
            N_FLOAT_POR_GRUPO + 1
        ):

            nome_feature = (
                f"{nome_grupo}_F{j:02d}"
            )

            minimo_fisico = rng.uniform(
                0.0,
                100.0
            )

            amplitude = rng.uniform(
                50.0,
                500.0
            )

            maximo_fisico = (
                minimo_fisico
                +
                amplitude
            )

            registros.append(
                {
                    "Feature":
                        nome_feature,

                    "Grupo_Verdadeiro":
                        nome_grupo,

                    "Min_Fisico":
                        minimo_fisico,

                    "Max_Fisico":
                        maximo_fisico,
                }
            )

    return pd.DataFrame(
        registros
    )


# ============================================================================
# DIRETÓRIOS
# ============================================================================

DIRETORIO_PROJETO = (
    "/home/wagner/areas/sintetico_subespacos"
)

DIRETORIO_DADOS = os.path.join(
    DIRETORIO_PROJETO,
    "dados"
)

DIRETORIO_RESULTADO = os.path.join(
    DIRETORIO_PROJETO,
    "resultado"
)

os.makedirs(
    DIRETORIO_DADOS,
    exist_ok=True
)

os.makedirs(
    DIRETORIO_RESULTADO,
    exist_ok=True
)


# ============================================================================
# PROCESSO AR(1) ESTACIONÁRIO
# ============================================================================

def gerar_ar1(
    n,
    phi,
    rng
):
    """
    Gera processo AR(1) estacionário com
    média aproximadamente zero e variância aproximadamente 1.
    """

    x = np.empty(
        n,
        dtype=np.float64
    )

    x[0] = rng.normal(
        loc=0.0,
        scale=1.0
    )

    sigma_inovacao = np.sqrt(
        1.0 - phi ** 2
    )

    inovacoes = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n - 1
    )

    for t in range(
        1,
        n
    ):
        x[t] = (
            phi * x[t - 1]
            +
            sigma_inovacao
            * inovacoes[t - 1]
        )

    return x


# ============================================================================
# GERAÇÃO DE UM CONJUNTO NORMAL
# ============================================================================

def gerar_conjunto_normal(
    n,
    seed,
    timestamp_inicial,
    parametros_sensores,
    alpha_grupo=ALPHA_GRUPO,
    beta_global=BETA_GLOBAL
):

    var_ruido = (
        1.0
        -
        alpha_grupo
        -
        beta_global
    )

    if alpha_grupo < 0:
        raise ValueError(
            "alpha_grupo deve ser >= 0."
        )

    if beta_global < 0:
        raise ValueError(
            "beta_global deve ser >= 0."
        )

    if var_ruido <= 0:
        raise ValueError(
            "alpha_grupo + beta_global deve ser < 1."
        )

    rng = np.random.default_rng(
        seed
    )

    print(
        f"Gerando {n} amostras..."
    )

    # ------------------------------------------------------------------------
    # FATOR GLOBAL
    # ------------------------------------------------------------------------

    fator_global = gerar_ar1(
        n=n,
        phi=PHI_GLOBAL,
        rng=rng
    )

    dados = {}

    mapa_features = []

    # ------------------------------------------------------------------------
    # GRUPOS
    # ------------------------------------------------------------------------

    for k in range(
        1,
        N_GRUPOS + 1
    ):

        nome_grupo = (
            f"P{k:02d}"
        )

        fator_grupo = gerar_ar1(
            n=n,
            phi=PHI_GRUPO,
            rng=rng
        )

        # ====================================================================
        # VARIÁVEIS FLOAT
        # ====================================================================

        for j in range(
            1,
            N_FLOAT_POR_GRUPO + 1
        ):

            ruido = rng.normal(
                loc=0.0,
                scale=1.0,
                size=n
            )

            valor_latente = (
                np.sqrt(
                    alpha_grupo
                )
                * fator_grupo
                +
                np.sqrt(
                    beta_global
                )
                * fator_global
                +
                np.sqrt(
                    var_ruido
                )
                * ruido
            )


            nome_feature = (
                f"{nome_grupo}_F{j:02d}"
            )

            parametros = (
                parametros_sensores[
                    nome_feature
                ]
            )

            minimo_fisico = (
                parametros[
                    "Min_Fisico"
                ]
            )

            maximo_fisico = (
                parametros[
                    "Max_Fisico"
                ]
            )

            # Transformação limitada.
            # Não existe clipping:
            # o valor já nasce dentro do range físico.
            valor_unitario = (
                np.tanh(
                    GANHO_TANH
                    * valor_latente
                )
                +
                1.0
            ) / 2.0

            valor = (
                minimo_fisico
                +
                (
                    maximo_fisico
                    -
                    minimo_fisico
                )
                * valor_unitario
            )

            dados[
                nome_feature
            ] = valor.astype(
                np.float32
            )
            mapa_features.append(
                {
                    "Feature":
                        nome_feature,

                    "Grupo_Verdadeiro":
                        nome_grupo,

                    "Tipo":
                        "float"
                }
            )

        # ====================================================================
        # VARIÁVEIS ON/OFF
        # ====================================================================

        for j in range(
            1,
            N_BINARIAS_POR_GRUPO + 1
        ):

            ruido = rng.normal(
                loc=0.0,
                scale=1.0,
                size=n
            )

            valor_latente = (
                np.sqrt(
                    alpha_grupo
                )
                * fator_grupo
                +
                np.sqrt(
                    beta_global
                )
                * fator_global
                +
                np.sqrt(
                    var_ruido
                )
                * ruido
            )

            threshold = (
                THRESHOLDS_BINARIOS[
                    j - 1
                ]
            )

            valor = (
                valor_latente
                >
                threshold
            ).astype(
                np.int8
            )

            nome_feature = (
                f"{nome_grupo}_B{j:02d}"
            )

            dados[
                nome_feature
            ] = valor

            mapa_features.append(
                {
                    "Feature":
                        nome_feature,

                    "Grupo_Verdadeiro":
                        nome_grupo,

                    "Tipo":
                        "binaria"
                }
            )

    # ------------------------------------------------------------------------
    # DATAFRAME
    # ------------------------------------------------------------------------

    df = pd.DataFrame(
        dados
    )

    timestamp = pd.date_range(
        start=timestamp_inicial,
        periods=n,
        freq="s"
    )

    df.insert(
        0,
        "timestamp",
        timestamp
    )

    # Todos os dados desta fase são normais.
    df[
        "Attack"
    ] = np.int8(
        0
    )

    df[
        "Tipo"
    ] = "Normal"

    mapa = pd.DataFrame(
        mapa_features
    )

    return (
        df,
        mapa
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "============================================"
    )
    print(
        "GERAÇÃO DO NORMAL SINTÉTICO — PILOTO"
    )
    print(
        "============================================"
    )

    print(
        f"Grupos: {N_GRUPOS}"
    )

    print(
        f"Floats por grupo: "
        f"{N_FLOAT_POR_GRUPO}"
    )

    print(
        f"Binárias por grupo: "
        f"{N_BINARIAS_POR_GRUPO}"
    )

    print(
        f"Features por grupo: "
        f"{N_FEATURES_POR_GRUPO}"
    )

    print(
        f"Features totais: "
        f"{N_FEATURES_TOTAL}"
    )

    print(
        f"Alpha grupo: "
        f"{ALPHA_GRUPO}"
    )

    print(
        f"Beta global: "
        f"{BETA_GLOBAL}"
    )

    print(
        f"Variância ruído: "
        f"{VAR_RUIDO}"
    )

    print(
        "\nCorrelação teórica aproximada "
        "float-float:"
    )

    print(
        f"rho_intra ~= "
        f"{ALPHA_GRUPO + BETA_GLOBAL:.3f}"
    )

    print(
        f"rho_inter ~= "
        f"{BETA_GLOBAL:.3f}"
    )

        # ------------------------------------------------------------------------
    # PARÂMETROS FÍSICOS FIXOS DOS SENSORES
    # ------------------------------------------------------------------------

    parametros_sensores_df = (
        criar_parametros_sensores()
    )

    parametros_sensores = (
        parametros_sensores_df
        .set_index(
            "Feature"
        )
        .to_dict(
            orient="index"
        )
    )

    arquivo_parametros_sensores = os.path.join(
        DIRETORIO_DADOS,
        "parametros_fisicos_sensores.csv"
    )

    parametros_sensores_df.to_csv(
        arquivo_parametros_sensores,
        index=False
    )

    print(
        "\nParâmetros físicos dos sensores salvos:"
    )

    print(
        arquivo_parametros_sensores
    )

    # ------------------------------------------------------------------------
    # TREINO
    # ------------------------------------------------------------------------

    treino, mapa = gerar_conjunto_normal(
        n=N_TREINO,
        seed=SEED,
        timestamp_inicial=(
            "2026-01-01 00:00:00"
        ),
        parametros_sensores=
            parametros_sensores
    )

    arquivo_treino = os.path.join(
        DIRETORIO_DADOS,
        "treino_normal_piloto.csv"
    )

    treino.to_csv(
        arquivo_treino,
        index=False
    )

    print(
        "\nTreino salvo:"
    )

    print(
        arquivo_treino
    )

    print(
        f"Dimensão treino: "
        f"{treino.shape}"
    )

    del treino

    # ------------------------------------------------------------------------
    # VALIDAÇÃO
    # ------------------------------------------------------------------------

    validacao, mapa_validacao = gerar_conjunto_normal(
        n=N_VALIDACAO,
        seed=SEED + 1,
        timestamp_inicial=(
            "2026-02-01 00:00:00"
        ),
        parametros_sensores=
            parametros_sensores
    )

    arquivo_validacao = os.path.join(
        DIRETORIO_DADOS,
        "validacao_normal_piloto.csv"
    )

    validacao.to_csv(
        arquivo_validacao,
        index=False
    )

    print(
        "\nValidação salva:"
    )

    print(
        arquivo_validacao
    )

    print(
        f"Dimensão validação: "
        f"{validacao.shape}"
    )

    # ------------------------------------------------------------------------
    # MAPA VERDADEIRO
    #
    # IMPORTANTE:
    # este arquivo é apenas para avaliação.
    # O algoritmo de agrupamento NÃO poderá usá-lo.
    # ------------------------------------------------------------------------

    if not mapa.equals(
        mapa_validacao
    ):
        raise RuntimeError(
            "Mapa de features inconsistente "
            "entre treino e validação."
        )

    arquivo_mapa = os.path.join(
        DIRETORIO_DADOS,
        "mapa_grupos_verdadeiros.csv"
    )

    mapa.to_csv(
        arquivo_mapa,
        index=False
    )

    print(
        "\nMapa verdadeiro salvo:"
    )

    print(
        arquivo_mapa
    )

    # ------------------------------------------------------------------------
    # PARÂMETROS DO GERADOR
    # ------------------------------------------------------------------------

    parametros = {
        "seed": SEED,
        "n_grupos": N_GRUPOS,
        "n_float_por_grupo":
            N_FLOAT_POR_GRUPO,
        "n_binarias_por_grupo":
            N_BINARIAS_POR_GRUPO,
        "n_features_por_grupo":
            N_FEATURES_POR_GRUPO,
        "n_features_total":
            N_FEATURES_TOTAL,
        "n_treino_piloto":
            N_TREINO,
        "n_validacao_piloto":
            N_VALIDACAO,
        "alpha_grupo":
            ALPHA_GRUPO,
        "beta_global":
            BETA_GLOBAL,
        "var_ruido":
            VAR_RUIDO,
        "phi_grupo":
            PHI_GRUPO,
        "phi_global":
            PHI_GLOBAL,
        "rho_float_intra_teorico":
            (
                ALPHA_GRUPO
                +
                BETA_GLOBAL
            ),
        "rho_float_inter_teorico":
            BETA_GLOBAL,
    }

    arquivo_parametros = os.path.join(
        DIRETORIO_DADOS,
        "parametros_gerador_piloto.json"
    )

    with open(
        arquivo_parametros,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            parametros,
            f,
            indent=4
        )

    print(
        "\nParâmetros salvos:"
    )

    print(
        arquivo_parametros
    )

    print(
        "\n============================================"
    )
    print(
        "GERAÇÃO CONCLUÍDA"
    )
    print(
        "============================================"
    )

    print(
        "\nATENÇÃO:"
    )

    print(
        "Nenhum ataque foi criado nesta etapa."
    )

    print(
        "O próximo passo será auditar se os "
        "20 grupos podem ser recuperados "
        "usando apenas os dados normais."
    )


if __name__ == "__main__":
    main()