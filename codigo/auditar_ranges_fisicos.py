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

ARQUIVO_VALIDACAO = os.path.join(
    PROJETO,
    "dados",
    "validacao_normal_piloto.csv"
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

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "auditoria_ranges"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# AUDITORIA DE UM SPLIT
# ============================================================================

def auditar_split(
    nome_split,
    df,
    parametros,
    mapa
):
    print()
    print("=" * 60)
    print(f"AUDITORIA: {nome_split}")
    print("=" * 60)

    features = mapa["Feature"].tolist()

    # ------------------------------------------------------------------------
    # NaN / infinito
    # ------------------------------------------------------------------------

    matriz = df[
        features
    ].to_numpy(
        dtype=np.float64
    )

    n_nan = int(
        np.isnan(
            matriz
        ).sum()
    )

    n_inf = int(
        np.isinf(
            matriz
        ).sum()
    )

    print(
        f"NaN encontrados: {n_nan}"
    )

    print(
        f"Inf encontrados: {n_inf}"
    )

    # ------------------------------------------------------------------------
    # Attack deve ser exclusivamente zero no normal
    # ------------------------------------------------------------------------

    valores_attack = sorted(
        df[
            "Attack"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"Valores em Attack: {valores_attack}"
    )

    # ------------------------------------------------------------------------
    # Binárias
    # ------------------------------------------------------------------------

    features_binarias = (
        mapa.loc[
            mapa["Tipo"] == "binaria",
            "Feature"
        ]
        .tolist()
    )

    valores_binarios = np.unique(
        df[
            features_binarias
        ].to_numpy()
    )

    binarias_validas = bool(
        np.all(
            np.isin(
                valores_binarios,
                [0, 1]
            )
        )
    )

    print(
        f"Features binárias: "
        f"{len(features_binarias)}"
    )

    print(
        "Valores binários encontrados: "
        f"{valores_binarios.tolist()}"
    )

    print(
        f"Binárias estritamente 0/1: "
        f"{binarias_validas}"
    )

    # ------------------------------------------------------------------------
    # Floats e ranges físicos
    # ------------------------------------------------------------------------

    registros = []

    total_abaixo = 0
    total_acima = 0

    for _, linha in parametros.iterrows():

        feature = linha["Feature"]

        minimo_fisico = float(
            linha["Min_Fisico"]
        )

        maximo_fisico = float(
            linha["Max_Fisico"]
        )

        valores = (
            df[
                feature
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        minimo_observado = float(
            np.min(
                valores
            )
        )

        maximo_observado = float(
            np.max(
                valores
            )
        )

        abaixo = int(
            np.sum(
                valores
                <
                minimo_fisico
            )
        )

        acima = int(
            np.sum(
                valores
                >
                maximo_fisico
            )
        )

        total_abaixo += abaixo
        total_acima += acima

        amplitude = (
            maximo_fisico
            -
            minimo_fisico
        )

        normalizado = (
            valores
            -
            minimo_fisico
        ) / amplitude

        registros.append(
            {
                "Split":
                    nome_split,

                "Feature":
                    feature,

                "Min_Fisico":
                    minimo_fisico,

                "Max_Fisico":
                    maximo_fisico,

                "Min_Observado":
                    minimo_observado,

                "Max_Observado":
                    maximo_observado,

                "Min_Normalizado":
                    float(
                        np.min(
                            normalizado
                        )
                    ),

                "Max_Normalizado":
                    float(
                        np.max(
                            normalizado
                        )
                    ),

                "N_Abaixo_Range":
                    abaixo,

                "N_Acima_Range":
                    acima,
            }
        )

    resultado = pd.DataFrame(
        registros
    )

    print(
        f"Features float auditadas: "
        f"{len(resultado)}"
    )

    print(
        f"Valores abaixo do range: "
        f"{total_abaixo}"
    )

    print(
        f"Valores acima do range: "
        f"{total_acima}"
    )

    print(
        "Menor valor normalizado observado: "
        f"{resultado['Min_Normalizado'].min():.9f}"
    )

    print(
        "Maior valor normalizado observado: "
        f"{resultado['Max_Normalizado'].max():.9f}"
    )

    passou = (
        n_nan == 0
        and
        n_inf == 0
        and
        valores_attack == [0]
        and
        binarias_validas
        and
        total_abaixo == 0
        and
        total_acima == 0
    )

    print()
    print(
        f"RESULTADO {nome_split}: "
        f"{'APROVADO' if passou else 'REPROVADO'}"
    )

    return resultado, passou


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "============================================"
    )

    print(
        "AUDITORIA DOS RANGES FÍSICOS"
    )

    print(
        "============================================"
    )

    print(
        "Carregando arquivos..."
    )

    treino = pd.read_csv(
        ARQUIVO_TREINO
    )

    validacao = pd.read_csv(
        ARQUIVO_VALIDACAO
    )

    parametros = pd.read_csv(
        ARQUIVO_PARAMETROS
    )

    mapa = pd.read_csv(
        ARQUIVO_MAPA
    )

    print(
        f"Treino: {treino.shape}"
    )

    print(
        f"Validação: {validacao.shape}"
    )

    print(
        f"Sensores com range físico: "
        f"{len(parametros)}"
    )

    resultado_treino, passou_treino = (
        auditar_split(
            "TREINO",
            treino,
            parametros,
            mapa
        )
    )

    resultado_validacao, passou_validacao = (
        auditar_split(
            "VALIDACAO",
            validacao,
            parametros,
            mapa
        )
    )

    resultado = pd.concat(
        [
            resultado_treino,
            resultado_validacao,
        ],
        ignore_index=True
    )

    arquivo_resultado = os.path.join(
        DIRETORIO_SAIDA,
        "auditoria_ranges.csv"
    )

    resultado.to_csv(
        arquivo_resultado,
        index=False
    )

    print()
    print("=" * 60)

    if (
        passou_treino
        and
        passou_validacao
    ):
        print(
            "AUDITORIA FINAL: APROVADA"
        )
    else:
        print(
            "AUDITORIA FINAL: REPROVADA"
        )

    print("=" * 60)

    print(
        "\nResultado salvo em:"
    )

    print(
        arquivo_resultado
    )


if __name__ == "__main__":
    main()