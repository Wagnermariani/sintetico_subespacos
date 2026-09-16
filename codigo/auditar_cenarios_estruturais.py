import os

import numpy as np
import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_MANIFESTO = os.path.join(
    PROJETO,
    "dados",
    "fronteira_estrutural",
    "manifesto_cenarios_estruturais.csv",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

ARQUIVO_MAPA = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "auditoria_normal",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_SAIDA = os.path.join(
    DIRETORIO_SAIDA,
    "auditoria_cenarios_estruturais.csv",
)


# ============================================================================
# AUDITORIA DE UM SPLIT
# ============================================================================

def auditar_split(
    cenario,
    split,
    df,
    mapa,
    parametros,
):

    features = mapa[
        "Feature"
    ].tolist()

    floats = (
        mapa.loc[
            mapa["Tipo"]
            ==
            "float",
            "Feature",
        ]
        .tolist()
    )

    binarias = (
        mapa.loc[
            mapa["Tipo"]
            ==
            "binaria",
            "Feature",
        ]
        .tolist()
    )

    # ------------------------------------------------------------------------
    # DIMENSÃO
    # ------------------------------------------------------------------------

    if len(features) != 400:

        raise RuntimeError(
            "O mapa de referência não possui 400 features."
        )

    faltantes = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    extras_processo = [
        coluna
        for coluna in df.columns
        if coluna not in features
        and coluna not in [
            "timestamp",
            "Attack",
            "Tipo",
        ]
    ]

    # ------------------------------------------------------------------------
    # NaN / INF
    # ------------------------------------------------------------------------

    X = (
        df[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    n_nan = int(
        np.isnan(
            X
        ).sum()
    )

    n_inf = int(
        np.isinf(
            X
        ).sum()
    )

    # ------------------------------------------------------------------------
    # NORMALIDADE
    # ------------------------------------------------------------------------

    attack_ok = bool(
        (
            df[
                "Attack"
            ]
            ==
            0
        ).all()
    )

    tipo_ok = bool(
        (
            df[
                "Tipo"
            ]
            ==
            "Normal"
        ).all()
    )

    # ------------------------------------------------------------------------
    # BINÁRIAS
    # ------------------------------------------------------------------------

    valores_binarios = np.unique(
        df[
            binarias
        ].to_numpy()
    )

    binarias_ok = bool(
        np.all(
            np.isin(
                valores_binarios,
                [0, 1],
            )
        )
    )

    # ------------------------------------------------------------------------
    # RANGES FÍSICOS
    # ------------------------------------------------------------------------

    abaixo_range = 0
    acima_range = 0

    minimo_normalizado = np.inf
    maximo_normalizado = -np.inf

    for feature in floats:

        minimo = float(
            parametros.loc[
                feature,
                "Min_Fisico",
            ]
        )

        maximo = float(
            parametros.loc[
                feature,
                "Max_Fisico",
            ]
        )

        valores = (
            df[
                feature
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        abaixo_range += int(
            np.sum(
                valores
                <
                minimo
            )
        )

        acima_range += int(
            np.sum(
                valores
                >
                maximo
            )
        )

        normalizado = (
            valores
            -
            minimo
        ) / (
            maximo
            -
            minimo
        )

        minimo_normalizado = min(
            minimo_normalizado,
            float(
                normalizado.min()
            ),
        )

        maximo_normalizado = max(
            maximo_normalizado,
            float(
                normalizado.max()
            ),
        )

    aprovado = (
        len(faltantes) == 0
        and
        len(extras_processo) == 0
        and
        n_nan == 0
        and
        n_inf == 0
        and
        attack_ok
        and
        tipo_ok
        and
        binarias_ok
        and
        abaixo_range == 0
        and
        acima_range == 0
    )

    return {
        "Cenario":
            cenario,

        "Split":
            split,

        "N_Linhas":
            len(df),

        "N_Features":
            len(features),

        "Features_Faltantes":
            len(faltantes),

        "Features_Extras":
            len(extras_processo),

        "NaN":
            n_nan,

        "Inf":
            n_inf,

        "Attack_OK":
            attack_ok,

        "Tipo_OK":
            tipo_ok,

        "Binarias_OK":
            binarias_ok,

        "Abaixo_Range":
            abaixo_range,

        "Acima_Range":
            acima_range,

        "Min_Normalizado":
            minimo_normalizado,

        "Max_Normalizado":
            maximo_normalizado,

        "Aprovado":
            aprovado,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 100)
    print("AUDITORIA NORMAL — EXPERIMENTO 4: FRONTEIRA ESTRUTURAL")
    print("=" * 100)

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    if len(manifesto) != 12:

        raise RuntimeError(
            "Esperados exatamente 12 cenários."
        )

    mapa = pd.read_csv(
        ARQUIVO_MAPA
    )

    parametros = (
        pd.read_csv(
            ARQUIVO_PARAMETROS
        )
        .set_index(
            "Feature"
        )
    )

    resultados = []

    for _, linha in manifesto.iterrows():

        cenario = str(
            linha[
                "Cenario"
            ]
        )

        print()
        print("=" * 100)
        print(cenario)
        print("=" * 100)

        arquivos = [
            (
                "TREINO",
                linha[
                    "Arquivo_Treino"
                ],
            ),
            (
                "VALIDACAO",
                linha[
                    "Arquivo_Validacao"
                ],
            ),
            (
                "TESTE",
                linha[
                    "Arquivo_Teste"
                ],
            ),
        ]

        for split, caminho in arquivos:

            df = pd.read_csv(
                caminho
            )

            resultado = auditar_split(
                cenario=
                    cenario,

                split=
                    split,

                df=
                    df,

                mapa=
                    mapa,

                parametros=
                    parametros,
            )

            resultados.append(
                resultado
            )

            print(
                f"{split:10s} | "
                f"linhas={resultado['N_Linhas']:5d} | "
                f"NaN={resultado['NaN']} | "
                f"Inf={resultado['Inf']} | "
                f"binárias={resultado['Binarias_OK']} | "
                f"fora range="
                f"{resultado['Abaixo_Range'] + resultado['Acima_Range']} | "
                f"norm=["
                f"{resultado['Min_Normalizado']:.6f}, "
                f"{resultado['Max_Normalizado']:.6f}] | "
                f"{'OK' if resultado['Aprovado'] else 'ERRO'}"
            )

    resultados_df = pd.DataFrame(
        resultados
    )

    resultados_df.to_csv(
        ARQUIVO_SAIDA,
        index=False,
    )

    aprovados = int(
        resultados_df[
            "Aprovado"
        ].sum()
    )

    total = len(
        resultados_df
    )

    print()
    print("=" * 100)
    print("RESULTADO FINAL")
    print("=" * 100)

    print(
        f"Splits aprovados: "
        f"{aprovados} / {total}"
    )

    if aprovados == total:

        print(
            "AUDITORIA NORMAL ESTRUTURAL: APROVADA"
        )

    else:

        print(
            "AUDITORIA NORMAL ESTRUTURAL: REPROVADA"
        )

    print()
    print(
        "Resultado salvo em:"
    )

    print(
        ARQUIVO_SAIDA
    )


if __name__ == "__main__":
    main()