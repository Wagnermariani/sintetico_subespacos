import os

import numpy as np
import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

DIRETORIO_CONFIRM = os.path.join(
    PROJETO,
    "dados",
    "confirmacao"
)

ARQUIVO_MANIFESTO = os.path.join(
    DIRETORIO_CONFIRM,
    "manifesto_realizacoes.csv"
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
    "auditoria_confirmacao_normal"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# AUDITORIA DE UM SPLIT
# ============================================================================

def auditar_split(
    realizacao,
    nome_split,
    df,
    mapa,
    parametros
):

    features = (
        mapa[
            "Feature"
        ]
        .tolist()
    )

    floats = (
        mapa.loc[
            mapa[
                "Tipo"
            ]
            ==
            "float",
            "Feature"
        ]
        .tolist()
    )

    binarias = (
        mapa.loc[
            mapa[
                "Tipo"
            ]
            ==
            "binaria",
            "Feature"
        ]
        .tolist()
    )

    # ------------------------------------------------------------------------
    # DIMENSÕES / COLUNAS
    # ------------------------------------------------------------------------

    if len(features) != 400:
        raise RuntimeError(
            "Mapa não possui exatamente 400 features."
        )

    faltantes = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if faltantes:
        raise RuntimeError(
            f"{realizacao} {nome_split}: "
            f"features faltantes: {faltantes[:5]}"
        )

    # ------------------------------------------------------------------------
    # NaN / INF
    # ------------------------------------------------------------------------

    matriz = (
        df[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
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

    # ------------------------------------------------------------------------
    # LABELS
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
                [0, 1]
            )
        )
    )

    # ------------------------------------------------------------------------
    # RANGES FÍSICOS
    # ------------------------------------------------------------------------

    abaixo = 0
    acima = 0

    menor_normalizado = np.inf
    maior_normalizado = -np.inf

    for feature in floats:

        linha = parametros.loc[
            feature
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
            df[
                feature
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        abaixo += int(
            np.sum(
                valores
                <
                minimo
            )
        )

        acima += int(
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

        menor_normalizado = min(
            menor_normalizado,
            float(
                normalizado.min()
            )
        )

        maior_normalizado = max(
            maior_normalizado,
            float(
                normalizado.max()
            )
        )

    passou = (
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
        abaixo == 0
        and
        acima == 0
    )

    return {
        "Realizacao":
            realizacao,

        "Split":
            nome_split,

        "N_Linhas":
            len(df),

        "N_Features":
            len(features),

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
            abaixo,

        "Acima_Range":
            acima,

        "Min_Normalizado":
            menor_normalizado,

        "Max_Normalizado":
            maior_normalizado,

        "Aprovado":
            passou,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 86)
    print("AUDITORIA DAS REALIZAÇÕES CONFIRMATÓRIAS — NORMAL")
    print("=" * 86)

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
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

    if len(manifesto) != 5:
        raise RuntimeError(
            "Esperadas exatamente 5 realizações."
        )

    resultados = []

    for _, linha in manifesto.iterrows():

        realizacao = str(
            linha[
                "Realizacao"
            ]
        )

        print()
        print("=" * 86)
        print(realizacao)
        print("=" * 86)

        arquivos = [
            (
                "TREINO",
                linha[
                    "Arquivo_Treino"
                ]
            ),
            (
                "VALIDACAO",
                linha[
                    "Arquivo_Validacao"
                ]
            ),
            (
                "TESTE",
                linha[
                    "Arquivo_Teste"
                ]
            ),
        ]

        for nome_split, caminho in arquivos:

            df = pd.read_csv(
                caminho
            )

            resultado = auditar_split(
                realizacao,
                nome_split,
                df,
                mapa,
                parametros
            )

            resultados.append(
                resultado
            )

            print(
                f"{nome_split:10s} | "
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

    arquivo_saida = os.path.join(
        DIRETORIO_SAIDA,
        "auditoria_realizacoes.csv"
    )

    resultados_df.to_csv(
        arquivo_saida,
        index=False
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
    print("=" * 86)
    print("RESULTADO FINAL")
    print("=" * 86)

    print(
        f"Splits aprovados: "
        f"{aprovados} / {total}"
    )

    if aprovados == total:

        print(
            "AUDITORIA CONFIRMATÓRIA NORMAL: APROVADA"
        )

    else:

        print(
            "AUDITORIA CONFIRMATÓRIA NORMAL: REPROVADA"
        )

    print()
    print(
        "Resultado salvo em:"
    )

    print(
        arquivo_saida
    )


if __name__ == "__main__":
    main()