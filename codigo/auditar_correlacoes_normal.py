import os

import numpy as np
import pandas as pd


DIRETORIO_PROJETO = (
    "/home/wagner/areas/sintetico_subespacos"
)

ARQUIVO_TREINO = os.path.join(
    DIRETORIO_PROJETO,
    "dados",
    "treino_normal_piloto.csv"
)

ARQUIVO_MAPA = os.path.join(
    DIRETORIO_PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv"
)

DIRETORIO_SAIDA = os.path.join(
    DIRETORIO_PROJETO,
    "resultado",
    "auditoria_correlacoes"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


def tipo_par(tipo_a, tipo_b):

    tipos = sorted(
        [tipo_a, tipo_b]
    )

    if tipos == [
        "float",
        "float"
    ]:
        return "float-float"

    if tipos == [
        "binaria",
        "float"
    ]:
        return "float-binaria"

    if tipos == [
        "binaria",
        "binaria"
    ]:
        return "binaria-binaria"

    return "outro"


def resumo(nome, valores):

    valores = np.asarray(
        valores,
        dtype=np.float64
    )

    print(
        f"{nome:28s} "
        f"N={len(valores):6d} "
        f"min={np.min(valores):.6f} "
        f"P01={np.percentile(valores, 1):.6f} "
        f"med={np.median(valores):.6f} "
        f"P99={np.percentile(valores, 99):.6f} "
        f"max={np.max(valores):.6f}"
    )


def main():

    print(
        "Carregando treino normal..."
    )

    df = pd.read_csv(
        ARQUIVO_TREINO
    )

    mapa = pd.read_csv(
        ARQUIVO_MAPA
    )

    features = (
        mapa["Feature"]
        .astype(str)
        .tolist()
    )

    if len(features) != 400:
        raise RuntimeError(
            f"Esperadas 400 features, "
            f"encontradas {len(features)}."
        )

    faltantes = [
        f
        for f in features
        if f not in df.columns
    ]

    if faltantes:
        raise RuntimeError(
            "Features ausentes no treino: "
            + ", ".join(
                faltantes[:10]
            )
        )

    # ================================================================
    # CORRELAÇÃO DE PEARSON
    # ================================================================

    print(
        "Calculando matriz de correlação..."
    )

    corr = (
        df[features]
        .corr(method="pearson")
    )

    grupo = dict(
        zip(
            mapa["Feature"],
            mapa["Grupo_Verdadeiro"]
        )
    )

    tipo = dict(
        zip(
            mapa["Feature"],
            mapa["Tipo"]
        )
    )

    registros = []

    for i in range(
        len(features)
    ):

        fi = features[i]

        for j in range(
            i + 1,
            len(features)
        ):

            fj = features[j]

            valor = float(
                corr.loc[
                    fi,
                    fj
                ]
            )

            mesmo_grupo = (
                grupo[fi]
                ==
                grupo[fj]
            )

            registros.append(
                {
                    "Feature_A":
                        fi,

                    "Feature_B":
                        fj,

                    "Grupo_A":
                        grupo[fi],

                    "Grupo_B":
                        grupo[fj],

                    "Mesmo_Grupo":
                        mesmo_grupo,

                    "Tipo_Par":
                        tipo_par(
                            tipo[fi],
                            tipo[fj]
                        ),

                    "Correlacao":
                        valor,

                    "Abs_Correlacao":
                        abs(valor),
                }
            )

    pares = pd.DataFrame(
        registros
    )

    intra = pares[
        pares["Mesmo_Grupo"]
    ]

    inter = pares[
        ~pares["Mesmo_Grupo"]
    ]

    # ================================================================
    # RESUMO GLOBAL
    # ================================================================

    print(
        "\n============================================"
    )
    print(
        "SEPARAÇÃO GLOBAL"
    )
    print(
        "============================================"
    )

    resumo(
        "INTRA |rho|",
        intra["Abs_Correlacao"]
    )

    resumo(
        "INTER |rho|",
        inter["Abs_Correlacao"]
    )

    rho_intra_min = float(
        intra[
            "Abs_Correlacao"
        ].min()
    )

    rho_inter_max = float(
        inter[
            "Abs_Correlacao"
        ].max()
    )

    margem = (
        rho_intra_min
        -
        rho_inter_max
    )

    print(
        "\nMínimo INTRA: "
        f"{rho_intra_min:.6f}"
    )

    print(
        "Máximo INTER: "
        f"{rho_inter_max:.6f}"
    )

    print(
        "Margem de separação: "
        f"{margem:.6f}"
    )

    if margem > 0:

        corte_sugerido = (
            rho_inter_max
            +
            rho_intra_min
        ) / 2.0

        print(
            "\nSEPARAÇÃO PERFEITA OBSERVADA."
        )

        print(
            "Existe um corte entre "
            "INTER e INTRA."
        )

        print(
            "Corte intermediário possível: "
            f"{corte_sugerido:.6f}"
        )

    else:

        corte_sugerido = np.nan

        print(
            "\nATENÇÃO:"
        )

        print(
            "Há sobreposição entre "
            "correlações intra e inter."
        )

    # ================================================================
    # RESUMO POR TIPO
    # ================================================================

    print(
        "\n============================================"
    )
    print(
        "SEPARAÇÃO POR TIPO DE PAR"
    )
    print(
        "============================================"
    )

    tipos_pares = [
        "float-float",
        "float-binaria",
        "binaria-binaria",
    ]

    resumo_tipos = []

    for tp in tipos_pares:

        intra_tp = intra[
            intra["Tipo_Par"]
            == tp
        ]

        inter_tp = inter[
            inter["Tipo_Par"]
            == tp
        ]

        print(
            f"\n--- {tp} ---"
        )

        resumo(
            "INTRA |rho|",
            intra_tp[
                "Abs_Correlacao"
            ]
        )

        resumo(
            "INTER |rho|",
            inter_tp[
                "Abs_Correlacao"
            ]
        )

        intra_min = float(
            intra_tp[
                "Abs_Correlacao"
            ].min()
        )

        inter_max = float(
            inter_tp[
                "Abs_Correlacao"
            ].max()
        )

        margem_tp = (
            intra_min
            -
            inter_max
        )

        print(
            "Margem: "
            f"{margem_tp:.6f}"
        )

        resumo_tipos.append(
            {
                "Tipo_Par":
                    tp,

                "Rho_Intra_Min":
                    intra_min,

                "Rho_Inter_Max":
                    inter_max,

                "Margem":
                    margem_tp,
            }
        )

    # ================================================================
    # SALVAR
    # ================================================================

    pares.to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "pares_correlacoes_treino.csv"
        ),
        index=False
    )

    pd.DataFrame(
        resumo_tipos
    ).to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_por_tipo.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [
            {
                "Rho_Intra_Min":
                    rho_intra_min,

                "Rho_Inter_Max":
                    rho_inter_max,

                "Margem":
                    margem,

                "Corte_Intermediario":
                    corte_sugerido,
            }
        ]
    ).to_csv(
        os.path.join(
            DIRETORIO_SAIDA,
            "resumo_global.csv"
        ),
        index=False
    )

    print(
        "\nResultados salvos em:"
    )

    print(
        DIRETORIO_SAIDA
    )


if __name__ == "__main__":
    main()