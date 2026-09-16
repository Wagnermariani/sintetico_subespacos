import os

import pandas as pd


PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_RESULTADOS = os.path.join(
    PROJETO,
    "resultado",
    "confirmacao",
    "resultados_confirmatorios.csv",
)

ARQUIVO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "confirmacao",
    "estabilidade_mahalanobis_global_local.csv",
)


def main():

    print("=" * 92)
    print("ESTABILIDADE CONFIRMATÓRIA — MAHALANOBIS GLOBAL × LOCAL")
    print("=" * 92)

    df = pd.read_csv(
        ARQUIVO_RESULTADOS
    )

    global_df = (
        df[
            df["Metodo"]
            ==
            "Global_Mahalanobis"
        ][
            [
                "Realizacao",
                "Condicao",
                "m",
                "delta_latente",
                "Recall",
                "FPR",
            ]
        ]
        .rename(
            columns={
                "Recall":
                    "Recall_Global",

                "FPR":
                    "FPR_Global",
            }
        )
    )

    local_df = (
        df[
            df["Metodo"]
            ==
            "Local_Mahalanobis"
        ][
            [
                "Realizacao",
                "Condicao",
                "m",
                "delta_latente",
                "Recall",
                "FPR",
            ]
        ]
        .rename(
            columns={
                "Recall":
                    "Recall_Local",

                "FPR":
                    "FPR_Local",
            }
        )
    )

    pares = global_df.merge(
        local_df,
        on=[
            "Realizacao",
            "Condicao",
            "m",
            "delta_latente",
        ],
        how="inner",
    )

    if len(pares) != 30:

        raise RuntimeError(
            f"Esperados 30 pares. Encontrados: {len(pares)}"
        )

    pares[
        "Delta_Recall"
    ] = (
        pares[
            "Recall_Local"
        ]
        -
        pares[
            "Recall_Global"
        ]
    )

    pares[
        "Local_Venceu"
    ] = (
        pares[
            "Delta_Recall"
        ]
        >
        0
    )

    resumo = (
        pares
        .groupby(
            [
                "Condicao",
                "m",
                "delta_latente",
            ],
            as_index=False,
        )
        .agg(
            Recall_Global_Medio=(
                "Recall_Global",
                "mean",
            ),

            Recall_Global_DP=(
                "Recall_Global",
                "std",
            ),

            Recall_Global_Min=(
                "Recall_Global",
                "min",
            ),

            Recall_Global_Max=(
                "Recall_Global",
                "max",
            ),

            Recall_Local_Medio=(
                "Recall_Local",
                "mean",
            ),

            Recall_Local_DP=(
                "Recall_Local",
                "std",
            ),

            Recall_Local_Min=(
                "Recall_Local",
                "min",
            ),

            Recall_Local_Max=(
                "Recall_Local",
                "max",
            ),

            FPR_Global_Medio=(
                "FPR_Global",
                "mean",
            ),

            FPR_Local_Medio=(
                "FPR_Local",
                "mean",
            ),

            Delta_Recall_Medio=(
                "Delta_Recall",
                "mean",
            ),

            Delta_Recall_DP=(
                "Delta_Recall",
                "std",
            ),

            Delta_Recall_Min=(
                "Delta_Recall",
                "min",
            ),

            Delta_Recall_Max=(
                "Delta_Recall",
                "max",
            ),

            Vitorias_Local=(
                "Local_Venceu",
                "sum",
            ),
        )
    )

    resumo.to_csv(
        ARQUIVO_SAIDA,
        index=False,
    )

    print()
    print(
        resumo[
            [
                "Condicao",
                "m",
                "delta_latente",
                "Recall_Global_Medio",
                "Recall_Local_Medio",
                "Delta_Recall_Medio",
                "Delta_Recall_Min",
                "Delta_Recall_Max",
                "Vitorias_Local",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 92)

    total_vitorias = int(
        pares[
            "Local_Venceu"
        ].sum()
    )

    print(
        f"Vitórias Local_Mahalanobis "
        f"contra Global_Mahalanobis: "
        f"{total_vitorias} / 30"
    )

    print()

    print(
        "Menor ganho observado: "
        f"{pares['Delta_Recall'].min():.6f}"
    )

    print(
        "Maior ganho observado: "
        f"{pares['Delta_Recall'].max():.6f}"
    )

    print(
        "Ganho médio geral: "
        f"{pares['Delta_Recall'].mean():.6f}"
    )

    print("=" * 92)

    print()
    print(
        "Resultado salvo em:"
    )

    print(
        ARQUIVO_SAIDA
    )


if __name__ == "__main__":
    main()