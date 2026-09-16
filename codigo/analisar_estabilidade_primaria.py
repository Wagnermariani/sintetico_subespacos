import os

import pandas as pd


PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_ENTRADA = os.path.join(
    PROJETO,
    "resultado",
    "confirmacao",
    "comparacao_primaria.csv",
)

ARQUIVO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "confirmacao",
    "estabilidade_comparacao_primaria.csv",
)


def main():

    print("=" * 96)
    print("ESTABILIDADE CONFIRMATÓRIA — LOCAL MAHALANOBIS × MELHOR GLOBAL")
    print("=" * 96)

    df = pd.read_csv(
        ARQUIVO_ENTRADA
    )

    if len(df) != 30:
        raise RuntimeError(
            f"Esperadas 30 comparações. Encontradas: {len(df)}"
        )

    df[
        "Local_Venceu"
    ] = (
        df[
            "Delta_Recall_Local_Menos_Global"
        ]
        >
        0
    )

    resumo = (
        df
        .groupby(
            [
                "Condicao",
                "m",
                "delta_latente",
            ],
            as_index=False,
        )
        .agg(
            Recall_Melhor_Global_Medio=(
                "Recall_Melhor_Global",
                "mean",
            ),

            Recall_Melhor_Global_DP=(
                "Recall_Melhor_Global",
                "std",
            ),

            Recall_Local_Medio=(
                "Recall_Local_Mahalanobis",
                "mean",
            ),

            Recall_Local_DP=(
                "Recall_Local_Mahalanobis",
                "std",
            ),

            FPR_Melhor_Global_Medio=(
                "FPR_Melhor_Global",
                "mean",
            ),

            FPR_Local_Medio=(
                "FPR_Local_Mahalanobis",
                "mean",
            ),

            Delta_Recall_Medio=(
                "Delta_Recall_Local_Menos_Global",
                "mean",
            ),

            Delta_Recall_DP=(
                "Delta_Recall_Local_Menos_Global",
                "std",
            ),

            Delta_Recall_Min=(
                "Delta_Recall_Local_Menos_Global",
                "min",
            ),

            Delta_Recall_Max=(
                "Delta_Recall_Local_Menos_Global",
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
                "Recall_Melhor_Global_Medio",
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
    print("=" * 96)

    total_vitorias = int(
        df[
            "Local_Venceu"
        ].sum()
    )

    print(
        "Vitórias Local_Mahalanobis contra "
        "o melhor baseline global: "
        f"{total_vitorias} / 30"
    )

    print()

    print(
        "Menor ganho observado: "
        f"{df['Delta_Recall_Local_Menos_Global'].min():.6f}"
    )

    print(
        "Maior ganho observado: "
        f"{df['Delta_Recall_Local_Menos_Global'].max():.6f}"
    )

    print(
        "Ganho médio geral: "
        f"{df['Delta_Recall_Local_Menos_Global'].mean():.6f}"
    )

    print()

    print(
        "Recall médio do melhor global: "
        f"{df['Recall_Melhor_Global'].mean():.6f}"
    )

    print(
        "Recall médio do Local_Mahalanobis: "
        f"{df['Recall_Local_Mahalanobis'].mean():.6f}"
    )

    print()

    print(
        "FPR médio do melhor global: "
        f"{df['FPR_Melhor_Global'].mean():.6f}"
    )

    print(
        "FPR médio do Local_Mahalanobis: "
        f"{df['FPR_Local_Mahalanobis'].mean():.6f}"
    )

    print("=" * 96)

    print()
    print(
        "Resultado salvo em:"
    )

    print(
        ARQUIVO_SAIDA
    )


if __name__ == "__main__":
    main()