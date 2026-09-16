import os

import numpy as np
import pandas as pd

import gerar_normal_sintetico as original
import gerar_normal_estrutural as estrutural


PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)


def main():

    print("=" * 88)
    print("TESTE DE REGRESSÃO — GERADOR ORIGINAL × GERADOR ESTRUTURAL")
    print("=" * 88)

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

    n = 2000
    seed = 987654
    timestamp_inicial = "2030-01-01 00:00:00"

    print()
    print("Gerando com o código original...")

    df_original, mapa_original = (
        original.gerar_conjunto_normal(
            n=n,
            seed=seed,
            timestamp_inicial=timestamp_inicial,
            parametros_sensores=parametros_sensores,
        )
    )

    print()
    print("Gerando com o código estrutural parametrizado...")

    df_estrutural, mapa_estrutural = (
        estrutural.gerar_conjunto_normal(
            n=n,
            seed=seed,
            timestamp_inicial=timestamp_inicial,
            parametros_sensores=parametros_sensores,
            alpha_grupo=0.65,
            beta_global=0.05,
        )
    )

    # ------------------------------------------------------------------------
    # COLUNAS E MAPA
    # ------------------------------------------------------------------------

    colunas_iguais = (
        list(df_original.columns)
        ==
        list(df_estrutural.columns)
    )

    mapa_igual = mapa_original.equals(
        mapa_estrutural
    )

    # ------------------------------------------------------------------------
    # METADADOS
    # ------------------------------------------------------------------------

    timestamp_igual = (
        df_original["timestamp"]
        .equals(
            df_estrutural["timestamp"]
        )
    )

    attack_igual = (
        df_original["Attack"]
        .equals(
            df_estrutural["Attack"]
        )
    )

    tipo_igual = (
        df_original["Tipo"]
        .equals(
            df_estrutural["Tipo"]
        )
    )

    # ------------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------------

    features = mapa_original[
        "Feature"
    ].tolist()

    X_original = (
        df_original[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_estrutural = (
        df_estrutural[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    diferenca_maxima = float(
        np.max(
            np.abs(
                X_original
                -
                X_estrutural
            )
        )
    )

    valores_exatamente_iguais = np.array_equal(
        X_original,
        X_estrutural
    )

    # ------------------------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------------------------

    print()
    print("=" * 88)
    print("RESULTADO")
    print("=" * 88)

    print(
        f"Colunas iguais:             {colunas_iguais}"
    )

    print(
        f"Mapa de features igual:     {mapa_igual}"
    )

    print(
        f"Timestamp igual:            {timestamp_igual}"
    )

    print(
        f"Attack igual:               {attack_igual}"
    )

    print(
        f"Tipo igual:                 {tipo_igual}"
    )

    print(
        f"Valores exatamente iguais:  {valores_exatamente_iguais}"
    )

    print(
        f"Diferença máxima:           {diferenca_maxima:.20e}"
    )

    aprovado = (
        colunas_iguais
        and
        mapa_igual
        and
        timestamp_igual
        and
        attack_igual
        and
        tipo_igual
        and
        valores_exatamente_iguais
        and
        diferenca_maxima == 0.0
    )

    print()

    if aprovado:

        print(
            "TESTE DE REGRESSÃO: APROVADO"
        )

    else:

        print(
            "TESTE DE REGRESSÃO: REPROVADO"
        )

        raise RuntimeError(
            "O gerador estrutural alterou o comportamento original."
        )


if __name__ == "__main__":
    main()