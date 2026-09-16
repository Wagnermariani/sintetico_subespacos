import os

import numpy as np
import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_BASE = os.path.join(
    PROJETO,
    "dados",
    "teste_normal_base_piloto.csv"
)

ARQUIVO_ATAQUE = os.path.join(
    PROJETO,
    "dados",
    "teste_ataque_piloto_m10_delta1.csv"
)

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_ataque_piloto_m10_delta1.csv"
)

ARQUIVO_MAPA = os.path.join(
    PROJETO,
    "dados",
    "mapa_grupos_verdadeiros.csv"
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "auditoria_ataques_piloto"
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True
)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 60)
    print("AUDITORIA ESTRUTURAL DOS ATAQUES — PILOTO")
    print("=" * 60)

    print("\nCarregando arquivos...")

    base = pd.read_csv(
        ARQUIVO_BASE
    )

    ataque = pd.read_csv(
        ARQUIVO_ATAQUE
    )

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    mapa = pd.read_csv(
        ARQUIVO_MAPA
    )

    features = (
        mapa["Feature"]
        .tolist()
    )

    print(
        f"Teste base: {base.shape}"
    )

    print(
        f"Teste ataque: {ataque.shape}"
    )

    print(
        f"Eventos: {len(eventos)}"
    )

    # ------------------------------------------------------------------------
    # DIMENSÕES
    # ------------------------------------------------------------------------

    if base.shape != ataque.shape:
        raise RuntimeError(
            "Base e ataque possuem dimensões diferentes."
        )

    # ------------------------------------------------------------------------
    # MÁSCARA DE TODOS OS EVENTOS
    # ------------------------------------------------------------------------

    mascara_eventos = np.zeros(
        len(ataque),
        dtype=bool
    )

    for _, evento in eventos.iterrows():

        inicio = int(
            evento[
                "Indice_Inicio"
            ]
        )

        fim = int(
            evento[
                "Indice_Fim"
            ]
        )

        mascara_eventos[
            inicio:fim + 1
        ] = True

    # ------------------------------------------------------------------------
    # FORA DOS ATAQUES NENHUMA FEATURE PODE MUDAR
    # ------------------------------------------------------------------------

    base_normal = (
        base.loc[
            ~mascara_eventos,
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    ataque_normal = (
        ataque.loc[
            ~mascara_eventos,
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    diferenca_fora = np.abs(
        ataque_normal
        -
        base_normal
    )

    max_diferenca_fora = float(
        diferenca_fora.max()
    )

    n_diferencas_fora = int(
        np.sum(
            diferenca_fora
            >
            1e-7
        )
    )

    print()
    print("=" * 60)
    print("FORA DOS EVENTOS")
    print("=" * 60)

    print(
        f"Maior diferença: "
        f"{max_diferenca_fora:.12f}"
    )

    print(
        f"Valores alterados fora dos eventos: "
        f"{n_diferencas_fora}"
    )

    # ------------------------------------------------------------------------
    # AUDITORIA EVENTO A EVENTO
    # ------------------------------------------------------------------------

    registros = []

    todos_eventos_ok = True

    print()
    print("=" * 60)
    print("EVENTO A EVENTO")
    print("=" * 60)

    mapa_grupo = (
        mapa
        .set_index(
            "Feature"
        )[
            "Grupo_Verdadeiro"
        ]
        .to_dict()
    )

    mapa_tipo = (
        mapa
        .set_index(
            "Feature"
        )[
            "Tipo"
        ]
        .to_dict()
    )

    for _, evento in eventos.iterrows():

        nome_evento = evento[
            "Evento"
        ]

        grupo_atacado = evento[
            "Grupo_Atacado"
        ]

        inicio = int(
            evento[
                "Indice_Inicio"
            ]
        )

        fim = int(
            evento[
                "Indice_Fim"
            ]
        )

        declaradas = (
            str(
                evento[
                    "Variaveis_Atacadas"
                ]
            )
            .split(";")
        )

        base_evento = (
            base.loc[
                inicio:fim,
                features
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        ataque_evento = (
            ataque.loc[
                inicio:fim,
                features
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        diferenca = np.abs(
            ataque_evento
            -
            base_evento
        )

        mudou_feature = np.any(
            diferenca
            >
            1e-7,
            axis=0
        )

        alteradas_observadas = [
            feature
            for feature, mudou
            in zip(
                features,
                mudou_feature
            )
            if mudou
        ]

        conjunto_declarado = set(
            declaradas
        )

        conjunto_observado = set(
            alteradas_observadas
        )

        somente_declaradas = (
            conjunto_declarado
            ==
            conjunto_observado
        )

        grupos_observados = sorted(
            {
                mapa_grupo[
                    feature
                ]
                for feature
                in alteradas_observadas
            }
        )

        apenas_um_grupo = (
            grupos_observados
            ==
            [grupo_atacado]
        )

        todas_float = all(
            mapa_tipo[
                feature
            ]
            ==
            "float"
            for feature
            in alteradas_observadas
        )

        n_alteradas = len(
            alteradas_observadas
        )

        evento_ok = (
            somente_declaradas
            and
            apenas_um_grupo
            and
            todas_float
            and
            n_alteradas == 10
        )

        todos_eventos_ok = (
            todos_eventos_ok
            and
            evento_ok
        )

        print(
            f"{nome_evento} | "
            f"{grupo_atacado} | "
            f"alteradas={n_alteradas:2d} | "
            f"grupo único={apenas_um_grupo} | "
            f"somente declaradas={somente_declaradas} | "
            f"floats={todas_float} | "
            f"{'OK' if evento_ok else 'ERRO'}"
        )

        registros.append(
            {
                "Evento":
                    nome_evento,

                "Grupo_Atacado":
                    grupo_atacado,

                "N_Alteradas":
                    n_alteradas,

                "Somente_Declaradas":
                    somente_declaradas,

                "Apenas_Grupo_Declarado":
                    apenas_um_grupo,

                "Todas_Float":
                    todas_float,

                "Evento_OK":
                    evento_ok,
            }
        )

    # ------------------------------------------------------------------------
    # LABEL ATTACK
    # ------------------------------------------------------------------------

    attack_esperado = (
        mascara_eventos
        .astype(
            np.int8
        )
    )

    attack_observado = (
        ataque[
            "Attack"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    labels_corretos = np.array_equal(
        attack_esperado,
        attack_observado
    )

    print()
    print("=" * 60)
    print("RÓTULOS")
    print("=" * 60)

    print(
        f"Attack coincide exatamente "
        f"com os eventos: {labels_corretos}"
    )

    # ------------------------------------------------------------------------
    # RESULTADO FINAL
    # ------------------------------------------------------------------------

    passou = (
        n_diferencas_fora == 0
        and
        todos_eventos_ok
        and
        labels_corretos
    )

    print()
    print("=" * 60)

    print(
        "AUDITORIA FINAL: "
        f"{'APROVADA' if passou else 'REPROVADA'}"
    )

    print("=" * 60)

    resultado = pd.DataFrame(
        registros
    )

    arquivo_saida = os.path.join(
        DIRETORIO_SAIDA,
        "auditoria_eventos.csv"
    )

    resultado.to_csv(
        arquivo_saida,
        index=False
    )

    print(
        "\nResultado salvo em:"
    )

    print(
        arquivo_saida
    )


if __name__ == "__main__":
    main()