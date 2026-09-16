import hashlib
import json
import os

import pandas as pd


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

DIRETORIO = os.path.join(
    PROJETO,
    "dados",
    "fronteira_estrutural",
)

os.makedirs(
    DIRETORIO,
    exist_ok=True,
)

ARQUIVO_GRADE = os.path.join(
    DIRETORIO,
    "grade_estrutural.csv",
)

ARQUIVO_PROTOCOLO = os.path.join(
    DIRETORIO,
    "protocolo_estrutural.json",
)

ARQUIVO_HASH = os.path.join(
    DIRETORIO,
    "sha256_protocolo_estrutural.txt",
)


# ============================================================================
# HASH
# ============================================================================

def sha256_arquivo(caminho):

    sha = hashlib.sha256()

    with open(caminho, "rb") as arquivo:

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
# CENÁRIOS
# ============================================================================

def adicionar_cenario(
    lista,
    cenario,
    eixo,
    alpha,
    beta,
    descricao,
):

    var_ruido = (
        1.0
        -
        alpha
        -
        beta
    )

    if var_ruido <= 0:

        raise ValueError(
            f"{cenario}: variância de ruído inválida."
        )

    lista.append(
        {
            "Cenario":
                cenario,

            "Eixo":
                eixo,

            "Alpha_Grupo":
                alpha,

            "Beta_Global":
                beta,

            "Var_Ruido":
                var_ruido,

            "Rho_Intra_Teorico":
                alpha
                +
                beta,

            "Rho_Inter_Teorico":
                beta,

            "Contraste_Grupos_Teorico":
                alpha,

            "Descricao":
                descricao,
        }
    )


def main():

    print("=" * 96)
    print("REGISTRO DA GRADE — EXPERIMENTO 4: FRONTEIRA ESTRUTURAL")
    print("=" * 96)

    cenarios = []

    # ========================================================================
    # EIXO A
    #
    # Beta global fixo em 0.05.
    # A coesão interna dos grupos é progressivamente aumentada.
    # ========================================================================

    adicionar_cenario(
        cenarios,
        "A01",
        "coesao_local",
        0.05,
        0.05,
        "Coesão local muito fraca.",
    )

    adicionar_cenario(
        cenarios,
        "A02",
        "coesao_local",
        0.15,
        0.05,
        "Coesão local fraca.",
    )

    adicionar_cenario(
        cenarios,
        "A03",
        "coesao_local",
        0.30,
        0.05,
        "Coesão local moderada-baixa.",
    )

    adicionar_cenario(
        cenarios,
        "A04",
        "coesao_local",
        0.45,
        0.05,
        "Coesão local moderada-alta.",
    )

    adicionar_cenario(
        cenarios,
        "A05",
        "coesao_local",
        0.65,
        0.05,
        "Configuração-base usada nos experimentos anteriores.",
    )

    adicionar_cenario(
        cenarios,
        "A06",
        "coesao_local",
        0.80,
        0.05,
        "Coesão local muito forte.",
    )

    # ========================================================================
    # EIXO B
    #
    # Rho intra teórico mantido em aproximadamente 0.70.
    #
    # À medida que beta aumenta, a diferença entre correlação intra e inter
    # diminui.
    #
    # A05 = alpha 0.65 / beta 0.05 já representa um ponto deste eixo
    # e não é duplicado.
    # ========================================================================

    adicionar_cenario(
        cenarios,
        "B01",
        "acoplamento_global",
        0.70,
        0.00,
        "Grupos fortes e praticamente sem fator global.",
    )

    adicionar_cenario(
        cenarios,
        "B02",
        "acoplamento_global",
        0.55,
        0.15,
        "Acoplamento global baixo-moderado.",
    )

    adicionar_cenario(
        cenarios,
        "B03",
        "acoplamento_global",
        0.40,
        0.30,
        "Acoplamento global moderado.",
    )

    adicionar_cenario(
        cenarios,
        "B04",
        "acoplamento_global",
        0.20,
        0.50,
        "Acoplamento global forte.",
    )

    adicionar_cenario(
        cenarios,
        "B05",
        "acoplamento_global",
        0.05,
        0.65,
        "Grupos apenas ligeiramente distinguíveis.",
    )

    adicionar_cenario(
        cenarios,
        "B06",
        "acoplamento_global",
        0.00,
        0.70,
        "Controle negativo: sem contraste estrutural entre grupos.",
    )

    grade = pd.DataFrame(
        cenarios
    )

    grade.to_csv(
        ARQUIVO_GRADE,
        index=False,
    )

    # ========================================================================
    # PROTOCOLO
    # ========================================================================

    protocolo = {
        "experimento":
            "Experimento 4 - Fronteira Estrutural",

        "natureza":
            "exploratorio_controlado",

        "pergunta":
            (
                "Quais propriedades observáveis somente no comportamento "
                "normal indicam que a decomposição em subespaços pode "
                "oferecer vantagem para detecção de anomalias?"
            ),

        "gerador":
            "codigo/gerar_normal_estrutural.py",

        "n_cenarios":
            int(
                len(
                    grade
                )
            ),

        "n_grupos":
            20,

        "features_por_grupo":
            20,

        "n_features":
            400,

        "n_treino":
            20000,

        "n_validacao":
            10000,

        "n_teste":
            20000,

        "seed_treino":
            6001,

        "seed_validacao":
            6002,

        "seed_teste":
            6003,

        "controle_seeds":
            (
                "As mesmas seeds serão usadas em todos os cenários para "
                "isolar o efeito da estrutura alpha/beta."
            ),

        "variancia_latente_total":
            1.0,

        "regra_variancia":
            "Var_Ruido = 1 - Alpha_Grupo - Beta_Global",

        "eixos": {
            "coesao_local":
                (
                    "Beta_Global fixo em 0.05; Alpha_Grupo varia."
                ),

            "acoplamento_global":
                (
                    "Alpha_Grupo + Beta_Global = 0.70; "
                    "Beta_Global aumenta progressivamente."
                ),
        },

        "diagnostico_normal_only":
            (
                "Burden Ratio e fatores estruturais calculados "
                "sem acesso a ataques."
            ),

        "avaliacao_posterior_com_ataques": {
            "uso":
                (
                    "Somente para verificar posteriormente se o indicador "
                    "normal-only acompanha a vantagem real. "
                    "Os ataques não entram no diagnóstico."
                ),

            "probes": [
                {
                    "m":
                        8,

                    "delta_latente":
                        1.25,
                },
                {
                    "m":
                        8,

                    "delta_latente":
                        1.50,
                },
            ],

            "geometria_eventos":
                "dados/eventos_confirmatorios_base.csv",

            "ordem_features":
                "dados/ordem_features_confirmatorias.csv",
        },

        "observacao_metodologica":
            (
                "Este experimento não altera nem reabre os experimentos "
                "exploratório e confirmatório anteriores. "
                "Não contém drift e não utiliza RNN."
            ),
    }

    with open(
        ARQUIVO_PROTOCOLO,
        "w",
        encoding="utf-8",
    ) as arquivo:

        json.dump(
            protocolo,
            arquivo,
            indent=4,
            ensure_ascii=False,
        )

    hash_grade = sha256_arquivo(
        ARQUIVO_GRADE
    )

    hash_protocolo = sha256_arquivo(
        ARQUIVO_PROTOCOLO
    )

    with open(
        ARQUIVO_HASH,
        "w",
        encoding="utf-8",
    ) as arquivo:

        arquivo.write(
            f"{hash_grade}  "
            f"dados/fronteira_estrutural/grade_estrutural.csv\n"
        )

        arquivo.write(
            f"{hash_protocolo}  "
            f"dados/fronteira_estrutural/protocolo_estrutural.json\n"
        )

    # ========================================================================
    # SAÍDA
    # ========================================================================

    print()

    print(
        grade[
            [
                "Cenario",
                "Eixo",
                "Alpha_Grupo",
                "Beta_Global",
                "Var_Ruido",
                "Rho_Intra_Teorico",
                "Rho_Inter_Teorico",
                "Contraste_Grupos_Teorico",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Total de cenários: {len(grade)}"
    )

    print()
    print(
        f"Hash grade:     {hash_grade}"
    )

    print(
        f"Hash protocolo: {hash_protocolo}"
    )

    print()
    print("=" * 96)
    print("GRADE ESTRUTURAL REGISTRADA")
    print("=" * 96)

    print()
    print(
        ARQUIVO_GRADE
    )

    print(
        ARQUIVO_PROTOCOLO
    )

    print(
        ARQUIVO_HASH
    )


if __name__ == "__main__":
    main()