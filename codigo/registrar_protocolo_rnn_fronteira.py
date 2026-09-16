import hashlib
import json
import os

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

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_confirmatorios_base.csv",
)

ARQUIVO_ORDEM = os.path.join(
    PROJETO,
    "dados",
    "ordem_features_confirmatorias.csv",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

CENARIOS_FRONTEIRA = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
]

CENARIO_REFERENCIA_PREVIA = "A06"


def caminho_grupos(cenario):

    return os.path.join(
        PROJETO,
        "resultado",
        "fronteira_estrutural",
        "descoberta_grupos",
        cenario,
        "mapa_grupos_descobertos.csv",
    )


ARQUIVO_RNN_BASE = os.path.join(
    PROJETO,
    "codigo",
    "rnn_hai_base.py",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "dados",
    "rnn_fronteira",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_PROTOCOLO = os.path.join(
    DIRETORIO_SAIDA,
    "protocolo_rnn_fronteira.json",
)

ARQUIVO_HASH = os.path.join(
    DIRETORIO_SAIDA,
    "sha256_protocolo_rnn_fronteira.txt",
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

            sha.update(bloco)

    return sha.hexdigest()


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 100)
    print("REGISTRO DO PROTOCOLO — FRONTEIRA RNN")
    print("=" * 100)

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    dados_cenarios = {}

    arquivos_fonte = {
        "eventos":
            ARQUIVO_EVENTOS,

        "ordem_features_ataque":
            ARQUIVO_ORDEM,

        "parametros_fisicos":
            ARQUIVO_PARAMETROS,

        "rnn_hai_usada_como_referencia":
            ARQUIVO_RNN_BASE,
    }

    for cenario in CENARIOS_FRONTEIRA:

        linha = manifesto.loc[
            manifesto["Cenario"] == cenario
        ]

        if len(linha) != 1:

            raise RuntimeError(
                f"{cenario} não encontrado de forma única no manifesto."
            )

        linha = linha.iloc[0]

        alpha = float(
            linha["Alpha_Grupo"]
        )

        beta = float(
            linha["Beta_Global"]
        )

        arquivo_treino = str(
            linha["Arquivo_Treino"]
        )

        arquivo_validacao = str(
            linha["Arquivo_Validacao"]
        )

        arquivo_teste = str(
            linha["Arquivo_Teste"]
        )

        arquivo_grupos = caminho_grupos(
            cenario
        )

        grupos = pd.read_csv(
            arquivo_grupos
        )

        if len(grupos) != 400:

            raise RuntimeError(
                f"{cenario}: a partição deve conter 400 linhas."
            )

        if grupos["Feature"].nunique() != 400:

            raise RuntimeError(
                f"{cenario}: as 400 features devem aparecer exatamente uma vez."
            )

        contagens = (
            grupos[
                "Grupo_Descoberto"
            ]
            .value_counts()
        )

        n_grupos = int(
            len(contagens)
        )

        tamanho_min = int(
            contagens.min()
        )

        tamanho_mediano = float(
            contagens.median()
        )

        tamanho_max = int(
            contagens.max()
        )

        dados_cenarios[
            cenario
        ] = {
            "alpha_grupo":
                alpha,

            "beta_global":
                beta,

            "treino":
                arquivo_treino,

            "validacao":
                arquivo_validacao,

            "teste_normal_base":
                arquivo_teste,

            "particao":
                arquivo_grupos,

            "n_grupos_descobertos":
                n_grupos,

            "tamanho_grupo_min":
                tamanho_min,

            "tamanho_grupo_mediano":
                tamanho_mediano,

            "tamanho_grupo_max":
                tamanho_max,
        }

        arquivos_fonte[
            f"treino_normal_{cenario}"
        ] = arquivo_treino

        arquivos_fonte[
            f"validacao_normal_{cenario}"
        ] = arquivo_validacao

        arquivos_fonte[
            f"teste_normal_base_{cenario}"
        ] = arquivo_teste

        arquivos_fonte[
            f"particao_descoberta_{cenario}"
        ] = arquivo_grupos

        print(
            f"{cenario}: "
            f"alpha={alpha:.2f} | "
            f"beta={beta:.2f} | "
            f"K={n_grupos} | "
            f"tam={tamanho_min}/{tamanho_mediano:.1f}/{tamanho_max}"
        )


    hashes_fonte = {}

    for nome, caminho in arquivos_fonte.items():

        if not os.path.exists(
            caminho
        ):

            raise FileNotFoundError(
                caminho
            )

        hashes_fonte[
            nome
        ] = sha256_arquivo(
            caminho
        )

    # ------------------------------------------------------------------------
    # PROTOCOLO
    # ------------------------------------------------------------------------

    protocolo = {

        "experimento": {
            "nome":
                "RNN_fronteira_estrutural_global_vs_local",

            "natureza":
                "experimento_controlado_pre_registrado",

            "objetivo":
                (
                    "Verificar, em diferentes regimes estruturais previamente "
                    "congelados, quando a avaliacao local dos mesmos residuos "
                    "produzidos por uma RNN-AE monolitica apresenta vantagem "
                    "sobre a avaliacao global."
                ),

            "objetivo_diagnostico":
                (
                    "Avaliar se propriedades calculadas exclusivamente a partir "
                    "dos residuos normais permitem distinguir regioes favoraveis, "
                    "incertas e desfavoraveis a decomposicao."
                ),

            "cenarios_prospectivos":
                CENARIOS_FRONTEIRA,

            "n_cenarios_prospectivos":
                len(CENARIOS_FRONTEIRA),

            "referencia_previa_ja_observada":
                CENARIO_REFERENCIA_PREVIA,

            "observacao_referencia":
                (
                    "A06 foi executado anteriormente como primeiro experimento "
                    "RNN e nao e tratado como resultado prospectivo desta bateria."
                ),
        },

        "dados": {
            "n_features":
                400,

            "normalizacao":
                (
                    "MinMax fisico conhecido para features continuas; "
                    "features binarias permanecem 0/1."
                ),

            "cenarios":
                dados_cenarios,

            "particoes":
                (
                    "Para cada cenario e usada exclusivamente a particao "
                    "previamente descoberta a partir de seus dados normais. "
                    "O numero e o tamanho dos grupos nao sao forçados."
                ),
        },

        "rnn": {
            "tipo":
                "RNN Autoencoder",

            "estrategia":
                (
                    "Uma RNN monolitica independente e treinada para cada "
                    "cenario, mantendo exatamente a mesma arquitetura e "
                    "os mesmos hiperparametros."
                ),

            "input_dim":
                400,

            "window_size":
                60,

            "batch_size":
                512,

            "hidden_size":
                64,

            "num_layers":
                1,

            "dropout":
                0.0,

            "nonlinearity":
                "tanh",

            "decoder_input":
                "sequencia de zeros inicializada pelo hidden final do encoder",

            "output_layer":
                "Linear(hidden_size, 400)",

            "optimizer":
                "Adam",

            "learning_rate":
                1e-3,

            "loss_treino":
                "MSE da janela completa",

            "max_epochs":
                100,

            "early_stopping":
                True,

            "early_stopping_patience":
                5,

            "checkpoint":
                "menor perda MSE na validacao normal",

            "seed":
                42,

            "ajuste_arquitetura_por_cenario":
                False,
        },

        "residuos": {
            "instante_avaliado":
                "ultimo instante de cada janela",

            "definicao":
                "e_t = x_t - xhat_t",

            "dimensao":
                400,

            "principio_comparacao":
                (
                    "Dentro de cada cenario, global e local recebem exatamente "
                    "os mesmos residuos produzidos pela mesma RNN e pelo mesmo "
                    "checkpoint."
                ),
        },

        "diagnostico_normal_only": {
            "momento":
                (
                    "depois do treinamento da RNN e antes da utilizacao "
                    "de qualquer ataque do teste"
                ),

            "dados_permitidos": [
                "residuos normais do treino",
                "residuos normais da validacao",
                "particao descoberta a partir do normal",
            ],

            "dados_proibidos": [
                "rotulos de ataque",
                "scores do teste atacado",
                "resultado dos probes",
            ],

            "finalidade":
                (
                    "Caracterizar previamente a geometria global e local "
                    "no espaco residual da RNN."
                ),
        },

        "score_global": {
            "metodo":
                "LedoitWolf Mahalanobis",

            "ajuste_covariancia":
                "residuos normais do treino",

            "score":
                "Mahalanobis sobre as 400 dimensoes residuais",

            "limiar":
                "percentil 99 dos scores da validacao normal",
        },

        "score_local": {
            "metodo":
                "LedoitWolf Mahalanobis",

            "ajuste_covariancia":
                (
                    "residuos normais do treino separadamente para "
                    "cada grupo descoberto no respectivo cenario"
                ),

            "numero_grupos":
                "determinado pela particao descoberta de cada cenario",

            "dimensao_grupos":
                "determinada pela particao descoberta de cada cenario",

            "fusao":
                "maximo dos scores Mahalanobis locais",

            "limiar":
                (
                    "percentil 99 do score de fusao maximo "
                    "na validacao normal"
                ),
        },

        "probes": [
            {
                "id":
                    "P01",

                "m":
                    8,

                "delta_latente":
                    1.25,
            },
            {
                "id":
                    "P02",

                "m":
                    8,

                "delta_latente":
                    1.50,
            },
        ],

        "ataques": {
            "eventos":
                "mesmos 20 eventos congelados anteriormente",

            "duracao":
                "100 segundos por evento",

            "ordem_features":
                "mesma ordem congelada anteriormente",

            "uso":
                (
                    "somente no teste, depois do treinamento da RNN, "
                    "extracao dos residuos normais, ajuste das covariancias, "
                    "calibracao dos limiares e diagnostico normal-only"
                ),
        },

        "avaliacao": {
            "comparacao_primaria":
                "Recall_Local - Recall_Global",

            "metricas_secundarias": [
                "FPR",
                "Precision",
                "F1",
                "MCC",
            ],

            "sem_calibracao_no_teste":
                True,

            "criterio_vantagem_observada":
                (
                    "Recall_Local > Recall_Global sob limiares independentes "
                    "calibrados para FPR nominal de 1% na mesma validacao normal"
                ),

            "questao_principal":
                (
                    "Verificar a correspondencia entre a geometria observada "
                    "somente no normal e a vantagem ou ausencia de vantagem "
                    "local posteriormente observada nos probes."
                ),
        },

        "restricoes": {
            "uma_unica_rnn_por_cenario":
                True,

            "mesma_arquitetura_em_todos_os_cenarios":
                True,

            "nenhuma_rnn_local":
                True,

            "nenhum_ajuste_de_arquitetura_apos_ver_o_teste":
                True,

            "nenhum_ataque_usado_no_treino":
                True,

            "nenhum_ataque_usado_na_validacao":
                True,

            "nenhum_ataque_usado_no_diagnostico_normal_only":
                True,

            "nenhuma_particao_forcada_para_20x20":
                True,
        },

        "hashes_fontes":
            hashes_fonte,
    }

    # ------------------------------------------------------------------------
    # SALVA PROTOCOLO
    # ------------------------------------------------------------------------

    with open(
        ARQUIVO_PROTOCOLO,
        "w",
        encoding="utf-8",
    ) as arquivo:

        json.dump(
            protocolo,
            arquivo,
            indent=2,
            ensure_ascii=False,
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
            hash_protocolo + "\n"
        )


    print()
    print("=" * 100)
    print("PROTOCOLO DA FRONTEIRA RNN REGISTRADO")
    print("=" * 100)

    print()
    print(
        f"Cenários prospectivos: {len(CENARIOS_FRONTEIRA)}"
    )

    print(
        "Cenários: "
        +
        ", ".join(
            CENARIOS_FRONTEIRA
        )
    )

    print(
        f"Referência prévia já observada: {CENARIO_REFERENCIA_PREVIA}"
    )

    print()
    print(
        "RNN por cenário: 400 -> hidden 64 -> 400 | janela=60 | batch=512"
    )

    print(
        "Mesma arquitetura e mesmos hiperparâmetros em todos os cenários."
    )

    print(
        "Treino: MSE da janela completa | Adam 1e-3"
    )

    print(
        "Early stopping: patience=5 | máximo=100 épocas"
    )

    print()
    print(
        "Comparação: Mahalanobis global versus Mahalanobis local "
        "sobre os mesmos resíduos."
    )

    print(
        "Limiar: P99 da validação normal."
    )

    print(
        "Diagnóstico estrutural: somente treino/validação normais."
    )

    print(
        "Probes: m=8/delta=1.25 e m=8/delta=1.50."
    )

    print()
    print(
        f"SHA256 protocolo: {hash_protocolo}"
    )

    print()
    print(
        f"Arquivo: {ARQUIVO_PROTOCOLO}"
    )

    print(
        f"Hash:    {ARQUIVO_HASH}"
    )

if __name__ == "__main__":
    main()