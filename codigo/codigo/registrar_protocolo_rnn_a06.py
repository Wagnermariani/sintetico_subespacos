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

ARQUIVO_GRUPOS_A06 = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "descoberta_grupos",
    "A06",
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
    "rnn_a06",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_PROTOCOLO = os.path.join(
    DIRETORIO_SAIDA,
    "protocolo_rnn_a06.json",
)

ARQUIVO_HASH = os.path.join(
    DIRETORIO_SAIDA,
    "sha256_protocolo_rnn_a06.txt",
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
    print("REGISTRO DO PROTOCOLO — RNN A06")
    print("=" * 100)

    manifesto = pd.read_csv(
        ARQUIVO_MANIFESTO
    )

    linha = manifesto.loc[
        manifesto["Cenario"] == "A06"
    ]

    if len(linha) != 1:

        raise RuntimeError(
            "A06 não encontrado de forma única no manifesto."
        )

    linha = linha.iloc[0]

    alpha = float(
        linha["Alpha_Grupo"]
    )

    beta = float(
        linha["Beta_Global"]
    )

    if not (
        abs(alpha - 0.80) < 1e-12
        and
        abs(beta - 0.05) < 1e-12
    ):

        raise RuntimeError(
            f"A06 inesperado: alpha={alpha}, beta={beta}"
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

    # ------------------------------------------------------------------------
    # AUDITORIA DA PARTIÇÃO
    # ------------------------------------------------------------------------

    grupos = pd.read_csv(
        ARQUIVO_GRUPOS_A06
    )

    contagens = (
        grupos[
            "Grupo_Descoberto"
        ]
        .value_counts()
    )

    if len(grupos) != 400:

        raise RuntimeError(
            "A partição A06 deve conter 400 features."
        )

    if len(contagens) != 20:

        raise RuntimeError(
            "A partição A06 deve conter 20 grupos."
        )

    if not (
        contagens == 20
    ).all():

        raise RuntimeError(
            "Todos os grupos A06 devem ter tamanho 20."
        )

    # ------------------------------------------------------------------------
    # ARQUIVOS QUE FICAM CONGELADOS COMO ENTRADA
    # ------------------------------------------------------------------------

    arquivos_fonte = {
        "treino_normal_A06":
            arquivo_treino,

        "validacao_normal_A06":
            arquivo_validacao,

        "teste_normal_base_A06":
            arquivo_teste,

        "eventos":
            ARQUIVO_EVENTOS,

        "ordem_features_ataque":
            ARQUIVO_ORDEM,

        "parametros_fisicos":
            ARQUIVO_PARAMETROS,

        "particao_descoberta_A06":
            ARQUIVO_GRUPOS_A06,

        "rnn_hai_usada_como_referencia":
            ARQUIVO_RNN_BASE,
    }

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
                "RNN_A06_agregacao_global_vs_local",

            "natureza":
                "exploratorio_controlado",

            "objetivo":
                (
                    "Verificar se a vantagem da agregacao local "
                    "sobrevive ao espaco de residuos produzido por "
                    "uma unica RNN-AE monolitica."
                ),

            "cenario":
                "A06",

            "alpha_grupo":
                alpha,

            "beta_global":
                beta,
        },

        "dados": {
            "n_features":
                400,

            "normalizacao":
                (
                    "MinMax fisico conhecido para features continuas; "
                    "features binarias permanecem 0/1."
                ),

            "treino":
                arquivo_treino,

            "validacao":
                arquivo_validacao,

            "teste_normal_base":
                arquivo_teste,

            "particao":
                (
                    "20 grupos de 20 features descobertos exclusivamente "
                    "a partir do treino normal de A06."
                ),
        },

        "rnn": {
            "tipo":
                "RNN Autoencoder",

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
                    "Global e local recebem exatamente os mesmos residuos "
                    "produzidos pela mesma RNN e pelo mesmo checkpoint."
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
                "residuos normais do treino por grupo",

            "grupos":
                20,

            "dimensao_grupo":
                20,

            "fusao":
                "maximo dos 20 scores locais",

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
                    "somente no teste, depois do treinamento, "
                    "ajuste das covariancias e calibracao dos limiares"
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

            "criterio_interpretacao":
                (
                    "vantagem local existe quando Recall_Local > Recall_Global "
                    "sob limiares independentes calibrados para FPR nominal "
                    "de 1% na mesma validacao normal"
                ),
        },

        "restricoes": {
            "uma_unica_rnn":
                True,

            "nenhuma_rnn_local_nesta_etapa":
                True,

            "nenhum_ajuste_de_arquitetura_apos_ver_o_teste":
                True,

            "nenhum_ataque_usado_no_treino":
                True,

            "nenhum_ataque_usado_na_validacao":
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
    print("PROTOCOLO RNN A06 REGISTRADO")
    print()

    print(
        f"Cenário: A06 | alpha={alpha:.2f} | beta={beta:.2f}"
    )

    print(
        "Partição: 20 grupos x 20 features"
    )

    print(
        "RNN: 400 -> hidden 64 -> 400 | janela=60 | batch=512"
    )

    print(
        "Treino: MSE da janela completa | Adam 1e-3"
    )

    print(
        "Early stopping: patience=5 | máximo=100 épocas"
    )

    print(
        "Scores: Mahalanobis global versus Mahalanobis local"
    )

    print(
        "Limiar: P99 da validação normal"
    )

    print(
        "Probes: m=8/delta=1.25 e m=8/delta=1.50"
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