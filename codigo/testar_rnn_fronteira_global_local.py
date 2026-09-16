import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from gerar_normal_estrutural import GANHO_TANH


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

CENARIOS_PERMITIDOS = [
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

if len(sys.argv) != 2:

    raise SystemExit(
        "Uso: python codigo/testar_rnn_fronteira_global_local.py CENARIO\n"
        "Exemplo: python codigo/testar_rnn_fronteira_global_local.py A01"
    )

CENARIO = sys.argv[1].strip().upper()

if CENARIO not in CENARIOS_PERMITIDOS:

    raise SystemExit(
        f"Cenário inválido: {CENARIO}. "
        f"Permitidos: {', '.join(CENARIOS_PERMITIDOS)}"
    )


ARQUIVO_PROTOCOLO = os.path.join(
    PROJETO,
    "dados",
    "rnn_fronteira",
    "protocolo_rnn_fronteira.json",
)

ARQUIVO_HASHES_ARTEFATOS = os.path.join(
    PROJETO,
    "dados",
    "rnn_fronteira",
    "sha256_artefatos_treino.txt",
)

ARQUIVO_HASHES_DIAGNOSTICOS = os.path.join(
    PROJETO,
    "dados",
    "rnn_fronteira",
    "sha256_diagnosticos_normal_only.txt",
)

DIRETORIO_TREINO = os.path.join(
    PROJETO,
    "resultado",
    "rnn_fronteira",
    CENARIO,
    "treino_monolitico",
)

ARQUIVO_MODELO = os.path.join(
    DIRETORIO_TREINO,
    f"modelo_rnn_{CENARIO.lower()}_best.pth",
)

ARQUIVO_RESIDUOS_TREINO = os.path.join(
    DIRETORIO_TREINO,
    "residuos_treino.npy",
)

ARQUIVO_RESIDUOS_VALIDACAO = os.path.join(
    DIRETORIO_TREINO,
    "residuos_validacao.npy",
)

ARQUIVO_FEATURES = os.path.join(
    DIRETORIO_TREINO,
    "ordem_features.csv",
)

ARQUIVO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "descoberta_grupos",
    CENARIO,
    "mapa_grupos_descobertos.csv",
)

ARQUIVO_DIAGNOSTICO_NORMAL_ONLY = os.path.join(
    PROJETO,
    "resultado",
    "rnn_fronteira",
    CENARIO,
    "diagnostico_residuos_normal_only",
    "diagnostico_residuos_resumo.csv",
)

ARQUIVO_EVENTOS = os.path.join(
    PROJETO,
    "dados",
    "eventos_confirmatorios_base.csv",
)

ARQUIVO_ORDEM_ATAQUES = os.path.join(
    PROJETO,
    "dados",
    "ordem_features_confirmatorias.csv",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "rnn_fronteira",
    CENARIO,
    "teste_global_local",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_RESULTADOS = os.path.join(
    DIRETORIO_SAIDA,
    f"resultados_rnn_{CENARIO.lower()}_global_local.csv",
)

ARQUIVO_SCORES = os.path.join(
    DIRETORIO_SAIDA,
    f"scores_rnn_{CENARIO.lower()}_global_local.csv",
)


# ============================================================================
# PROTOCOLO CONGELADO
# ============================================================================

HASH_PROTOCOLO = (
    "82290d6945c00cfc0d06f39ce6d538a3593a13b9996afaab38328a776dc53b60"
)

# ============================================================================
# PARÂMETROS CONGELADOS
# ============================================================================

WINDOW_SIZE = 60
BATCH_SIZE = 512

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0

FPR_ALVO = 0.01

PROBES = [
    {
        "Probe": "P01",
        "m": 8,
        "delta": 1.25,
    },
    {
        "Probe": "P02",
        "m": 8,
        "delta": 1.50,
    },
]


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
# RNN
# ============================================================================

class AnomalyRNN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_size,
        num_layers=1,
        dropout=0.0,
    ):

        super().__init__()

        self.hidden_size = hidden_size

        self.rnn_encoder = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity="tanh",
        )

        self.rnn_decoder = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity="tanh",
        )

        self.fc_out = nn.Linear(
            hidden_size,
            input_dim,
        )

    def forward(self, x):

        _, hidden_e = self.rnn_encoder(x)

        seq_len = x.size(1)
        batch_size = x.size(0)

        decoder_input = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_size,
            device=x.device,
        )

        decoder_output, _ = self.rnn_decoder(
            decoder_input,
            hidden_e,
        )

        return self.fc_out(
            decoder_output
        )


# ============================================================================
# NORMALIZAÇÃO
# ============================================================================

def normalizar_fisicamente(
    df,
    features,
    parametros,
):

    X = df[
        features
    ].copy()

    for feature in features:

        if feature not in parametros.index:
            continue

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

        X[
            feature
        ] = (
            X[
                feature
            ]
            -
            minimo
        ) / (
            maximo
            -
            minimo
        )

    X = X.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "NaN ou Inf após normalização."
        )

    return X


# ============================================================================
# JANELAMENTO
# ============================================================================

def criar_blocos(X):

    tensor = torch.from_numpy(X)

    windows = tensor.unfold(
        dimension=0,
        size=WINDOW_SIZE,
        step=1,
    ).permute(
        0,
        2,
        1,
    )

    return [
        windows[
            inicio:
            inicio + BATCH_SIZE
        ]
        for inicio in range(
            0,
            windows.shape[0],
            BATCH_SIZE,
        )
    ]


# ============================================================================
# RESÍDUOS DA RNN
# ============================================================================

def extrair_residuos(
    model,
    X,
    device,
):

    blocos = criar_blocos(
        X
    )

    partes = []

    model.eval()

    with torch.no_grad():

        for bloco in blocos:

            bloco = bloco.to(
                device
            )

            saida = model(
                bloco
            )

            residuo = (
                bloco[
                    :,
                    -1,
                    :
                ]
                -
                saida[
                    :,
                    -1,
                    :
                ]
            )

            partes.append(
                residuo
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

    return np.concatenate(
        partes,
        axis=0,
    )


# ============================================================================
# MÉTRICAS
# ============================================================================

def avaliar(
    scores,
    y_true,
    limiar,
):

    y_pred = (
        scores
        >
        limiar
    ).astype(
        np.int8
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp
        /
        (
            fp
            +
            tn
        )
    )

    return {
        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "FPR":
            float(fpr),

        "Precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y_true,
                    y_pred,
                )
            ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 108)
    print(f"RNN {CENARIO} — MESMA REDE, GLOBAL VERSUS LOCAL")
    print("=" * 108)

    # ------------------------------------------------------------------------
    # HASHES CONGELADOS
    # ------------------------------------------------------------------------

    def carregar_manifesto_hashes(caminho_manifesto):

        hashes = {}

        with open(
            caminho_manifesto,
            "r",
            encoding="utf-8",
        ) as arquivo:

            for linha in arquivo:

                linha = linha.strip()

                if not linha:
                    continue

                partes = linha.split(
                    maxsplit=1
                )

                if len(partes) != 2:

                    raise RuntimeError(
                        f"Linha inválida no manifesto de hashes: {linha}"
                    )

                hash_arquivo = partes[0].strip()

                caminho = partes[1].strip()

                if caminho.startswith("*"):
                    caminho = caminho[1:]

                hashes[
                    caminho
                ] = hash_arquivo

        return hashes


    hashes_artefatos = carregar_manifesto_hashes(
        ARQUIVO_HASHES_ARTEFATOS
    )

    hashes_diagnosticos = carregar_manifesto_hashes(
        ARQUIVO_HASHES_DIAGNOSTICOS
    )


    def hash_congelado(
        caminho,
        manifesto,
    ):

        caminho_relativo = os.path.relpath(
            caminho,
            PROJETO,
        )

        if caminho_relativo not in manifesto:

            raise RuntimeError(
                f"Arquivo não encontrado no manifesto congelado: "
                f"{caminho_relativo}"
            )

        return manifesto[
            caminho_relativo
        ]


    verificacoes = [
        (
            ARQUIVO_PROTOCOLO,
            HASH_PROTOCOLO,
            "protocolo",
        ),
        (
            ARQUIVO_MODELO,
            hash_congelado(
                ARQUIVO_MODELO,
                hashes_artefatos,
            ),
            "checkpoint RNN",
        ),
        (
            ARQUIVO_RESIDUOS_TREINO,
            hash_congelado(
                ARQUIVO_RESIDUOS_TREINO,
                hashes_artefatos,
            ),
            "residuos treino",
        ),
        (
            ARQUIVO_RESIDUOS_VALIDACAO,
            hash_congelado(
                ARQUIVO_RESIDUOS_VALIDACAO,
                hashes_artefatos,
            ),
            "residuos validacao",
        ),
        (
            ARQUIVO_FEATURES,
            hash_congelado(
                ARQUIVO_FEATURES,
                hashes_artefatos,
            ),
            "ordem features",
        ),
        (
            ARQUIVO_DIAGNOSTICO_NORMAL_ONLY,
            hash_congelado(
                ARQUIVO_DIAGNOSTICO_NORMAL_ONLY,
                hashes_diagnosticos,
            ),
            "diagnostico normal-only",
        ),
    ]

    print()
    print(
        f"Verificando artefatos congelados de {CENARIO}..."
    )

    for caminho, esperado, nome in verificacoes:

        observado = sha256_arquivo(
            caminho
        )

        print(
            f"{nome:24s}: "
            f"{'OK' if observado == esperado else 'ALTERADO'}"
        )

        if observado != esperado:

            raise RuntimeError(
                f"Hash alterado: {nome}"
            )
    
    # ------------------------------------------------------------------------
    # PROTOCOLO
    # ------------------------------------------------------------------------

    with open(
        ARQUIVO_PROTOCOLO,
        "r",
        encoding="utf-8",
    ) as arquivo:

        protocolo = json.load(
            arquivo
        )

    if CENARIO not in protocolo[
        "experimento"
    ][
        "cenarios_prospectivos"
    ]:

        raise RuntimeError(
            f"{CENARIO} não pertence aos cenários prospectivos congelados."
        )

    dados_cenario = protocolo[
        "dados"
    ][
        "cenarios"
    ][
        CENARIO
    ]

    arquivo_teste = dados_cenario[
        "teste_normal_base"
    ]

    if (
        os.path.abspath(
            dados_cenario["particao"]
        )
        !=
        os.path.abspath(
            ARQUIVO_GRUPOS
        )
    ):

        raise RuntimeError(
            f"{CENARIO}: partição diferente da registrada no protocolo."
        )

    # ------------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------------

    ordem_features = pd.read_csv(
        ARQUIVO_FEATURES
    ).sort_values(
        "Posicao"
    )

    features = ordem_features[
        "Feature"
    ].tolist()

    if len(features) != 400:

        raise RuntimeError(
            "Esperadas 400 features."
        )

    indice_feature = {
        feature: indice
        for indice, feature
        in enumerate(features)
    }

    # ------------------------------------------------------------------------
    # GRUPOS DESCOBERTOS
    # ------------------------------------------------------------------------

    grupos_df = pd.read_csv(
        ARQUIVO_GRUPOS
    )

    grupos = sorted(
        grupos_df[
            "Grupo_Descoberto"
        ]
        .unique()
        .tolist()
    )

    if len(grupos) < 1:

        raise RuntimeError(
            "Nenhum grupo descoberto."
        )

    indices_por_grupo = {}

    for grupo in grupos:

        features_grupo = grupos_df.loc[
            grupos_df[
                "Grupo_Descoberto"
            ]
            ==
            grupo,
            "Feature",
        ].tolist()


        indices_por_grupo[
            grupo
        ] = np.asarray(
            [
                indice_feature[
                    feature
                ]
                for feature in features_grupo
            ],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------------
    # RESÍDUOS NORMAIS CONGELADOS
    # ------------------------------------------------------------------------

    residuos_train = np.load(
        ARQUIVO_RESIDUOS_TREINO
    ).astype(
        np.float64,
        copy=False,
    )

    residuos_val = np.load(
        ARQUIVO_RESIDUOS_VALIDACAO
    ).astype(
        np.float64,
        copy=False,
    )

    # ========================================================================
    # MAHALANOBIS GLOBAL
    # ========================================================================

    print()
    print("Ajustando Mahalanobis global...")

    modelo_global = LedoitWolf(
        store_precision=True,
        assume_centered=False,
    )

    modelo_global.fit(
        residuos_train
    )

    score_global_val = modelo_global.mahalanobis(
        residuos_val
    )

    tau_global = float(
        np.quantile(
            score_global_val,
            1.0 - FPR_ALVO,
        )
    )

    # ========================================================================
    # MAHALANOBIS LOCAL
    # ========================================================================

    print("Ajustando Mahalanobis local...")

    modelos_locais = {}

    scores_locais_val = []

    for grupo in grupos:

        idx = indices_por_grupo[
            grupo
        ]

        modelo_local = LedoitWolf(
            store_precision=True,
            assume_centered=False,
        )

        modelo_local.fit(
            residuos_train[
                :,
                idx
            ]
        )

        modelos_locais[
            grupo
        ] = modelo_local

        scores_locais_val.append(
            modelo_local.mahalanobis(
                residuos_val[
                    :,
                    idx
                ]
            )
        )

    score_local_val = np.max(
        np.column_stack(
            scores_locais_val
        ),
        axis=1,
    )

    tau_local = float(
        np.quantile(
            score_local_val,
            1.0 - FPR_ALVO,
        )
    )

    print()
    print(
        f"Tau global P99: {tau_global:.6f}"
    )

    print(
        f"Tau local P99:  {tau_local:.6f}"
    )

    print(
        f"Alarmes global validação: "
        f"{np.sum(score_global_val > tau_global)} / {len(score_global_val)}"
    )

    print(
        f"Alarmes local validação:  "
        f"{np.sum(score_local_val > tau_local)} / {len(score_local_val)}"
    )

    # ------------------------------------------------------------------------
    # TESTE NORMAL BASE
    # ------------------------------------------------------------------------

    teste = pd.read_csv(
        arquivo_teste
    )

    if not (
        teste[
            "Attack"
        ]
        ==
        0
    ).all():

        raise RuntimeError(
            "Teste-base contém ataque antes da injeção."
        )

    parametros = pd.read_csv(
        ARQUIVO_PARAMETROS
    ).set_index(
        "Feature"
    )

    X_test_base = normalizar_fisicamente(
        teste,
        features,
        parametros,
    )

    # ------------------------------------------------------------------------
    # EVENTOS E ORDEM CONGELADOS
    # ------------------------------------------------------------------------

    eventos = pd.read_csv(
        ARQUIVO_EVENTOS
    )

    ordem_ataques = pd.read_csv(
        ARQUIVO_ORDEM_ATAQUES
    )

    ordem_por_grupo = {}

    for grupo, parte in ordem_ataques.groupby(
        "Grupo"
    ):

        parte = parte.sort_values(
            "Posicao"
        )

        ordem_por_grupo[
            str(grupo)
        ] = parte[
            "Feature"
        ].tolist()

    resultados = []
    scores_saida = []

    # ========================================================================
    # PROBES
    # ========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print()
    print(
        f"Dispositivo RNN: {device}"
    )

    model = AnomalyRNN(
        input_dim=400,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(
        device
    )

    model.load_state_dict(
        torch.load(
            ARQUIVO_MODELO,
            map_location=device,
        )
    )

    model.eval()

    for probe in PROBES:

        nome_probe = str(
            probe[
                "Probe"
            ]
        )

        m = int(
            probe[
                "m"
            ]
        )

        delta = float(
            probe[
                "delta"
            ]
        )

        print()
        print("=" * 108)
        print(
            f"{nome_probe} | m={m} | delta={delta:.2f}"
        )
        print("=" * 108)

        X_test = X_test_base.copy()

        y_original = np.zeros(
            len(X_test),
            dtype=np.int8,
        )

        # --------------------------------------------------------------------
        # INJEÇÃO DOS MESMOS 20 EVENTOS
        # --------------------------------------------------------------------

        for _, evento in eventos.iterrows():

            grupo_verdadeiro = str(
                evento[
                    "Grupo_Atacado"
                ]
            )

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

            direcao = int(
                evento[
                    "Direcao"
                ]
            )

            y_original[
                inicio:
                fim + 1
            ] = 1

            features_atacadas = ordem_por_grupo[
                grupo_verdadeiro
            ][
                :m
            ]

            idx = np.asarray(
                [
                    indice_feature[
                        feature
                    ]
                    for feature in features_atacadas
                ],
                dtype=np.int64,
            )

            linhas = np.arange(
                inicio,
                fim + 1,
            )

            valores_base = X_test_base[
                np.ix_(
                    linhas,
                    idx,
                )
            ]

            eps = 1e-12

            valores_seguros = np.clip(
                valores_base,
                eps,
                1.0 - eps,
            )

            latente = (
                np.arctanh(
                    2.0
                    *
                    valores_seguros
                    -
                    1.0
                )
                /
                GANHO_TANH
            )

            valores_atacados = (
                np.tanh(
                    GANHO_TANH
                    *
                    (
                        latente
                        +
                        direcao
                        *
                        delta
                    )
                )
                +
                1.0
            ) / 2.0

            X_test[
                np.ix_(
                    linhas,
                    idx,
                )
            ] = valores_atacados

        if int(
            y_original.sum()
        ) != 2000:

            raise RuntimeError(
                f"Esperados 2000 pontos atacados, obtidos {y_original.sum()}."
            )

        # --------------------------------------------------------------------
        # RNN: UMA ÚNICA RECONSTRUÇÃO
        # --------------------------------------------------------------------

        residuos_test = extrair_residuos(
            model,
            X_test,
            device,
        )

        esperado = (
            len(X_test)
            -
            WINDOW_SIZE
            +
            1
        )

        if residuos_test.shape != (
            esperado,
            400,
        ):

            raise RuntimeError(
                f"Shape residual inesperado: {residuos_test.shape}"
            )

        # --------------------------------------------------------------------
        # ALINHAMENTO DO RÓTULO AO ÚLTIMO INSTANTE DA JANELA
        # --------------------------------------------------------------------

        y_test = y_original[
            WINDOW_SIZE - 1:
        ]

        if len(y_test) != len(
            residuos_test
        ):

            raise RuntimeError(
                "Desalinhamento entre resíduos e rótulos."
            )

        if int(
            y_test.sum()
        ) != 2000:

            raise RuntimeError(
                f"Após alinhamento deveriam restar 2000 ataques, "
                f"obtidos {y_test.sum()}."
            )

        # --------------------------------------------------------------------
        # SCORE GLOBAL
        # --------------------------------------------------------------------

        score_global_test = modelo_global.mahalanobis(
            residuos_test
        )

        # --------------------------------------------------------------------
        # SCORE LOCAL DOS EXATAMENTE MESMOS RESÍDUOS
        # --------------------------------------------------------------------

        scores_locais_test = []

        for grupo in grupos:

            idx = indices_por_grupo[
                grupo
            ]

            scores_locais_test.append(
                modelos_locais[
                    grupo
                ].mahalanobis(
                    residuos_test[
                        :,
                        idx
                    ]
                )
            )

        score_local_test = np.max(
            np.column_stack(
                scores_locais_test
            ),
            axis=1,
        )

        # --------------------------------------------------------------------
        # MÉTRICAS
        # --------------------------------------------------------------------

        resultado_global = avaliar(
            score_global_test,
            y_test,
            tau_global,
        )

        resultado_local = avaliar(
            score_local_test,
            y_test,
            tau_local,
        )

        delta_recall = (
            resultado_local[
                "Recall"
            ]
            -
            resultado_global[
                "Recall"
            ]
        )

        print()
        print(
            f"GLOBAL | "
            f"Recall={resultado_global['Recall']:.6f} | "
            f"FPR={100.0 * resultado_global['FPR']:.4f}% | "
            f"F1={resultado_global['F1']:.6f} | "
            f"MCC={resultado_global['MCC']:.6f}"
        )

        print(
            f"LOCAL  | "
            f"Recall={resultado_local['Recall']:.6f} | "
            f"FPR={100.0 * resultado_local['FPR']:.4f}% | "
            f"F1={resultado_local['F1']:.6f} | "
            f"MCC={resultado_local['MCC']:.6f}"
        )

        print(
            f"DELTA RECALL = {delta_recall:+.6f}"
        )

        resultados.append(
            {
                "Probe":
                    nome_probe,

                "m":
                    m,

                "delta":
                    delta,

                "Recall_Global":
                    resultado_global[
                        "Recall"
                    ],

                "Recall_Local":
                    resultado_local[
                        "Recall"
                    ],

                "Delta_Recall":
                    delta_recall,

                "FPR_Global":
                    resultado_global[
                        "FPR"
                    ],

                "FPR_Local":
                    resultado_local[
                        "FPR"
                    ],

                "Precision_Global":
                    resultado_global[
                        "Precision"
                    ],

                "Precision_Local":
                    resultado_local[
                        "Precision"
                    ],

                "F1_Global":
                    resultado_global[
                        "F1"
                    ],

                "F1_Local":
                    resultado_local[
                        "F1"
                    ],

                "MCC_Global":
                    resultado_global[
                        "MCC"
                    ],

                "MCC_Local":
                    resultado_local[
                        "MCC"
                    ],

                "TP_Global":
                    resultado_global[
                        "TP"
                    ],

                "TP_Local":
                    resultado_local[
                        "TP"
                    ],

                "FP_Global":
                    resultado_global[
                        "FP"
                    ],

                "FP_Local":
                    resultado_local[
                        "FP"
                    ],
            }
        )

        for i in range(
            len(y_test)
        ):

            scores_saida.append(
                {
                    "Probe":
                        nome_probe,

                    "Indice_Original":
                        i
                        +
                        WINDOW_SIZE
                        -
                        1,

                    "Attack":
                        int(
                            y_test[
                                i
                            ]
                        ),

                    "Score_Global":
                        float(
                            score_global_test[
                                i
                            ]
                        ),

                    "Score_Local":
                        float(
                            score_local_test[
                                i
                            ]
                        ),

                    "Tau_Global":
                        tau_global,

                    "Tau_Local":
                        tau_local,
                }
            )

    # =========================================================================
    # SALVAMENTO
    # =========================================================================

    resultados_df = pd.DataFrame(
        resultados
    )

    resultados_df.to_csv(
        ARQUIVO_RESULTADOS,
        index=False,
    )

    pd.DataFrame(
        scores_saida
    ).to_csv(
        ARQUIVO_SCORES,
        index=False,
    )

    print()
    print("=" * 108)
    print(f"RESULTADO FINAL — RNN {CENARIO}")
    print("=" * 108)

    print()

    print(
        resultados_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Comparação usa uma única RNN e exatamente os mesmos resíduos."
    )

    print(
        "A única diferença é Mahalanobis global versus Mahalanobis por grupos + máximo."
    )

    print()
    print(
        f"Resultados: {ARQUIVO_RESULTADOS}"
    )

    print(
        f"Scores:     {ARQUIVO_SCORES}"
    )


if __name__ == "__main__":
    main()