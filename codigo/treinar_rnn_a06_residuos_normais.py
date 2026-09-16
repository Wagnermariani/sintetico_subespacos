import gc
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# CAMINHOS
# ============================================================================

PROJETO = "/home/wagner/areas/sintetico_subespacos"

ARQUIVO_PROTOCOLO = os.path.join(
    PROJETO,
    "dados",
    "rnn_a06",
    "protocolo_rnn_a06.json",
)

ARQUIVO_HASH_PROTOCOLO = os.path.join(
    PROJETO,
    "dados",
    "rnn_a06",
    "sha256_protocolo_rnn_a06.txt",
)

ARQUIVO_PARAMETROS = os.path.join(
    PROJETO,
    "dados",
    "parametros_fisicos_sensores.csv",
)

ARQUIVO_GRUPOS = os.path.join(
    PROJETO,
    "resultado",
    "fronteira_estrutural",
    "descoberta_grupos",
    "A06",
    "mapa_grupos_descobertos.csv",
)

DIRETORIO_SAIDA = os.path.join(
    PROJETO,
    "resultado",
    "rnn_a06",
    "treino_monolitico",
)

os.makedirs(
    DIRETORIO_SAIDA,
    exist_ok=True,
)

ARQUIVO_MODELO = os.path.join(
    DIRETORIO_SAIDA,
    "modelo_rnn_a06_best.pth",
)

ARQUIVO_HISTORICO = os.path.join(
    DIRETORIO_SAIDA,
    "historico_treino.csv",
)

ARQUIVO_RESIDUOS_TREINO = os.path.join(
    DIRETORIO_SAIDA,
    "residuos_treino.npy",
)

ARQUIVO_RESIDUOS_VALIDACAO = os.path.join(
    DIRETORIO_SAIDA,
    "residuos_validacao.npy",
)

ARQUIVO_FEATURES = os.path.join(
    DIRETORIO_SAIDA,
    "ordem_features.csv",
)

ARQUIVO_ALINHAMENTO_TREINO = os.path.join(
    DIRETORIO_SAIDA,
    "alinhamento_treino.csv",
)

ARQUIVO_ALINHAMENTO_VALIDACAO = os.path.join(
    DIRETORIO_SAIDA,
    "alinhamento_validacao.csv",
)

ARQUIVO_RESUMO = os.path.join(
    DIRETORIO_SAIDA,
    "resumo_treino.json",
)


# ============================================================================
# PROTOCOLO CONGELADO
# ============================================================================

HASH_PROTOCOLO_ESPERADO = (
    "b5211740d1ac54ecd37a37fdbde781f03dca9e25dcd42b100b624e59128d2aff"
)

WINDOW_SIZE = 60
BATCH_SIZE = 512

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0

LEARNING_RATE = 1e-3

MAX_EPOCHS = 100
PATIENCE = 5

SEED = 42


# ============================================================================
# HASH
# ============================================================================

def sha256_arquivo(caminho):

    sha = hashlib.sha256()

    with open(
        caminho,
        "rb",
    ) as arquivo:

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
# MODELO
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
        self.num_layers = num_layers

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

    def forward(
        self,
        x,
    ):

        _, hidden_e = self.rnn_encoder(
            x
        )

        seq_len = x.size(
            1
        )

        batch_size = x.size(
            0
        )

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

        x_reconstructed = self.fc_out(
            decoder_output
        )

        return x_reconstructed


# ============================================================================
# CARREGAMENTO E NORMALIZAÇÃO
# ============================================================================

def carregar_normal(
    caminho,
    features,
    parametros,
):

    df = pd.read_csv(
        caminho
    )

    obrigatorias = [
        "timestamp",
        "Attack",
        "Tipo",
    ]

    faltantes = [
        coluna
        for coluna in obrigatorias + features
        if coluna not in df.columns
    ]

    if faltantes:

        raise RuntimeError(
            f"{caminho}: colunas faltantes: {faltantes[:10]}"
        )

    if not (
        df["Attack"] == 0
    ).all():

        raise RuntimeError(
            f"{caminho}: contém Attack != 0."
        )

    if not (
        df["Tipo"] == "Normal"
    ).all():

        raise RuntimeError(
            f"{caminho}: contém Tipo diferente de Normal."
        )

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

    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            f"{caminho}: NaN ou Inf após normalização."
        )

    minimo_observado = float(
        X.min()
    )

    maximo_observado = float(
        X.max()
    )

    if (
        minimo_observado < -1e-6
        or
        maximo_observado > 1.0 + 1e-6
    ):

        raise RuntimeError(
            f"{caminho}: valores fora de [0,1]: "
            f"min={minimo_observado}, max={maximo_observado}"
        )

    print(
        f"[LOAD] {os.path.basename(caminho)} | "
        f"linhas={len(df)} | "
        f"features={len(features)} | "
        f"range=[{minimo_observado:.6f}, {maximo_observado:.6f}]"
    )

    return (
        X,
        df[
            [
                "timestamp",
                "Attack",
                "Tipo",
            ]
        ].copy(),
    )


# ============================================================================
# JANELAMENTO
# ============================================================================

def criar_blocos(
    X,
):

    if len(
        X
    ) < WINDOW_SIZE:

        raise RuntimeError(
            "Conjunto menor que WINDOW_SIZE."
        )

    tensor = torch.from_numpy(
        X
    )

    windows = tensor.unfold(
        dimension=0,
        size=WINDOW_SIZE,
        step=1,
    ).permute(
        0,
        2,
        1,
    )

    blocos = [
        windows[
            inicio:
            inicio + BATCH_SIZE
        ]
        for inicio
        in range(
            0,
            windows.shape[0],
            BATCH_SIZE,
        )
    ]

    esperado = (
        len(X)
        -
        WINDOW_SIZE
        +
        1
    )

    obtido = sum(
        bloco.shape[0]
        for bloco in blocos
    )

    if obtido != esperado:

        raise RuntimeError(
            f"Janelamento inconsistente: esperado={esperado}, obtido={obtido}"
        )

    return blocos


# ============================================================================
# PERDA EM UM CONJUNTO
# ============================================================================

def calcular_perda(
    model,
    blocos,
    device,
    loss_fn,
):

    model.eval()

    soma = 0.0
    amostras = 0

    with torch.no_grad():

        for bloco in blocos:

            bloco = bloco.to(
                device
            )

            saida = model(
                bloco
            )

            loss = loss_fn(
                saida,
                bloco,
            )

            n = bloco.size(
                0
            )

            soma += (
                loss.item()
                *
                n
            )

            amostras += n

    if amostras == 0:

        raise RuntimeError(
            "Nenhuma janela disponível."
        )

    return (
        soma
        /
        amostras
    )


# ============================================================================
# EXTRAÇÃO DOS RESÍDUOS
# ============================================================================

def extrair_residuos(
    model,
    blocos,
    device,
):

    model.eval()

    partes = []

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
                    np.float32,
                    copy=False,
                )
            )

    residuos = np.concatenate(
        partes,
        axis=0,
    )

    return residuos


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 104)
    print("TREINO DA RNN MONOLÍTICA — A06")
    print("=" * 104)

    # ------------------------------------------------------------------------
    # PROTOCOLO
    # ------------------------------------------------------------------------

    hash_observado = sha256_arquivo(
        ARQUIVO_PROTOCOLO
    )

    print()
    print(
        f"SHA256 protocolo: {hash_observado}"
    )

    if (
        hash_observado
        !=
        HASH_PROTOCOLO_ESPERADO
    ):

        raise RuntimeError(
            "O protocolo RNN A06 foi alterado."
        )

    with open(
        ARQUIVO_HASH_PROTOCOLO,
        "r",
        encoding="utf-8",
    ) as arquivo:

        hash_registrado = (
            arquivo
            .read()
            .strip()
        )

    if (
        hash_registrado
        !=
        HASH_PROTOCOLO_ESPERADO
    ):

        raise RuntimeError(
            "Arquivo de hash do protocolo não confere."
        )

    with open(
        ARQUIVO_PROTOCOLO,
        "r",
        encoding="utf-8",
    ) as arquivo:

        protocolo = json.load(
            arquivo
        )

    arquivo_treino = protocolo[
        "dados"
    ][
        "treino"
    ]

    arquivo_validacao = protocolo[
        "dados"
    ][
        "validacao"
    ]

    # ------------------------------------------------------------------------
    # VERIFICAÇÃO DOS ARQUIVOS USADOS
    # ------------------------------------------------------------------------

    hashes_fontes = protocolo[
        "hashes_fontes"
    ]

    verificacoes = [
        (
            "treino_normal_A06",
            arquivo_treino,
        ),
        (
            "validacao_normal_A06",
            arquivo_validacao,
        ),
        (
            "parametros_fisicos",
            ARQUIVO_PARAMETROS,
        ),
        (
            "particao_descoberta_A06",
            ARQUIVO_GRUPOS,
        ),
    ]

    print()
    print(
        "Verificando arquivos congelados..."
    )

    for nome, caminho in verificacoes:

        observado = sha256_arquivo(
            caminho
        )

        esperado = hashes_fontes[
            nome
        ]

        print(
            f"{nome:28s}: "
            f"{'OK' if observado == esperado else 'ALTERADO'}"
        )

        if observado != esperado:

            raise RuntimeError(
                f"Hash alterado: {nome}"
            )

    # ------------------------------------------------------------------------
    # REPRODUTIBILIDADE
    # ------------------------------------------------------------------------

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

    # ------------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------------

    grupos_df = pd.read_csv(
        ARQUIVO_GRUPOS
    )

    features = sorted(
        grupos_df[
            "Feature"
        ].tolist()
    )

    if len(
        features
    ) != 400:

        raise RuntimeError(
            "Esperadas exatamente 400 features."
        )

    pd.DataFrame(
        {
            "Posicao":
                np.arange(
                    400
                ),

            "Feature":
                features,
        }
    ).to_csv(
        ARQUIVO_FEATURES,
        index=False,
    )

    # ------------------------------------------------------------------------
    # PARÂMETROS FÍSICOS
    # ------------------------------------------------------------------------

    parametros = (
        pd.read_csv(
            ARQUIVO_PARAMETROS
        )
        .set_index(
            "Feature"
        )
    )

    # ------------------------------------------------------------------------
    # CARREGAMENTO
    # ------------------------------------------------------------------------

    print()
    print(
        "Carregando treino e validação normais..."
    )

    X_train, meta_train = carregar_normal(
        arquivo_treino,
        features,
        parametros,
    )

    X_val, meta_val = carregar_normal(
        arquivo_validacao,
        features,
        parametros,
    )

    # ------------------------------------------------------------------------
    # JANELAMENTO
    # ------------------------------------------------------------------------

    print()
    print(
        "Criando janelas..."
    )

    blocos_train = criar_blocos(
        X_train
    )

    blocos_val = criar_blocos(
        X_val
    )

    n_train = (
        len(X_train)
        -
        WINDOW_SIZE
        +
        1
    )

    n_val = (
        len(X_val)
        -
        WINDOW_SIZE
        +
        1
    )

    print(
        f"Janelas treino:     {n_train}"
    )

    print(
        f"Janelas validação:  {n_val}"
    )

    # ------------------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print()
    print(
        f"Dispositivo: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------------

    model = AnomalyRNN(
        input_dim=400,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(
        device
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------------
    # TREINO
    # ------------------------------------------------------------------------

    historico = []

    melhor_val = float(
        "inf"
    )

    melhor_epoca = 0

    sem_melhora = 0

    print()
    print("=" * 104)
    print("INÍCIO DO TREINAMENTO")
    print("=" * 104)

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        soma_train = 0.0
        amostras_train = 0

        for bloco in blocos_train:

            bloco = bloco.to(
                device
            )

            optimizer.zero_grad()

            saida = model(
                bloco
            )

            loss = loss_fn(
                saida,
                bloco,
            )

            loss.backward()

            optimizer.step()

            n = bloco.size(
                0
            )

            soma_train += (
                loss.item()
                *
                n
            )

            amostras_train += n

        perda_train = (
            soma_train
            /
            amostras_train
        )

        perda_val = calcular_perda(
            model,
            blocos_val,
            device,
            loss_fn,
        )

        melhorou = (
            perda_val
            <
            melhor_val
        )

        if melhorou:

            melhor_val = (
                perda_val
            )

            melhor_epoca = (
                epoch
            )

            sem_melhora = 0

            torch.save(
                model.state_dict(),
                ARQUIVO_MODELO,
            )

        else:

            sem_melhora += 1

        historico.append(
            {
                "Epoca":
                    epoch,

                "Loss_Treino":
                    perda_train,

                "Loss_Validacao":
                    perda_val,

                "Melhorou":
                    int(
                        melhorou
                    ),
            }
        )

        print(
            f"Época {epoch:3d} | "
            f"train={perda_train:.8f} | "
            f"val={perda_val:.8f} | "
            f"{'BEST' if melhorou else f'sem melhora={sem_melhora}/{PATIENCE}'}"
        )

        pd.DataFrame(
            historico
        ).to_csv(
            ARQUIVO_HISTORICO,
            index=False,
        )

        if (
            sem_melhora
            >=
            PATIENCE
        ):

            print()
            print(
                f"Early stopping acionado na época {epoch}."
            )

            break

    if not os.path.exists(
        ARQUIVO_MODELO
    ):

        raise RuntimeError(
            "Nenhum checkpoint foi salvo."
        )

    # ------------------------------------------------------------------------
    # RECARREGA O MELHOR CHECKPOINT
    # ------------------------------------------------------------------------

    model.load_state_dict(
        torch.load(
            ARQUIVO_MODELO,
            map_location=device,
        )
    )

    model.eval()

    print()
    print(
        f"Melhor época: {melhor_epoca}"
    )

    print(
        f"Melhor loss validação: {melhor_val:.10f}"
    )

    # ------------------------------------------------------------------------
    # RESÍDUOS NORMAIS
    # ------------------------------------------------------------------------

    print()
    print(
        "Extraindo resíduos normais do melhor checkpoint..."
    )

    residuos_train = extrair_residuos(
        model,
        blocos_train,
        device,
    )

    residuos_val = extrair_residuos(
        model,
        blocos_val,
        device,
    )

    if residuos_train.shape != (
        n_train,
        400,
    ):

        raise RuntimeError(
            f"Shape inesperado treino: {residuos_train.shape}"
        )

    if residuos_val.shape != (
        n_val,
        400,
    ):

        raise RuntimeError(
            f"Shape inesperado validação: {residuos_val.shape}"
        )

    if not np.isfinite(
        residuos_train
    ).all():

        raise RuntimeError(
            "Resíduos de treino contêm NaN ou Inf."
        )

    if not np.isfinite(
        residuos_val
    ).all():

        raise RuntimeError(
            "Resíduos de validação contêm NaN ou Inf."
        )

    np.save(
        ARQUIVO_RESIDUOS_TREINO,
        residuos_train,
    )

    np.save(
        ARQUIVO_RESIDUOS_VALIDACAO,
        residuos_val,
    )

    # ------------------------------------------------------------------------
    # ALINHAMENTO
    # ------------------------------------------------------------------------

    alinhamento_train = (
        meta_train
        .iloc[
            WINDOW_SIZE - 1:
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    alinhamento_val = (
        meta_val
        .iloc[
            WINDOW_SIZE - 1:
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    alinhamento_train.insert(
        0,
        "Indice_Original",
        np.arange(
            WINDOW_SIZE - 1,
            len(meta_train),
        ),
    )

    alinhamento_val.insert(
        0,
        "Indice_Original",
        np.arange(
            WINDOW_SIZE - 1,
            len(meta_val),
        ),
    )

    if len(
        alinhamento_train
    ) != len(
        residuos_train
    ):

        raise RuntimeError(
            "Desalinhamento treino."
        )

    if len(
        alinhamento_val
    ) != len(
        residuos_val
    ):

        raise RuntimeError(
            "Desalinhamento validação."
        )

    alinhamento_train.to_csv(
        ARQUIVO_ALINHAMENTO_TREINO,
        index=False,
    )

    alinhamento_val.to_csv(
        ARQUIVO_ALINHAMENTO_VALIDACAO,
        index=False,
    )

    # ------------------------------------------------------------------------
    # RESUMO
    # ------------------------------------------------------------------------

    resumo = {
        "cenario":
            "A06",

        "protocolo_sha256":
            HASH_PROTOCOLO_ESPERADO,

        "device":
            str(
                device
            ),

        "seed":
            SEED,

        "window_size":
            WINDOW_SIZE,

        "batch_size":
            BATCH_SIZE,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "learning_rate":
            LEARNING_RATE,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "melhor_epoca":
            melhor_epoca,

        "melhor_loss_validacao":
            melhor_val,

        "n_janelas_treino":
            int(
                n_train
            ),

        "n_janelas_validacao":
            int(
                n_val
            ),

        "shape_residuos_treino":
            list(
                residuos_train.shape
            ),

        "shape_residuos_validacao":
            list(
                residuos_val.shape
            ),

        "teste_carregado":
            False,

        "ataques_utilizados":
            False,
    }

    with open(
        ARQUIVO_RESUMO,
        "w",
        encoding="utf-8",
    ) as arquivo:

        json.dump(
            resumo,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 104)
    print("TREINAMENTO E EXTRAÇÃO CONCLUÍDOS")
    print("=" * 104)

    print()
    print(
        f"Resíduos treino:    {residuos_train.shape}"
    )

    print(
        f"Resíduos validação: {residuos_val.shape}"
    )

    print(
        f"Melhor época:       {melhor_epoca}"
    )

    print(
        f"Melhor val loss:    {melhor_val:.10f}"
    )

    print()
    print(
        "IMPORTANTE: nenhum conjunto de teste foi carregado "
        "e nenhum ataque foi utilizado."
    )

    print()
    print(
        f"Modelo:   {ARQUIVO_MODELO}"
    )

    print(
        f"Histórico:{ARQUIVO_HISTORICO}"
    )

    print(
        f"Resumo:   {ARQUIVO_RESUMO}"
    )

    # ------------------------------------------------------------------------
    # LIMPEZA
    # ------------------------------------------------------------------------

    del model
    del residuos_train
    del residuos_val
    del blocos_train
    del blocos_val

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()