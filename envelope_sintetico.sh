#!/bin/bash

# ========================================================================
# Configura o ambiente virtual "envelope_sintetico"
# Ambiente: /home/wagner/areas/envelope_sintetico
# Projeto:  /home/wagner/areas/sintetico_subespacos
# Requirements: /home/wagner/areas/sintetico_subespacos/codigo/requirements.txt
# ========================================================================

BASE_DIR="/home/wagner/areas"
PROJECT_DIR="/home/wagner/areas/sintetico_subespacos"
ENV_NAME="envelope_sintetico"
ENV_PATH="$BASE_DIR/$ENV_NAME"
REQUIREMENTS_FILE="$PROJECT_DIR/codigo/requirements.txt"

echo "Ambiente virtual:"
echo "$ENV_PATH"

# Cria somente se ainda não existir
if [ ! -d "$ENV_PATH" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv "$ENV_PATH"
else
    echo "Ambiente virtual já existe."
fi

echo "Ativando ambiente..."
source "$ENV_PATH/bin/activate"

# Configuração de memória PyTorch/CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "Atualizando pip..."
python -m pip install --upgrade pip

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Instalando dependências de:"
    echo "$REQUIREMENTS_FILE"
    python -m pip install -r "$REQUIREMENTS_FILE"
else
    echo "ERRO: requirements.txt não encontrado:"
    echo "$REQUIREMENTS_FILE"
    exit 1
fi

echo
echo "Ambiente configurado com sucesso."
echo
echo "Para ativar futuramente:"
echo "source $ENV_PATH/bin/activate"