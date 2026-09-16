# Detecção de Anomalias em Subespaços

Este repositório contém os códigos, protocolos, documentação e resultados associados aos experimentos sintéticos utilizados na investigação de detecção de anomalias por meio de avaliação global e avaliação em subespaços.

## Organização

- `codigo/`: programas utilizados na geração dos dados, auditorias, descoberta de grupos, experimentos estáticos, diagnóstico estrutural e experimentos com RNN;
- `dados/`: protocolos, configurações, manifestos e arquivos auxiliares necessários à reprodução;
- `resultado/`: resultados experimentais e arquivos de resumo mantidos no repositório;
- `Documentos/`: documentação do projeto, incluindo o Guia de Reprodutibilidade dos Experimentos;
- `envelope_sintetico.sh`: script de preparação do ambiente experimental.

## Artefatos experimentais completos

Os conjuntos de dados sintéticos completos, resíduos, checkpoints e demais artefatos pesados não são armazenados diretamente neste repositório devido ao seu tamanho.

Esses arquivos estão disponíveis no Google Drive:

https://drive.google.com/drive/folders/1tpnx00O56_zoKrukBQxkyotJsAr2DC47?usp=sharing

Os pacotes disponibilizados são:

- `dados_reprodutibilidade.tar.gz`
- `resultado.tar.gz`
- `artefatos_sha256.txt`

### SHA-256

```text
7f62b0d814b5448858a7389b4481f9dcfddee8dcd7d6439916d930cbc2c6dbab  dados_reprodutibilidade.tar.gz
6473c546615fdea77507318ac204e5b2caf6a694789a53b708eef26987c7429f  resultado.tar.gz