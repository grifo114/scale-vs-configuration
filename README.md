# Scale vs. Configuration

Investigação experimental da decomposição do erro métrico de modelos
visão-linguagem em duas componentes:

- escala global compartilhada pela cena;
- configuração geométrica residual entre os objetos.

## Estado atual

Fase 1A: piloto técnico G-oracle.

O checkpoint v0.1 contém:

- 5 capturas do CA-1M;
- 12 relações por captura;
- 60 consultas válidas;
- 1 modelo avaliado;
- validação cruzada entre folds A e B dentro de cada captura.

Resultado primário:

- média de G_cv por captura: 0.025484312;
- mediana de G_cv: 0.038486118;
- 3 de 5 capturas com ganho positivo;
- 1 de 5 capturas com G_cv maior que 0.5.

Esses resultados ainda nao representam a Fase 1 completa, planejada com
50 cenas e tres modelos.

## Reprodução da análise

A análise usa apenas a biblioteca padrão do Python:

```bash
python3 scripts/analyze_oracle.py
```

Saidas:

```text
checkpoints/v0.1/metrics_recomputed.json
checkpoints/v0.1/scene_metrics.csv
```

## Estrutura

```text
scripts/             scripts reproduzíveis
configs/             configurações experimentais
docs/                documentação do projeto
checkpoints/v0.1/    entradas e resultados numéricos congelados
```

## Arquivos não versionados

Não são armazenados neste repositório:

- arquivos TAR e imagens do CA-1M;
- estímulos visuais derivados;
- pesos e cache dos modelos;
- ambiente virtual;
- repositório externo apple/ml-cubifyanything;
- saídas intermediárias em outputs/.
