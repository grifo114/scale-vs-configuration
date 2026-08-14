# Scale vs. Configuration

Investigação experimental da decomposição do erro métrico de modelos
visão-linguagem em duas componentes:

- escala global compartilhada pela cena;
- configuração geométrica residual entre os objetos.

## Estado atual

Fase 1A: piloto técnico G-oracle e consolidação do pipeline reproduzível.

O checkpoint v0.1 contém:

- 5 capturas do CA-1M;
- 12 relações por captura;
- 60 consultas válidas;
- 1 modelo avaliado;
- validação cruzada entre folds A e B dentro de cada captura.

Resultado primário:

- média de `G_cv` por captura: 0.025484312;
- mediana de `G_cv`: 0.038486118;
- 3 de 5 capturas com ganho positivo;
- 1 de 5 capturas com `G_cv` maior que 0.5.

Esses resultados ainda não representam a Fase 1 completa, planejada com
50 cenas e três modelos.

## Ambiente congelado

O ambiente usado no piloto foi registrado em:

```text
configs/pip-freeze-macos-arm64-python311.txt
```

Plataforma e versões principais:

- macOS arm64;
- Python 3.11.9;
- MLX 0.32.0;
- MLX-VLM 0.6.6;
- Transformers 5.15.0;
- modelo `mlx-community/Qwen3-VL-4B-Instruct-4bit`;
- revisão `2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b`.

O arquivo de ambiente registra 73 distribuições Python. Os pesos do modelo e
o ambiente virtual não fazem parte do repositório.

## Validação do checkpoint e do ambiente

O dry-run confere o hash do manifesto, a ordem determinística das consultas,
os hashes dos 60 estímulos, o snapshot local do modelo e as versões principais
do ambiente. Ele não carrega o modelo nem executa inferência.

```bash
python3 scripts/run_oracle_mlx.py --dry-run
```

## Smoke test do runner

Uma consulta foi executada de ponta a ponta com o runner versionado:

- relação: `45662921-P07`;
- ground truth: 0.4283366554832823 m;
- previsão: 0.52 m;
- respostas válidas: 1/1;
- tempo de geração observado: 4.37 s.

A previsão coincidiu com a registrada no piloto original para essa relação.
Esse teste valida uma consulta, não a reprodução integral das 60 consultas.

Comando usado:

```bash
python3 scripts/run_oracle_mlx.py \
  --limit 1 \
  --output-dir outputs/phase1/smoke_runner_v0_1
```

O runner exige que novas execuções sejam gravadas fora de
`checkpoints/v0.1`. O checkpoint permanece congelado.

## Reprodução completa das inferências

O comando abaixo está preparado para uma nova execução das 60 consultas. Essa
execução ainda não foi realizada com o runner versionado.

```bash
python3 scripts/run_oracle_mlx.py \
  --output-dir outputs/phase1/reproduction_qwen3vl4b_oracle
```

Uma execução interrompida pode ser retomada com o mesmo diretório:

```bash
python3 scripts/run_oracle_mlx.py \
  --output-dir outputs/phase1/reproduction_qwen3vl4b_oracle \
  --resume
```

## Reprodução da análise

A análise do checkpoint usa apenas a biblioteca padrão do Python:

```bash
python3 scripts/analyze_oracle.py
```

Saídas:

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
- saídas intermediárias em `outputs/`.
