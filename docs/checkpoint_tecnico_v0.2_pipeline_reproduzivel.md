# Checkpoint técnico v0.2 — pipeline reproduzível do piloto G-oracle

**Projeto:** *Escala ou configuração? Decomposição validada do erro métrico em VLMs e medição com escala revisável*<br>
**Checkpoint:** v0.2 — pipeline reproduzível<br>
**Data de consolidação:** 15 de agosto de 2026<br>
**Repositório:** [grifo114/scale-vs-configuration](https://github.com/grifo114/scale-vs-configuration)<br>
**Visibilidade registrada:** privado<br>
**Tag de referência:** `v0.2-reproducible-pipeline`<br>
**Commit de referência:** `617a8a16be4895aaec60e1e74a546b615d5a5a95`<br>
**Checkpoint científico anterior:** `v0.1-piloto-oracle`

---

## 1. Resumo executivo

O checkpoint v0.2 transforma o piloto G-oracle do v0.1 em uma execução computacional reproduzível no ambiente em que ele foi originalmente realizado.

O trabalho técnico concluído inclui:

- versionamento de um runner único para validar entradas e executar inferência com MLX;
- congelamento do ambiente Python usado no Mac Apple Silicon;
- verificação prévia de manifesto, estímulos, snapshot do modelo e versões críticas;
- smoke test de uma consulta;
- repetição integral das 60 consultas do piloto;
- comparação automatizada entre a execução original e a reprodução;
- teste positivo e teste negativo do comparador;
- integração em `main` por pull request;
- criação e publicação da tag anotada `v0.2-reproducible-pipeline`;
- remoção segura das branches temporárias após o merge.

Resultado central do checkpoint:

> A reprodução integral obteve 60 respostas válidas em 60 consultas e foi idêntica à execução de referência em todos os campos determinísticos comparados. A maior diferença absoluta entre as predições numéricas foi `0 m`.

Este resultado valida a reexecução do piloto no mesmo ambiente computacional e com o mesmo snapshot do modelo. Ele não amplia a amostra científica, não adiciona modelos e não altera os resultados numéricos do v0.1.

---

## 2. O que o v0.2 representa

O v0.2 é um checkpoint de engenharia experimental e reprodutibilidade.

Ele estabelece que:

1. as entradas congeladas do piloto podem ser validadas automaticamente;
2. o modelo pode ser carregado a partir do snapshot local exato;
3. a ordem das consultas é determinística;
4. as 60 inferências podem ser repetidas com sucesso;
5. as respostas reproduzidas coincidem com as respostas originais nos campos determinísticos;
6. a comparação entre execuções pode ser repetida por um script versionado;
7. o estado do código está identificado por commit e tag públicos no repositório privado do projeto.

### 2.1 O que permaneceu inalterado

O v0.2 reutiliza o mesmo desenho científico do checkpoint v0.1:

- 5 capturas do CA-1M;
- 12 relações por captura;
- 60 consultas no total;
- um único modelo avaliado;
- condição G-oracle, com os objetos A e B explicitamente marcados;
- validação cruzada entre folds A e B dentro de cada captura;
- erro primário no domínio logarítmico;
- ganho de correção de escala medido por `G_cv`.

Portanto, os valores de `G_cv`, RMSE-log, MAE, MAPE e fração descritiva de escala continuam sendo os resultados científicos do v0.1.

---

## 3. Estado do projeto neste checkpoint

O projeto está no encerramento técnico do piloto da Fase 1A e ainda não concluiu a Fase 1 planejada.

| Componente | Estado no v0.2 |
|---|---|
| Auditoria inicial do CA-1M | concluída para o piloto |
| Seleção das cinco capturas | concluída para o piloto |
| Geração dos 60 estímulos G-oracle | concluída |
| Inferência original | concluída, 60/60 válidas |
| Análise por cena | concluída |
| Checkpoint científico v0.1 | concluído e versionado |
| Runner de inferência | versionado |
| Ambiente Python | congelado |
| Reprodução integral | concluída, 60/60 válidas |
| Comparação automatizada | concluída |
| Integração no branch principal | concluída |
| Tag técnica v0.2 | publicada |
| Fase 1 com 50 cenas e 3 modelos | ainda não executada |

O próximo avanço científico começa depois deste checkpoint, com a formalização do pipeline de seleção e geração dos dados da Fase 1 e a expansão controlada da amostra.

---

## 4. Base científica herdada do v0.1

### 4.1 Pergunta do piloto

O piloto investiga se parte do erro métrico de um VLM pode ser explicada por um fator de escala coerente por cena, separado de erros de configuração relativa.

Na condição G-oracle, o problema de localizar os dois objetos é reduzido por caixas coloridas:

- objeto A: caixa vermelha;
- objeto B: caixa azul;
- valor real da distância: oculto do modelo.

A distância-alvo é a distância euclidiana em metros entre os centros das caixas 3D dos dois objetos.

### 4.2 Prompt congelado

```text
Object A is marked with a red box and object B with a blue box. Estimate the 3D Euclidean distance between the centers of their 3D bounding boxes, in meters. Return only JSON in this format: {"distance_m": number}
```

### 4.3 Capturas e relações

| Captura | Ambiente visual | Relações | GT mínimo | GT máximo |
|---|---|---:|---:|---:|
| `42897545` | banheiro | 12 | 0,266 m | 0,886 m |
| `45261179` | banheiro | 12 | 0,623 m | 1,900 m |
| `45261615` | sala/quarto | 12 | 0,355 m | 1,851 m |
| `45662921` | cozinha | 12 | 0,285 m | 1,107 m |
| `47115543` | escritório/sala de jantar | 12 | 0,354 m | 2,126 m |

Total: 5 capturas, 60 relações.

`[Não verificado]` As capturas possuem identificadores e ambientes visuais distintos, mas não foi estabelecido que correspondam a cinco imóveis fisicamente independentes. A unidade conservadora de análise continua sendo a captura.

### 4.4 Métrica primária

Para uma cena `s`, relação `k` e modelo `m`, o erro logarítmico é:

```text
e_skm = log(predição_skm) - log(GT_sk)
```

Um fator multiplicativo de escala é estimado em um fold e aplicado ao fold retido. O ganho de validação cruzada é:

```text
G_cv = 1 - SSE_corrigido / SSE_bruto
```

Interpretação:

- `G_cv > 0`: a correção de escala aprendida no outro fold reduziu o erro;
- `G_cv = 0`: a correção não alterou o erro quadrático total;
- `G_cv < 0`: a correção piorou a generalização;
- `G_cv > 0,5`: mais da metade do erro quadrático logarítmico foi removida.

---

## 5. Resultados científicos preservados

### 5.1 Resultado por captura

| Captura | RMSE-log bruto | RMSE-log corrigido | `G_cv` | Fração descritiva de escala | MAE | MAPE |
|---|---:|---:|---:|---:|---:|---:|
| `42897545` | 0,176 | 0,123 | 0,510 | 0,517 | 0,083 m | 13,0% |
| `45261179` | 0,367 | 0,359 | 0,038 | 0,165 | 0,349 m | 36,4% |
| `45261615` | 0,588 | 0,448 | 0,420 | 0,491 | 0,410 m | 69,3% |
| `45662921` | 0,322 | 0,424 | -0,733 | 0,201 | 0,130 m | 28,3% |
| `47115543` | 0,247 | 0,260 | -0,108 | 0,0004 | 0,242 m | 22,1% |

### 5.2 Agregado primário por captura

| Medida | Resultado |
|---|---:|
| Média de `G_cv` | 0,025484312 |
| Mediana de `G_cv` | 0,038486118 |
| Capturas com `G_cv > 0` | 3 de 5 |
| Capturas com `G_cv > 0,5` | 1 de 5 |
| RMSE-log agrupado bruto | 0,367 |
| RMSE-log agrupado corrigido | 0,344 |
| `G_cv` agrupado secundário | 0,124 |
| Média da fração descritiva de escala | 0,275 |
| Mediana da fração descritiva de escala | 0,201 |
| Média do MAE por captura | 0,243 m |
| Média do MAPE por captura | 33,8% |

Nenhum intervalo de confiança foi calculado no piloto porque a unidade agregada contém apenas cinco capturas.

### 5.3 Interpretação permitida pelos dados

Os resultados demonstram que uma correção multiplicativa aprendida em relações de uma cena pode generalizar para relações retidas em algumas capturas. Essa generalização, porém, é heterogênea:

- três capturas apresentaram ganho positivo;
- apenas uma ultrapassou `G_cv = 0,5`;
- duas apresentaram ganho negativo;
- a média por captura ficou próxima de zero.

`[Inferência]` No modelo 4B quantizado avaliado, a coerência de escala parece depender fortemente da cena.

`[Inferência]` O piloto sustenta a continuidade do diagnóstico estrutural, mas não sustenta uma alegação geral de que o erro métrico seja predominantemente um erro de escala.

### 5.4 Preferências numéricas observadas

Foram observados somente 18 valores preditos distintos nas 60 respostas. O valor `1,32 m` apareceu em 15 consultas, e os cinco valores mais frequentes responderam por 66,7% das saídas.

`[Inferência]` Essa concentração é compatível com níveis numéricos preferenciais do modelo. A causa não foi isolada e pode envolver o modelo, a quantização, o prompt ou a distribuição visual.

---

## 6. Dívida técnica identificada no v0.1

O checkpoint v0.1 congelou entradas, resultados, métricas, documentação e hashes, mas a execução de inferência havia sido construída em comandos e scripts transitórios usados no terminal.

Os principais riscos eram:

- não haver um runner único versionado;
- dependência de ativação correta do ambiente virtual;
- uso implícito do cache local do Hugging Face;
- ausência de validação automática dos 60 estímulos;
- ausência de checagem programática do manifesto e de sua hash;
- ausência de uma ferramenta versionada para comparar duas execuções;
- ausência de um registro completo das versões instaladas;
- possibilidade de repetir a análise, mas não a inferência integral por um caminho único documentado.

O v0.2 resolve a parte de inferência, ambiente e comparação. O pipeline anterior à inferência ainda possui dívida residual, detalhada na Seção 15.

---

## 7. Ambiente computacional congelado

### 7.1 Hardware e sistema

| Item | Valor |
|---|---|
| Equipamento | MacBook Air |
| Chip | Apple M3 |
| CPU | 8 núcleos, sendo 4 de desempenho e 4 de eficiência |
| Memória | 16 GB |
| Arquitetura | `arm64` |
| Python | 3.11.9 |

### 7.2 Pacotes críticos

| Pacote | Versão |
|---|---:|
| `mlx` | 0.32.0 |
| `mlx-metal` | 0.32.0 |
| `mlx-vlm` | 0.6.6 |
| `transformers` | 5.15.0 |
| `huggingface_hub` | 1.27.0 |
| `numpy` | 2.4.6 |
| `pillow` | 12.3.0 |

O arquivo `configs/pip-freeze-macos-arm64-python311.txt` preserva 73 pacotes. A auditoria não encontrou caminhos locais ou instalações editáveis no `pip freeze`.

### 7.3 Modelo

| Campo | Valor |
|---|---|
| Identificador | `mlx-community/Qwen3-VL-4B-Instruct-4bit` |
| Revisão | `2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b` |
| Origem usada na reprodução | snapshot local do cache Hugging Face |
| Temperatura | 0,0 |
| Máximo de tokens | 64 |
| Semente de geração | 0 |
| Semente da ordem | 20260814 |

O uso direto do snapshot local evita que a resolução de `main` no Hub selecione uma revisão posterior.

---

## 8. Artefatos técnicos adicionados no v0.2

### 8.1 Runner de inferência

Arquivo:

```text
scripts/run_oracle_mlx.py
```

Responsabilidades:

- localizar a raiz do projeto e o checkpoint v0.1;
- validar a hash do manifesto;
- carregar e validar as 60 linhas do manifesto;
- confirmar as cinco capturas e as 60 consultas;
- reconstruir a ordem determinística;
- localizar os 60 estímulos e detectar ausências;
- localizar o snapshot exato do modelo;
- verificar versões essenciais do ambiente;
- oferecer modo `--dry-run` sem carregar o modelo;
- permitir execução limitada por `--limit`;
- executar o lote completo;
- extrair o JSON retornado pelo modelo;
- registrar resultado por consulta;
- registrar metadados e status da execução;
- gravar saídas fora do Git em `outputs/`.

### 8.2 Comparador de execuções

Arquivo:

```text
scripts/compare_oracle_runs.py
```

Responsabilidades:

- carregar a execução original e uma execução candidata;
- validar quantidade e identidade das consultas;
- comparar ordem e entradas estáveis;
- comparar respostas brutas;
- comparar predições numéricas;
- comparar a parte determinística dos metadados;
- calcular a maior diferença absoluta entre predições;
- reportar diferenças por categoria;
- retornar código de saída diferente de zero quando houver divergência;
- escrever um relatório JSON opcional.

Tempos de execução, timestamps e caminhos locais não são tratados como campos determinísticos equivalentes.

### 8.3 Snapshot do ambiente

Arquivo:

```text
configs/pip-freeze-macos-arm64-python311.txt
```

Esse arquivo registra o ambiente efetivamente usado. Ele não substitui uma especificação mínima e multiplataforma de dependências; funciona como evidência do ambiente reproduzido.

---

## 9. Validação do runner

### 9.1 Compilação

O runner e o comparador foram validados com `py_compile` antes da execução.

```bash
python3 -m py_compile scripts/run_oracle_mlx.py
python3 -m py_compile scripts/compare_oracle_runs.py
```

### 9.2 Dry-run

Comando:

```bash
python3 scripts/run_oracle_mlx.py --dry-run
```

Resultado:

- manifesto reconhecido pela hash esperada;
- cinco capturas reconhecidas;
- 60 consultas reconhecidas;
- 60 estímulos verificados;
- zero estímulos ausentes;
- snapshot do modelo localizado;
- ambiente crítico identificado;
- modelo não carregado;
- inferência não executada.

Primeiras cinco consultas da ordem determinística:

1. `45662921-P07`;
2. `45261179-P09`;
3. `42897545-P11`;
4. `45261615-P01`;
5. `42897545-P06`.

### 9.3 Smoke test

Comando:

```bash
python3 scripts/run_oracle_mlx.py \
  --limit 1 \
  --output-dir outputs/phase1/smoke_runner_v0_1
```

Resultado:

| Campo | Valor |
|---|---|
| Par | `45662921-P07` |
| GT | 0,4283366554832823 m |
| Predição | 0,52 m |
| Tempo observado | 4,37 s |
| Respostas válidas | 1 de 1 |
| Status | `completed` |

A predição coincidiu com a execução original.

---

## 10. Reprodução integral das 60 consultas

Comando:

```bash
python3 scripts/run_oracle_mlx.py \
  --output-dir outputs/phase1/reproduction_qwen3vl4b_oracle
```

Resultado operacional:

| Medida | Resultado |
|---|---:|
| Consultas previstas | 60 |
| Consultas executadas | 60 |
| Respostas válidas | 60 |
| Respostas inválidas | 0 |
| Capturas cobertas | 5 |

Saídas locais:

```text
outputs/phase1/reproduction_qwen3vl4b_oracle/results.jsonl
outputs/phase1/reproduction_qwen3vl4b_oracle/metadata.json
```

Essas saídas permanecem fora do Git porque `outputs/` é ignorado. Os resultados originais congelados continuam em `checkpoints/v0.1/`.

---

## 11. Comparação da reprodução com o checkpoint

Comando:

```bash
python3 scripts/compare_oracle_runs.py \
  --report outputs/phase1/reproduction_qwen3vl4b_oracle/comparison_to_checkpoint.json
```

### 11.1 Resultado detalhado

| Verificação | Diferenças |
|---|---:|
| Linhas de referência | 60 |
| Linhas candidatas | 60 |
| Ordem das consultas | 0 |
| Entradas | 0 |
| Campos estáveis de execução | 0 |
| Respostas brutas | 0 |
| Saídas numéricas | 0 |
| Metadados determinísticos | 0 |
| Maior diferença absoluta de predição | 0 m |

Resultado emitido pelo comparador:

```text
IDENTICAL for every compared deterministic field.
```

### 11.2 Tempos observados

| Execução | Mediana por consulta |
|---|---:|
| Referência v0.1 | 4,650 s |
| Reprodução v0.2 | 4,236 s |

A diferença de tempo não contradiz a reprodução determinística, pois duração não é uma saída científica determinística e pode variar com estado térmico, carga do sistema e condições de execução.

### 11.3 Escopo exato da equivalência

A expressão “reprodução idêntica” neste documento significa:

- mesma quantidade de consultas;
- mesma ordem;
- mesmas entradas estáveis;
- mesmas respostas textuais brutas;
- mesmas predições numéricas;
- mesmos metadados definidos como determinísticos;
- diferença numérica máxima igual a zero.

Ela não significa que os arquivos completos sejam idênticos byte a byte. Timestamps, tempos por consulta, caminhos absolutos e outros campos dependentes da execução podem mudar.

---

## 12. Validação do comparador

### 12.1 Teste positivo

O comparador foi executado em uma condição sem diferenças e retornou sucesso.

### 12.2 Teste negativo

Uma cópia temporária recebeu uma alteração artificial de `+0,001 m` em uma predição. O comparador:

- detectou uma diferença numérica;
- retornou código de saída `1`;
- não classificou a execução alterada como idêntica.

Esse teste demonstra que a igualdade não foi produzida apenas por um caminho que sempre retorna sucesso.

---

## 13. Integridade e identificadores

### 13.1 Entradas congeladas do v0.1

| Arquivo | SHA-256 |
|---|---|
| `checkpoints/v0.1/manifest.jsonl` | `8042a906d9e86d579b165da63ce3973c3581bb997851991336146079034c4698` |
| `checkpoints/v0.1/metadata.json` | `59f885b6c62e619e7a815dbc07e8b185a98a52a3224df26ed002194608ee0c88` |
| `checkpoints/v0.1/protocol.json` | `43b88f65e4c305ecbc787421480e8c345fb3384a2c7092fd2a5a7fe540fa264d` |
| `checkpoints/v0.1/qwen3vl4b_oracle_metrics.json` | `5749cb27dc457f16777e384dc5b86fd25a291271ef15079b0b8627ed06fb0d50` |
| `checkpoints/v0.1/results.jsonl` | `088fa9d281f7520b4380f64c5f9ea0391078ce57fc89f7be8d256341de35c897` |

### 13.2 Artefatos técnicos do v0.2

| Arquivo | Identificador |
|---|---|
| `scripts/run_oracle_mlx.py` | SHA-256 `58512f1d921f67cc046f9b60863df5a1a91a4dbb1c09cc5b53b4f46a2d2c8b9f` |
| `scripts/compare_oracle_runs.py` | SHA-256 `33393fd1bb5f156c83cad2026f37a3aba69aa64ca257ab1e6e338984b3e71889` |
| `configs/pip-freeze-macos-arm64-python311.txt` | SHA-256 `36a0c58b5924793c12eaa70eebeb45cb245761ea498968cfd414aede06b585bb` |
| `README.md` no merge | blob Git `95596968bfa6c35f5640f9fd3d673061c37eb016` |

---

## 14. Histórico Git e GitHub

### 14.1 Checkpoint v0.1

| Item | Valor |
|---|---|
| Commit | `f4f76b48be2a5cdbcd459e9569451bf199458a63` |
| Tag | `v0.1-piloto-oracle` |
| Conteúdo | piloto, entradas, resultados, métricas, análise e documentação |

### 14.2 Desenvolvimento do v0.2

| Commit | Descrição |
|---|---|
| `a90848a6d4afd40bbc361e2920baaae5593b49f6` | runner, ambiente congelado e documentação de reprodução |
| `9ac69221fa020a23bd9663fb71fc56aaeb61f9c3` | comparador e validação da reprodução integral |

### 14.3 Pull request e merge

| Item | Valor |
|---|---|
| Pull request | [#1 — pipeline reproduzível do piloto G-oracle](https://github.com/grifo114/scale-vs-configuration/pull/1) |
| Estado | mesclado |
| Commit de merge | `617a8a16be4895aaec60e1e74a546b615d5a5a95` |
| Arquivos alterados | 4 |
| Adições | 1432 linhas |
| Remoções | 9 linhas |

### 14.4 Tag v0.2

| Item | Valor |
|---|---|
| Tag | `v0.2-reproducible-pipeline` |
| Tipo | anotada |
| Commit apontado | `617a8a16be4895aaec60e1e74a546b615d5a5a95` |
| Mensagem | `Pipeline reproduzível G-oracle: runner, ambiente e reprodução integral` |

### 14.5 Estado final do repositório

- branch local ativo: `main`;
- branch remoto padrão: `origin/main`;
- `main` local sincronizado com `origin/main`;
- árvore de trabalho limpa;
- branches temporárias removidas local e remotamente;
- tags v0.1 e v0.2 preservadas.

---

## 15. Limitações e dívida residual

### 15.1 Generalização científica

O piloto contém cinco capturas e um modelo. Não representa a Fase 1 completa planejada com 50 cenas e três modelos.

### 15.2 Independência estatística

As 12 relações de uma captura compartilham a mesma imagem e vários objetos. Elas não devem ser tratadas como 12 cenas independentes. O agregado primário permanece por captura.

### 15.3 Seleção exploratória

As capturas do piloto foram escolhidas após auditoria exploratória de quantidade de objetos, nitidez e inspeção visual. Essa escolha é adequada para testar a mecânica, mas exige separação explícita de uma futura amostra confirmatória.

### 15.4 Condição oracle

As caixas coloridas reduzem a ambiguidade de grounding. Isso é intencional para isolar a estimativa métrica, mas não mede o desempenho completo de um sistema que também precisa encontrar os objetos.

### 15.5 Portabilidade

A reprodução exata foi demonstrada no mesmo tipo de máquina, stack MLX e snapshot do modelo. Execução em CUDA, Linux, outra versão do MLX ou outra quantização ainda não foi validada.

### 15.6 Disponibilidade do modelo

O runner depende do snapshot local identificado pela revisão congelada. A persistência de longo prazo do repositório externo do modelo não está sob controle deste projeto.

### 15.7 Pipeline anterior à inferência

Ainda não estão materializados e versionados como pipeline único os scripts usados para:

- extrair quadros dos TARs do CA-1M;
- auditar todas as instâncias;
- calcular os filtros `oracle` e `natural`;
- selecionar quadros candidatos;
- gerar imagens anotadas;
- selecionar os pares estratificados;
- renderizar os estímulos vermelho/azul;
- construir o manifesto unificado.

Essa é a principal dívida técnica para o próximo ciclo. O v0.2 reproduz a inferência a partir dos estímulos congelados, mas ainda não reconstrói os estímulos a partir dos arquivos originais do CA-1M com um único comando.

### 15.8 Testes automatizados e integração contínua

O runner e o comparador foram testados manualmente, incluindo um teste negativo do comparador. Ainda não existe uma suíte de testes automatizada no repositório nem CI configurada para executar validações que não dependam do modelo.

### 15.9 Artefatos locais ignorados

Os diretórios `data/`, `outputs/`, `.cache/`, `.venv/` e `ml-cubifyanything/` permanecem fora do Git. Essa decisão evita versionar dados pesados, caches e dependências, mas exige preservação separada dos arquivos originais e dos estímulos.

### 15.10 Licença e redistribuição

O checkpoint v0.1 registrou o CA-1M como distribuído sob licença CC-BY-NC-ND. Por cautela, imagens, TARs e estímulos derivados não foram publicados no repositório. Qualquer pacote público de reprodução deve passar por revisão específica da licença e das regras de redistribuição.

---

## 16. Estado dos portões metodológicos

| Portão | Estado no v0.2 | Evidência |
|---|---|---|
| G0 — auditoria de novidade | aberto/incompleto | revisão sistemática ainda não encerrada |
| G1 — pelo menos 3 medidas por cena | aprovado no piloto | 12 relações por captura |
| G1b — escala generaliza dentro da cena | misto | 3/5 positivas; 1/5 acima de 0,5 |
| G2 — manipulação de focal | não iniciado | depende da fase sintética |
| G3 — efeito de âncora textual | não iniciado | experimento ainda não executado |
| G4 — método fatorado competitivo | não iniciado | modelo fatorado ainda não treinado |
| G5 — corrigibilidade superior | não iniciado | depende de G4 |
| G6 — calibração de incerteza | não iniciado | depende do método |
| G7 — auditoria final de contribuição | não iniciado | depende dos resultados completos |

O v0.2 não altera o estado científico de G1b. Sua contribuição é tornar a evidência do piloto tecnicamente reexecutável.

---

## 17. Procedimento de reprodução

### 17.1 Pré-condições

- macOS em Apple Silicon;
- Python 3.11;
- ambiente compatível com o snapshot congelado;
- 60 estímulos disponíveis nos caminhos esperados;
- snapshot local do modelo na revisão registrada;
- repositório no commit ou tag de referência.

### 17.2 Validação sem inferência

```bash
python3 scripts/run_oracle_mlx.py --dry-run
```

### 17.3 Smoke test

```bash
python3 scripts/run_oracle_mlx.py \
  --limit 1 \
  --output-dir outputs/phase1/smoke_runner_v0_1
```

### 17.4 Execução integral

```bash
python3 scripts/run_oracle_mlx.py \
  --output-dir outputs/phase1/reproduction_qwen3vl4b_oracle
```

### 17.5 Comparação com a referência

```bash
python3 scripts/compare_oracle_runs.py \
  --report outputs/phase1/reproduction_qwen3vl4b_oracle/comparison_to_checkpoint.json
```

### 17.6 Recomputação das métricas científicas

```bash
python3 scripts/analyze_oracle.py
```

Saídas recomputadas:

```text
checkpoints/v0.1/metrics_recomputed.json
checkpoints/v0.1/scene_metrics.csv
```

---

## 18. Critérios de retomada

Antes de iniciar uma nova branch científica, devem permanecer verdadeiros:

- `main` local sincronizado com `origin/main`;
- árvore de trabalho limpa;
- tag `v0.2-reproducible-pipeline` disponível;
- `python3 scripts/run_oracle_mlx.py --dry-run` aprovado;
- 60 estímulos acessíveis localmente;
- snapshot exato do modelo disponível;
- hashes do checkpoint v0.1 preservadas;
- documento v0.1 preservado sem substituição.

Ponto de retomada recomendado:

1. versionar o pipeline de preparação de dados anterior à inferência;
2. congelar uma política de seleção de cenas antes da expansão;
3. separar cenas exploratórias das cenas confirmatórias;
4. validar o pipeline em uma nova captura não usada no piloto;
5. só então escalar para 50 cenas e três modelos.

---

## 19. Próximo checkpoint sugerido

O próximo checkpoint técnico deve demonstrar reconstrução reproduzível dos estímulos a partir de um TAR original do CA-1M.

Entregáveis mínimos sugeridos:

- script de indexação de capturas;
- script de auditoria de quadros e instâncias;
- configuração versionada dos filtros;
- seleção determinística ou manifesto de decisões manuais;
- geração determinística dos pares;
- renderização dos estímulos;
- hashes das imagens produzidas;
- teste em uma captura nova;
- documentação da proveniência dos dados;
- teste automatizado sem dependência de inferência.

Nome de trabalho possível:

```text
v0.3-data-preparation-pipeline
```

Esse nome é apenas uma convenção sugerida; nenhuma branch ou tag com esse nome foi criada neste checkpoint.

---

## 20. Nota sobre a imutabilidade da tag v0.2

A tag `v0.2-reproducible-pipeline` já foi publicada e aponta para o merge `617a8a16be4895aaec60e1e74a546b615d5a5a95`.

Este documento foi consolidado após a publicação da tag. Portanto, ele descreve fielmente o estado da tag, mas não pertence ao conteúdo histórico dessa tag enquanto não for adicionado por um commit posterior.

Não é recomendável mover ou reescrever a tag publicada. A forma segura de versionar este documento é adicioná-lo em um commit de documentação posterior e, se necessário, criar uma nova tag documental ou incluí-lo no próximo checkpoint.

---

## 21. Conclusão

O checkpoint v0.2 fecha a principal lacuna de reprodutibilidade da inferência do piloto G-oracle.

O estado alcançado é:

- piloto científico preservado;
- ambiente computacional registrado;
- runner versionado;
- entradas validadas;
- smoke test aprovado;
- 60 consultas reproduzidas;
- 60 respostas válidas;
- nenhuma divergência determinística encontrada;
- comparador capaz de detectar uma alteração artificial;
- código integrado em `main`;
- tag v0.2 publicada;
- branches temporárias removidas;
- ponto de retomada claramente definido.

O projeto está pronto para iniciar o trabalho de reprodução do pipeline de preparação de dados. A expansão científica deve começar somente depois que as regras de seleção e geração de estímulos estiverem congeladas, reduzindo o risco de transformar escolhas exploratórias em evidência confirmatória.

---

## 22. Arquivos relacionados

| Arquivo | Função |
|---|---|
| `docs/checkpoint_v0.1_piloto_oracle.md` | documentação científica e cronológica do piloto |
| `checkpoints/v0.1/README.md` | guia do checkpoint congelado |
| `checkpoints/v0.1/manifest.jsonl` | manifesto das 60 consultas |
| `checkpoints/v0.1/results.jsonl` | respostas originais congeladas |
| `checkpoints/v0.1/metadata.json` | metadados da execução original |
| `checkpoints/v0.1/protocol.json` | protocolo do piloto |
| `checkpoints/v0.1/metrics_recomputed.json` | métricas recomputadas |
| `checkpoints/v0.1/scene_metrics.csv` | resumo por captura |
| `scripts/analyze_oracle.py` | análise científica |
| `scripts/run_oracle_mlx.py` | runner reproduzível de inferência |
| `scripts/compare_oracle_runs.py` | comparação de execuções |
| `configs/pip-freeze-macos-arm64-python311.txt` | snapshot do ambiente Python |
