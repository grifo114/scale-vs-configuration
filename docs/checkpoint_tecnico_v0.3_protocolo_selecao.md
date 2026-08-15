# Relatório técnico do checkpoint v0.3

## Protocolo reproduzível de seleção de cenas e relações métricas

**Projeto:** Scale vs. Configuration  
**Checkpoint:** v0.3  
**Data:** 15 de agosto de 2026  
**Repositório:** `grifo114/scale-vs-configuration`  
**Branch:** `phase1/scene-selection-protocol`  
**Pull request:** [PR #3](https://github.com/grifo114/scale-vs-configuration/pull/3)  
**Commit dos artefatos congelados:** `206d1d050935fd62c7c84e54da4a645d428ad3fd`  
**Estado:** conjunto de desenvolvimento congelado; avaliação confirmatória ainda não iniciada

---

## 1. Resumo executivo

Este relatório registra a construção e a validação do protocolo usado para
selecionar cenas e pares de objetos do CA-1M para experimentos de medição de
distância 3D com modelos visão-linguagem.

A etapa começou com regras exploratórias parcialmente registradas e terminou
com uma cadeia reproduzível que:

1. reconstitui a auditoria histórica de quadros;
2. audita todas as imagens das cinco capturas de desenvolvimento;
3. ranqueia quadros candidatos sem usar resultados de modelos;
4. calcula a viabilidade geométrica e visual dos pares;
5. avalia 64 configurações de legibilidade;
6. rejeita capturas que não sustentam o número mínimo de relações;
7. seleciona 12 relações balanceadas por cena;
8. renderiza os estímulos com caixas vermelha e azul;
9. valida automaticamente o manifesto final;
10. congela protocolos, hashes e evidências visuais.

Foram auditados **2.968 quadros**. A busca produziu **547 quadros candidatos**.
Quatro das cinco capturas forneceram 12 pares sob os critérios finais. A captura
`45261615` foi rejeitada. O conjunto congelado contém **4 cenas e 48
relações**.

O manifesto final passou em todas as verificações automáticas e tem SHA-256:

```text
8347d700cf8874b511ca266644c8f5bca224b821c6e745b38dc8e61add610c44
```

Este checkpoint não mede desempenho de VLMs e não demonstra melhoria de
medição. Ele registra a infraestrutura experimental que antecede essa
avaliação.

---

## 2. Posição desta etapa no projeto

O projeto está organizado, até aqui, em três checkpoints principais.

### 2.1 Checkpoint v0.1

O v0.1 registrou o piloto G-oracle:

- 5 cenas;
- 60 relações;
- Qwen3-VL-4B-Instruct-4bit;
- respostas válidas em 60 de 60 consultas;
- análise por cena;
- validação cruzada dos fatores de escala;
- documentação e hashes dos resultados.

O piloto mostrou que o procedimento de inferência funcionava, mas também que
os resultados variavam fortemente entre as cenas. A média de `G_cv` foi
aproximadamente 0,0255 e a mediana aproximadamente 0,0385. Três das cinco cenas
apresentaram ganho positivo e apenas uma superou 0,5.

Esses números não constituíram a Fase 1 planejada, pois havia apenas cinco
cenas e um modelo.

### 2.2 Checkpoint v0.2

O v0.2 tornou o piloto reproduzível:

- runner MLX versionado;
- snapshot do ambiente Python;
- ordem determinística das consultas;
- validação dos estímulos e do manifesto;
- comparador entre execuções;
- reprodução integral das 60 respostas;
- nenhuma diferença numérica ou textual nos campos determinísticos.

A reprodução confirmou que a execução poderia ser repetida sem alterar as
previsões registradas.

### 2.3 Checkpoint v0.3

O v0.3 trata de outro problema: a seleção das cenas não poderia permanecer
manual ou dependente de exemplos visualmente convenientes.

O checkpoint atual congela:

- regras de auditoria;
- critérios de elegibilidade;
- ranking de quadros;
- métricas de legibilidade;
- critérios de rejeição;
- construção balanceada dos pares;
- validação do manifesto;
- evidência visual dos estímulos.

A Fase 1 confirmatória ainda não começou. O v0.3 é o protocolo que será aplicado
antes dela.

---

## 3. Objetivo científico do protocolo

O experimento pretende estudar quanto do erro métrico de um VLM pode ser
explicado por um fator de escala comum dentro de uma cena e quanto permanece
como erro de configuração relativa.

Para que essa decomposição seja interpretável, a seleção das cenas precisa
evitar confusões básicas:

- objetos minúsculos;
- caixas cortadas pela borda;
- alvos quase totalmente sobrepostos;
- relações concentradas numa faixa estreita de distância;
- repetição excessiva do mesmo objeto;
- escolha de cenas baseada no desempenho posterior do modelo.

O protocolo v0.3 foi desenvolvido para controlar esses fatores antes da
inferência confirmatória.

---

## 4. Ambiente de desenvolvimento

A execução foi realizada no ambiente local registrado no projeto:

| Componente | Configuração |
|---|---|
| Plataforma | macOS, arm64 |
| Processador | Apple M3 |
| Núcleos | 8, sendo 4 de desempenho e 4 de eficiência |
| Memória | 16 GB |
| Python | 3.11.9 |
| OpenCV Python | 5.0.0.93 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| MLX | 0.32.0 |
| MLX-VLM | 0.6.6 |
| Transformers | 5.15.0 |

O snapshot completo do ambiente permanece em:

```text
configs/pip-freeze-macos-arm64-python311.txt
```

O ambiente atual foi comparado com o snapshot do v0.2 e não apresentou
diferenças na lista de pacotes.

---

## 5. Dados de desenvolvimento

Foram usadas cinco capturas da partição de validação do CA-1M.

| Captura | Tamanho aproximado do TAR | Quadros auditados |
|---|---:|---:|
| 42897545 | 0,23 GiB | 335 |
| 45261179 | 0,18 GiB | 280 |
| 45261615 | 0,36 GiB | 518 |
| 45662921 | 0,90 GiB | 1.135 |
| 47115543 | 0,62 GiB | 700 |
| **Total** | **2,29 GiB** | **2.968** |

Os cinco arquivos TAR foram validados como legíveis antes da auditoria.

A lista de validação contém 107 URLs válidas e únicas. Após a exclusão das cinco
capturas de desenvolvimento, restam 102 IDs candidatos para a etapa
confirmatória.

Os IDs de desenvolvimento precisam permanecer excluídos das avaliações
posteriores:

```text
42897545
45261179
45261615
45662921
47115543
```

---

## 6. Estrutura dos dados CA-1M usada

Cada quadro utilizado contém, no mínimo:

```text
image.png
image/K.json
depth.png
depth/K.json
instances.json
T_gravity.json
```

As relações métricas foram calculadas a partir das posições tridimensionais das
instâncias em `instances.json`.

Os campos relevantes por instância incluem:

- `id`;
- `category`;
- `caption`;
- `position`;
- `scale`;
- `R`;
- `corners`;
- `box_2d_rend`;
- `box_2d_proj`.

A distância de referência entre dois objetos é a distância euclidiana entre os
centros das respectivas caixas 3D.

A consistência entre `position` e a média dos oito cantos foi verificada nos
quadros exploratórios, com diferenças numéricas próximas da precisão de ponto
flutuante.

---

## 7. Reconstrução das regras históricas

### 7.1 Motivo

A primeira auditoria exploratória havia produzido contagens por quadro, mas as
regras exatas não estavam formalizadas num script versionado. Sem reconstrução,
a seleção posterior não seria reproduzível.

O script `validate_frame_audit_rules.py` avaliou combinações possíveis de:

- campo de caixa 2D;
- política para categorias duplicadas;
- exigência ou não de legenda;
- definição de interior;
- ordem das etapas;
- margem de borda.

### 7.2 Regra reconstruída

A configuração que reproduziu o histórico foi:

- campo de caixa: `box_2d_rend`;
- duplicatas avaliadas no conjunto de todas as instâncias;
- legenda ignorada;
- interior avaliado pela caixa completa;
- natural interior calculado como subconjunto dos objetos naturais;
- margem interior de 5 px.

A regra geométrica básica exige:

| Critério | Valor |
|---|---:|
| área mínima da caixa | 1.000 px² |
| largura mínima | 20 px |
| altura mínima | 20 px |
| cobertura máxima da imagem | 0,5 |

Categorias genéricas excluídas:

```text
baseboard
ceiling
floor
object
wall
```

Um objeto natural é uma instância geometricamente válida cuja categoria:

1. não pertence à lista genérica;
2. aparece uma única vez entre todas as instâncias do quadro.

A presença de `caption` não é obrigatória.

### 7.3 Resultado da reconstrução

A comparação foi realizada em 1.833 quadros das quatro capturas auditadas
historicamente.

Resultado:

```text
Mismatched rows: 0/1833
```

Hash da auditoria reconstruída:

```text
6a3fd405df2e8d1a20bdf5edfdc560ff0da1a22ee13ad21b08bb0edc36be06f4
```

A captura 45662921 foi processada separadamente com as mesmas regras:

- quadros: 1.135;
- hash:

```text
03f163a7aeecb40a8a24a00e093d2b3431468f76ceff55421b62531ce3d57d7f
```

---

## 8. Reprodução da métrica de nitidez

A métrica de nitidez histórica também foi reconstruída.

Pipeline confirmado:

1. imagem decodificada pelo Pillow em RGB;
2. conversão para tons de cinza com `cv2.COLOR_RGB2GRAY`;
3. Laplaciano com `cv2.CV_64F`;
4. variância da resposta do Laplaciano.

Expressão:

```python
cv2.Laplacian(gray, cv2.CV_64F).var()
```

Foram testadas alternativas, incluindo conversões do Pillow e cálculo por
canais RGB. Pequenas diferenças de pixels e valores mostraram que essas
alternativas não reproduziam exatamente o histórico.

A implementação confirmada foi registrada em `ca1m_audit_core.py` e
`audit_ca1m_frames.py`.

---

## 9. Auditoria integral dos quadros

O script `audit_ca1m_frames.py` foi executado em duas etapas.

### 9.1 Quatro capturas históricas

- arquivos TAR: 4;
- quadros: 1.833;
- divergências em relação ao histórico: 0.

### 9.2 Quinta captura

- captura: 45662921;
- quadros: 1.135;
- execução integral sem referência histórica.

### 9.3 Total

```text
1.833 + 1.135 = 2.968 quadros
```

Os metadados congelados registram:

- arquivos de entrada;
- contagens por captura;
- regras;
- operador de nitidez;
- caminhos dos resultados;
- hashes dos JSONL completos.

Os JSONL completos permanecem fora do Git por serem resultados derivados. Eles
podem ser reconstruídos a partir dos TARs e scripts versionados.

---

## 10. Desenvolvimento do ranking de cenas

### 10.1 Critérios iniciais de quadro

Um quadro poderia entrar no ranking quando atendesse a:

| Critério | Valor |
|---|---:|
| objetos oracle | pelo menos 8 |
| objetos naturais | pelo menos 6 |
| objetos naturais interiores | pelo menos 3 |

A prioridade de ranking foi:

1. naturais interiores, decrescente;
2. naturais, decrescente;
3. oracle, decrescente;
4. nitidez, decrescente;
5. índice do quadro, crescente;
6. nome do membro de imagem, crescente.

### 10.2 Primeira shortlist

A primeira shortlist continha 15 quadros. A inspeção visual mostrou problemas
que as contagens básicas não capturavam:

- objetos pequenos;
- regiões próximas da borda;
- fragmentos estreitos;
- caixas que continham muita interferência visual;
- pares tecnicamente válidos, mas fracos para o estímulo final.

Essa etapa foi usada apenas para desenvolvimento das regras.

### 10.3 Shortlist expandida

O ranking foi ampliado para 54 candidatos:

- até cinco candidatos por bin temporal;
- até 25 candidatos por captura.

Mesmo com a expansão, nenhuma configuração avaliada sustentou 12 pares
legíveis nas cinco capturas.

### 10.4 Busca exaustiva

A restrição de shortlist foi removida no conjunto de desenvolvimento:

- quantil de nitidez: 0;
- limite por bin: 10.000;
- limite por captura: 10.000.

Resultados:

| Captura | Quadros auditados | Quadros candidatos |
|---|---:|---:|
| 42897545 | 335 | 23 |
| 45261179 | 280 | 51 |
| 45261615 | 518 | 53 |
| 45662921 | 1.135 | 204 |
| 47115543 | 700 | 216 |
| **Total** | **2.968** | **547** |

Hash do manifesto de candidatos:

```text
533860c0dfd683380d2fc595b6b49c0aec43dbe44b03a122094cbc3a63eb22fd
```

O protocolo registra explicitamente:

```json
"uses_model_outputs": false
```

---

## 11. Avaliação de viabilidade dos pares

O script `evaluate_pair_feasibility.py` constrói todos os pares entre objetos
naturais elegíveis de cada quadro.

Critérios e diagnósticos calculados:

- distância 3D;
- fração visível;
- IoU entre caixas;
- interseção dividida pela área da menor caixa;
- posição em relação às bordas;
- número de pares interiores;
- suporte sob diferentes limiares de visibilidade.

Configuração básica:

| Critério | Valor |
|---|---:|
| distância mínima | 0,25 m |
| pares mínimos | 12 |
| visibilidades reportadas | 0,8; 0,9; 1,0 |
| IoUs reportados | 0,25; 0,50 |
| interseção sobre menor caixa | 0,80 |

Resultado:

- candidatos avaliados: 547;
- candidatos geometricamente viáveis na regra básica: 547;
- quadros inicialmente escolhidos pelo ranking com suporte básico: 5 de 5.

Isso não significava que todos forneciam 12 pares legíveis sob critérios mais
estritos. A legibilidade foi tratada na etapa seguinte.

Hash da análise completa de viabilidade:

```text
f554a7b61f7d9972c5b78021b9f5a944f024e37dcae592c01df6427c4a9b56e9
```

---

## 12. Sweep de legibilidade

### 12.1 Espaço de configurações

O script `sweep_pair_legibility.py` avaliou o produto cartesiano de:

- quatro frações mínimas de área;
- quatro tamanhos mínimos de lado;
- quatro margens mínimas de borda.

Total:

```text
4 × 4 × 4 = 64 configurações
```

Os filtros geométricos comuns foram mantidos:

- distância mínima de 0,25 m;
- fração visível de 1,0;
- IoU máximo de 0,50;
- interseção sobre a menor caixa de 0,80;
- 12 pares necessários por captura.

### 12.2 Resultado do sweep

- capturas avaliadas: 5;
- candidatos avaliados: 547;
- configurações com cobertura de 5 de 5: 0;
- maior cobertura: 4 de 5;
- configurações com a maior cobertura: 24.

Hash do sweep completo:

```text
cfd6d766ddc362ebf501c10336468235e7483a596ae14ac189f40375bfcfcc61
```

O arquivo completo tem aproximadamente 8,2 MB e permanece como resultado
derivado. O checkpoint contém um resumo de aproximadamente 90 KB com:

- as 64 configurações;
- o melhor quadro por captura em cada configuração;
- número de pares;
- viabilidade por captura;
- hash do arquivo completo.

### 12.3 Configuração escolhida

Entre as configurações de cobertura máxima, foi adotada:

| Critério | Valor |
|---|---:|
| área mínima da menor caixa | 0,00375 da imagem |
| menor lado | 30 px |
| margem da borda | 20 px |
| capturas viáveis | 4 de 5 |
| menor suporte entre as cinco capturas | 6 pares |

Essa configuração fornece 12 pares ou mais em quatro capturas e apenas seis na
captura 45261615.

A captura 45261615 foi rejeitada. Os critérios não foram relaxados para mantê-la.

---

## 13. Regra final de seleção da cena

Para cada captura:

1. todos os quadros candidatos são avaliados sob os filtros congelados;
2. escolhe-se o quadro com o maior número de pares elegíveis;
3. o primeiro desempate é o rank do candidato;
4. o segundo desempate é o índice do quadro;
5. a captura é rejeitada se o melhor quadro tiver menos de 12 pares.

Resultado:

| Captura | Candidatos | Máximo de pares | Rank escolhido | Quadro | Estado |
|---|---:|---:|---:|---:|---|
| 42897545 | 23 | 12 | 2 | 100 | aceita |
| 45261179 | 51 | 14 | 1 | 31 | aceita |
| 45261615 | 53 | 6 | — | — | rejeitada |
| 45662921 | 204 | 48 | 2 | 946 | aceita |
| 47115543 | 216 | 21 | 4 | 86 | aceita |

A rejeição é parte do protocolo e não um erro de execução.

---

## 14. Regra final de seleção das relações

### 14.1 Filtros duros

Um par entra no conjunto elegível quando atende simultaneamente a:

| Critério | Valor |
|---|---:|
| distância 3D | ≥ 0,25 m |
| menor fração visível | ≥ 1,0 |
| IoU | ≤ 0,50 |
| interseção sobre a menor caixa | ≤ 0,80 |
| menor área de caixa | ≥ 0,00375 da imagem |
| menor lado de caixa | ≥ 30 px |
| menor margem da borda | ≥ 20 px |

### 14.2 Particionamento por distância

Os pares elegíveis são ordenados pela distância 3D e divididos em três grupos
contíguos:

- curta;
- média;
- longa.

Cada faixa fornece quatro relações.

Dentro de cada faixa:

1. os pares são separados em quatro subintervalos contíguos;
2. um par é escolhido por subintervalo;
3. a prioridade favorece maior área mínima das caixas;
4. depois maior distância entre centros 2D;
5. depois menor IoU;
6. depois menor interseção sobre a caixa menor;
7. depois pares interiores;
8. por fim, índices das instâncias.

### 14.3 Folds

A atribuição é determinística:

- índices ímpares: fold A;
- índices pares: fold B.

Por cena:

- 6 relações no fold A;
- 6 no fold B;
- 2 relações de cada fold em cada faixa de distância.

---

## 15. Manifesto final

### 15.1 Resumo por cena

| Captura | Pares elegíveis | Pares finais | Distância final | Objetos | Uso máximo |
|---|---:|---:|---:|---:|---:|
| 42897545 | 12 | 12 | 0,266–0,886 m | 6 | 5 |
| 45261179 | 14 | 12 | 0,378–1,140 m | 6 | 5 |
| 45662921 | 48 | 12 | 0,288–0,775 m | 9 | 5 |
| 47115543 | 21 | 12 | 0,364–2,627 m | 7 | 4 |
| **Total** | — | **48** | — | — | — |

### 15.2 Razão entre maior e menor distância

| Captura | Razão |
|---|---:|
| 42897545 | 3,332 |
| 45261179 | 3,013 |
| 45662921 | 2,695 |
| 47115543 | 7,217 |

O validador exige razão mínima de 2,5. O menor valor observado foi 2,695 na
captura 45662921.

Esse limiar foi definido durante o desenvolvimento. Ele deve permanecer
congelado no conjunto confirmatório.

### 15.3 Faixas por cena

#### Captura 42897545

| Faixa | Intervalo selecionado |
|---|---:|
| curta | 0,266–0,463 m |
| média | 0,464–0,676 m |
| longa | 0,862–0,886 m |

#### Captura 45261179

| Faixa | Intervalo selecionado |
|---|---:|
| curta | 0,378–0,542 m |
| média | 0,793–0,846 m |
| longa | 1,072–1,140 m |

#### Captura 45662921

| Faixa | Intervalo selecionado |
|---|---:|
| curta | 0,288–0,494 m |
| média | 0,538–0,631 m |
| longa | 0,642–0,775 m |

#### Captura 47115543

| Faixa | Intervalo selecionado |
|---|---:|
| curta | 0,364–0,896 m |
| média | 1,192–1,462 m |
| longa | 1,657–2,627 m |

As faixas são relativas à distribuição de pares de cada cena, não intervalos
métricos globais compartilhados.

---

## 16. Renderização dos estímulos

Cada estímulo contém:

- a imagem RGB original;
- objeto A marcado em vermelho;
- objeto B marcado em azul;
- letras A e B nas caixas.

As imagens de inferência não contêm:

- distância de referência;
- categoria dos objetos.

Essas informações aparecem apenas nas folhas usadas para auditoria humana.

Parâmetros registrados:

```json
{
  "object_a": {"label": "A", "color": "#E53935"},
  "object_b": {"label": "B", "color": "#1565E8"},
  "ground_truth_in_stimulus": false,
  "categories_in_stimulus": false
}
```

Resultados:

- cenas renderizadas: 4;
- estímulos: 48;
- hash do manifesto renderizado:

```text
ba98b546b82af133e6146bd2d2a9739d2ed6fa43784bf79ef42bcffdc7c087ce
```

- hash da folha de contato geral:

```text
9089f842b0b5fa1de90f6451788c6fc8a2f96711b1d25c145d2dacf4fd003aa2
```

---

## 17. Auditoria visual

A auditoria visual foi feita depois da seleção automática.

Foram examinados:

- posição das caixas;
- tamanho dos alvos;
- proximidade das bordas;
- sobreposição;
- presença de fragmentos;
- quantidade de interferência visual;
- correspondência aparente entre caixa e região anotada.

### 17.1 Evolução observada durante o desenvolvimento

As primeiras folhas de contato revelaram:

- objetos cortados;
- caixas estreitas próximas às bordas;
- alvos pequenos;
- relações dominadas por regiões difíceis de distinguir.

Esses problemas motivaram o sweep de legibilidade e a busca exaustiva no
conjunto de desenvolvimento.

### 17.2 Folha final

Na folha final:

- não permaneceram alvos cortados pela borda;
- as duas caixas aparecem nos 48 estímulos;
- os pares respeitam os limites automáticos;
- as quatro cenas selecionadas foram mantidas.

A avaliação visual foi feita por um único pesquisador e não foi cega. Portanto,
ela deve ser tratada como controle de qualidade de desenvolvimento, não como
anotação humana independente.

A folha final está congelada em:

```text
checkpoints/v0.3/contact_sheet_all.png
```

---

## 18. Validador do manifesto

O script `validate_pair_manifest.py` verifica:

- JSONL válido;
- 12 pares por cena;
- IDs únicos;
- índices contíguos;
- pares sem duplicação invertida;
- objetos A e B distintos;
- distâncias positivas e ordenadas;
- distância mínima;
- quatro pares em cada faixa;
- seis pares em cada fold;
- dois pares por combinação faixa/fold;
- fração visível;
- IoU;
- interseção sobre a menor caixa;
- área mínima;
- lado mínimo;
- margem da borda;
- objetos interiores;
- número mínimo de objetos distintos;
- uso máximo por objeto;
- razão mínima entre distâncias;
- correspondência do SHA-256 com o protocolo.

Comando aplicado ao checkpoint:

```bash
python3 scripts/validate_pair_manifest.py \
  checkpoints/v0.3/pair_selection_manifest.jsonl \
  --protocol checkpoints/v0.3/pair_selection_protocol.json
```

Resultado:

```text
Pair-manifest validation
Scenes: 4
Pairs: 48
Result: PASSED
```

---

## 19. Integridade do checkpoint

O diretório `checkpoints/v0.3` contém 13 artefatos verificados por
`SHA256SUMS`, além do próprio arquivo de hashes.

| Artefato | SHA-256 |
|---|---|
| README.md | `dbcd353d7c07198c3f9e10b9bf92c89151e6e6e4346448953a85735696cf3582` |
| audit_45662921_metadata.json | `f0772fa0611385270dfec871c41c4b6784273d04588a389696bac8c87b19a22d` |
| audit_reconstructed_metadata.json | `dcc668ef04fbb40622f0327c396718876fd2bd2df80b54af1aa44a242528ee7c` |
| audit_rule_reconstruction.txt | `50b26474ee21a49fd474085fa1695de4934afe43135e9c5cd9cb1de39d9d1c01` |
| candidate_ranking_protocol.json | `5443a4af0e02e057a88847cfe514c5ddc2bb727ec5db12cd7f420a39f964410c` |
| contact_sheet_all.png | `9089f842b0b5fa1de90f6451788c6fc8a2f96711b1d25c145d2dacf4fd003aa2` |
| legibility_sweep_summary.json | `43247616ee8714b06c0815774aeac99563e769782fb72b988518f4ee1c421f46` |
| pair_feasibility_protocol.json | `fc6d04d7f70e9c300d2ee6dedcba97c79478649aa1ae7893aa758bca364a6567` |
| pair_selection_manifest.jsonl | `8347d700cf8874b511ca266644c8f5bca224b821c6e745b38dc8e61add610c44` |
| pair_selection_protocol.json | `9ad8e51bc14566c5cc8f781c332ecf4f52b85f2ce2f970bfc9f06f90f953447f` |
| render_protocol.json | `ef1b0097274aeb94a1c81bf6faa6c4724bc62ba0f9017ba998c51d703d580f0e` |
| rendered_manifest.jsonl | `ba98b546b82af133e6146bd2d2a9739d2ed6fa43784bf79ef42bcffdc7c087ce` |
| validation_report.txt | `05d8990dcfdf9f7c230a536cb3e0294eb183ddd8309444532ef13e1e3fe33d3b` |

A verificação local retornou `OK` para todos os arquivos.

Nenhum padrão de credencial foi encontrado no checkpoint.

Alguns metadados preservam o caminho local
`/Users/jeffersonlopes/scale-vs-configuration`. Esse caminho não é uma
credencial, mas deverá ser revisado se o repositório for publicado.

---

## 20. Scripts adicionados nesta etapa

| Script | Responsabilidade |
|---|---|
| `ca1m_audit_core.py` | regras compartilhadas da auditoria CA-1M |
| `validate_frame_audit_rules.py` | reconstrução das regras históricas |
| `audit_ca1m_frames.py` | auditoria reproduzível dos TARs |
| `rank_scene_candidates.py` | ranking determinístico dos quadros |
| `evaluate_pair_feasibility.py` | análise geométrica e visual dos pares |
| `render_scene_shortlist.py` | folha visual dos candidatos |
| `sweep_pair_legibility.py` | busca dos limiares de legibilidade |
| `select_balanced_pairs.py` | seleção das cenas e relações |
| `render_pair_stimuli.py` | geração dos estímulos A/B |
| `validate_pair_manifest.py` | validação final do manifesto |

Os scripts foram adicionados ao branch
`phase1/scene-selection-protocol` e fazem parte do PR #3.

---

## 21. Decisões metodológicas congeladas

As decisões seguintes não devem ser alteradas depois de observar saídas
confirmatórias dos modelos:

1. cinco capturas de desenvolvimento excluídas;
2. categorias genéricas excluídas;
3. unicidade da categoria no quadro;
4. campo `box_2d_rend`;
5. distância mínima de 0,25 m;
6. fração visível de 1,0;
7. IoU máximo de 0,50;
8. interseção sobre menor caixa de 0,80;
9. área mínima de 0,00375;
10. lado mínimo de 30 px;
11. margem mínima de 20 px;
12. 12 pares por cena;
13. três faixas com quatro pares cada;
14. folds A/B alternados;
15. pelo menos seis objetos;
16. uso máximo de cinco por objeto;
17. razão mínima de distâncias de 2,5;
18. rejeição da captura quando o melhor quadro tem menos de 12 pares.

Se o conjunto confirmatório não alcançar o número planejado de cenas, qualquer
fallback precisa ser registrado antes da inferência dos modelos.

---

## 22. Limitações

### 22.1 Número pequeno de capturas

O protocolo foi desenvolvido em apenas cinco capturas. Não é possível inferir
a taxa de aprovação nas 102 capturas restantes a partir desse conjunto pequeno.

### 22.2 Ajuste no conjunto de desenvolvimento

Os limiares foram definidos depois de inspecionar candidatos e folhas de
contato do conjunto de desenvolvimento. Isso é aceitável como calibração de
protocolo, mas impede usar as cinco capturas como evidência confirmatória.

### 22.3 Rejeição de uma captura

A captura 45261615 não forneceu 12 pares sob os critérios finais. Essa rejeição
mostra que a disponibilidade geométrica básica não equivale à legibilidade
necessária para os estímulos.

### 22.4 Auditoria visual individual

A inspeção foi realizada por um pesquisador. Não houve dupla anotação,
cegamento ou medida de concordância.

### 22.5 Faixas relativas à cena

Curta, média e longa são partições internas de cada cena. Os limites não são
iguais entre capturas. Comparações que dependam de faixas métricas absolutas
precisam tratar essa diferença explicitamente.

### 22.6 Distribuição desigual de amplitude

A razão entre distâncias varia de 2,695 a 7,217. O protocolo controla um mínimo,
mas não equaliza completamente a amplitude entre cenas.

### 22.7 Dependência das anotações CA-1M

O experimento assume que posições e caixas do CA-1M são referências adequadas.
O checkpoint verificou consistência estrutural, mas não realizou uma nova
medição física independente das cenas.

### 22.8 Ausência de resultado de modelo

Nenhum VLM foi avaliado neste conjunto congelado. O checkpoint não responde se
a decomposição escala/configuração se confirma nem se um método posterior
melhora a medição.

---

## 23. Riscos para a próxima etapa

### 23.1 Insuficiência de cenas

Ainda não foi medido quantas das 102 capturas restantes produzirão 12 pares sob
os critérios congelados.

### 23.2 Custo de armazenamento e download

As capturas CA-1M variam de centenas de MB a mais de 1 GiB. A seleção de 50
cenas pode exigir o processamento de mais de 50 arquivos TAR.

### 23.3 Dependência entre relações

Cada cena fornece 12 relações e reutiliza objetos. As relações não devem ser
tratadas estatisticamente como observações totalmente independentes. A unidade
primária continua sendo a cena.

### 23.4 Seleção orientada por disponibilidade

Cenas com muitos objetos legíveis têm maior chance de entrar. O conjunto final
representará cenas adequadas ao protocolo de medição, não necessariamente toda
a distribuição do CA-1M.

### 23.5 Mudanças após observar resultados

Alterar filtros, pares ou cenas depois de executar os modelos introduziria
dependência entre seleção e desempenho. O checkpoint v0.3 existe para separar
essas etapas.

---

## 24. Próximos passos

### 24.1 Fechamento do v0.3

1. revisar o PR #3;
2. marcar o PR como pronto;
3. integrar em `main`;
4. criar a tag `v0.3-scene-selection-protocol`;
5. remover o branch após a integração.

### 24.2 Construção do conjunto confirmatório

1. excluir os cinco IDs de desenvolvimento;
2. processar as capturas restantes em lotes;
3. auditar todos os quadros de cada captura;
4. aplicar o ranking congelado;
5. calcular a viabilidade dos pares;
6. aplicar os filtros finais;
7. registrar capturas aceitas e rejeitadas;
8. interromper a seleção ao atingir o número planejado de cenas;
9. renderizar os estímulos;
10. realizar a auditoria visual conforme regra previamente registrada;
11. congelar manifesto, protocolos e hashes;
12. validar o conjunto;
13. somente depois executar os modelos.

### 24.3 Avaliação experimental

A Fase 1 planejada prevê:

- 50 cenas;
- 12 relações por cena;
- 600 relações por modelo;
- três modelos;
- análise por cena;
- decomposição em escala e configuração;
- bootstrap com a cena como unidade;
- comparação com baselines e controles definidos no plano experimental.

Esses números permanecem planejados e não executados.

---

## 25. Critério de conclusão desta etapa

O protocolo de seleção pode ser considerado concluído para desenvolvimento
porque:

- as regras históricas foram reproduzidas em 1.833 de 1.833 quadros;
- os 2.968 quadros foram auditados;
- a busca foi ampliada até todos os 547 candidatos elegíveis;
- 64 configurações foram comparadas;
- a regra de rejeição foi aplicada;
- o manifesto final foi criado sem saídas de modelo;
- os 48 estímulos foram renderizados;
- a inspeção visual foi registrada;
- o validador retornou `PASSED`;
- os artefatos receberam hashes;
- o checkpoint foi versionado.

Isso encerra a construção do protocolo no conjunto de desenvolvimento. A
próxima evidência científica virá da aplicação congelada às capturas não usadas
e da execução dos modelos sobre o conjunto confirmatório.
