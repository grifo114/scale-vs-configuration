# Checkpoint v0.1 — Piloto G-oracle de escala versus configuração em VLMs

**Projeto:** *Escala ou configuração? Decomposição validada do erro métrico em VLMs e medição com escala revisável*
**Data do checkpoint:** 13 de agosto de 2026, horário local; execução registrada entre 01:41:47 e 01:46:38 UTC de 14 de agosto de 2026
**Fase atual:** Fase 1A — piloto técnico do Experimento 1
**Condição avaliada:** G-oracle
**Modelo:** `mlx-community/Qwen3-VL-4B-Instruct-4bit`
**Revisão do modelo:** `2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b`
**Escopo deste checkpoint:** cinco capturas, 12 relações por captura, 60 consultas válidas

---

## 1. Resumo executivo

Este checkpoint registra a construção e a primeira execução multiscena do núcleo diagnóstico do projeto. O objetivo foi verificar se um único fator multiplicativo, estimado em algumas relações métricas de uma cena, reduz o erro em relações não utilizadas no ajuste.

O pipeline técnico foi concluído com sucesso:

- cinco capturas do CA-1M foram preparadas;
- 60 relações centro–centro receberam verdade-terreno 3D;
- cada consulta usou uma imagem própria com apenas dois alvos marcados, A em vermelho e B em azul;
- o modelo retornou 60 de 60 respostas válidas no formato JSON;
- a validação cruzada foi calculada dentro de cada captura e agregada depois com peso igual por captura;
- o manifesto, o protocolo, os metadados, as respostas brutas e seus hashes foram preservados.

O resultado científico preliminar é misto:

- média de \(G^{\mathrm{CV}}\) por captura: **0,025**;
- mediana de \(G^{\mathrm{CV}}\): **0,038**;
- capturas com ganho positivo: **3 de 5**;
- capturas com \(G^{\mathrm{CV}}>0{,}5\): **1 de 5**;
- fração descritiva média de escala: **0,275**;
- fração descritiva mediana de escala: **0,201**.

Nesse piloto, a correção de escala foi útil em algumas cenas, quase neutra em outra e prejudicial em duas. O critério de dominância não foi atingido no agregado primário. O resultado não encerra a hipótese geral, pois a Fase 1 prevê 50 cenas e três modelos, enquanto este checkpoint cobre cinco capturas e um modelo quantizado de 4 bilhões de parâmetros.

O principal resultado deste estágio é duplo:

1. o protocolo é executável, rastreável e produz o endpoint pré-especificado;
2. a coerência de escala não apareceu como propriedade universal do modelo testado, variando fortemente entre as capturas.

---

## 2. Pergunta científica e hipótese operacional

### 2.1 Problema

VLMs podem fornecer estimativas métricas erradas por razões estruturalmente diferentes. Duas componentes são tratadas separadamente:

- **erro de escala:** todas ou muitas relações da cena são multiplicadas aproximadamente pelo mesmo fator;
- **erro de configuração:** as relações internas sofrem distorções diferentes, mesmo depois da remoção de um fator global.

A distinção importa porque uma escala coerente pode ser revisada por uma única referência física. Uma configuração deformada exige corrigir relações individualmente ou melhorar a representação geométrica.

### 2.2 Hipótese principal

> Uma parte mensurável do erro métrico de VLMs é uma componente multiplicativa coerente por cena: um fator estimado em algumas relações reduz o erro em outras relações da mesma cena.

### 2.3 Hipótese de dominância

> Em condições comparáveis, a componente de escala responde por mais da metade do erro quadrático em log.

A existência de alguma coerência de escala e a dominância dessa componente são hipóteses diferentes. O projeto não deve usar evidência da primeira como prova automática da segunda.

### 2.4 Convenção métrica

A medida primária é a distância euclidiana entre os centros das caixas 3D dos dois objetos:

\[
d(a,b)=\lVert \mathbf{p}_a-\mathbf{p}_b\rVert_2.
\]

Essa definição foi escolhida porque o campo `position` do CA-1M coincide numericamente com a média dos oito cantos da caixa 3D. Nas cinco cenas selecionadas, a maior diferença observada entre essas duas formas de calcular o centro foi de aproximadamente \(4\times10^{-9}\) m nas quatro novas cenas; a auditoria inicial encontrou diferença máxima de aproximadamente \(5{,}4\times10^{-8}\) m em outro frame.

---

## 3. Posição atual no plano completo

| Fase | Objetivo | Estado neste checkpoint |
|---|---|---|
| Fase 0 | Auditoria bibliográfica, dados, convenções e especificação | Parcialmente concluída; plano, dados e métrica definidos, mas a auditoria sistemática de novidade ainda precisa ser fechada |
| Fase 1 | Piloto com 50 cenas, ao menos 8 relações e três modelos | Em andamento; 5 capturas, 12 relações por captura e 1 modelo concluídos |
| Fase 2 | Estudo principal com 300–500 cenas e painel de modelos | Não iniciado |
| Fase 3 | Experimentos de focal e âncoras | Não iniciado |
| Fase 4 | Método modular M0 | Não iniciado |
| Fase 5 | VLM fatorado M1 | Não iniciado |
| Fase 6 | Consolidação, reprodução limpa e submissão | Não iniciado |

Portanto, este documento congela um **piloto técnico multiscena**, não o experimento principal do artigo.

---

## 4. Atualização do posicionamento bibliográfico

A novidade provisória foi reposicionada de “descobrir a ambiguidade de escala” para “decompor e sistematizar, com validação fora da amostra, escala global e configuração residual”. A revisão preliminar registrada no plano precisa considerar explicitamente:

- [DepthLM](https://arxiv.org/html/2509.25413v1), sobre intrínsecos e focal;
- [Language as Prior, Vision as Calibration](https://arxiv.org/abs/2601.01457), sobre linguagem como pista de escala;
- [VANGUARD](https://arxiv.org/pdf/2603.04277), sobre correção por âncora física em prompt;
- SD-VLM, CRISP, SpatialVLM, SpatialRGPT, SpatialPrompt, Q-Spatial, WorDepth, RSA e trabalhos correlatos listados no plano atualizado.

A alegação de novidade permanece provisória até a auditoria sistemática da Fase 0 e uma nova busca imediatamente antes da submissão.

---

## 5. Ambiente computacional

### 5.1 Hardware

- computador: MacBook Air;
- chip: Apple M3;
- CPU: 8 núcleos, sendo 4 de desempenho e 4 de eficiência;
- memória unificada: 16 GB;
- arquitetura: `arm64`;
- armazenamento livre antes da rodada principal: aproximadamente 143 GiB.

### 5.2 Software

- Python: `3.11.9`;
- MLX: `0.32.0`;
- MLX-VLM: `0.6.6`;
- Transformers: `5.15.0`;
- plataforma registrada: `macOS-26.6.1-arm64-arm-64bit`;
- ambiente virtual: `.venv` na raiz do projeto.

### 5.3 Modelo

- identificador: `mlx-community/Qwen3-VL-4B-Instruct-4bit`;
- modelo-base: `Qwen/Qwen3-VL-4B-Instruct`;
- quantização: 4 bits para MLX;
- revisão congelada: `2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b`;
- temperatura: `0.0`;
- máximo de tokens: `64`;
- semente de geração: `0`;
- semente da ordem aleatória: `20260814`.

O checkpoint usa uma conversão quantizada da comunidade, e não o checkpoint original em precisão integral. Os resultados desta rodada devem ser atribuídos especificamente a essa configuração.

Fontes técnicas: [modelo-base oficial](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [conversão MLX](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit) e [MLX-VLM](https://github.com/Blaizzy/mlx-vlm).

---

## 6. Preparação do projeto e dos dados

### 6.1 Estrutura inicial

O trabalho começou com a criação do diretório `scale-vs-configuration` e o clone do repositório oficial da Apple:

```bash
mkdir -p scale-vs-configuration
cd scale-vs-configuration
git clone https://github.com/apple/ml-cubifyanything.git
```

Os dados foram mantidos fora do repositório upstream, em `data/ca1m/`, evitando misturar dados pesados com o código de origem.

### 6.2 Primeiro arquivo

O primeiro item de `ml-cubifyanything/data/val.txt` foi:

```text
https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-45662921.tar
```

O servidor informou 964.608.000 bytes, aproximadamente 0,90 GiB. O arquivo local ocupou cerca de 920 MiB e passou pela leitura integral do índice TAR.

### 6.3 Estrutura observada em um frame

O arquivo contém, por frame:

```text
T_gravity.json
depth.png
depth/K.json
image.png
image/K.json
instances.json
```

O primeiro `instances.json` inspecionado tinha raiz do tipo lista, 20 instâncias e os campos:

```text
R
box_2d_proj
box_2d_rend
caption
category
corners
id
position
scale
```

Essa estrutura fornece associação 2D–3D, centros, dimensões, rotação, cantos 3D, categoria e descrição textual.

### 6.4 Licença e proveniência

Os dados vieram do [repositório oficial `apple/ml-cubifyanything`](https://github.com/apple/ml-cubifyanything). A licença registrada para o conjunto é CC-BY-NC-ND. Por isso, os TARs, imagens e estímulos derivados devem permanecer fora de um repositório público até uma revisão específica das condições de redistribuição. Manifestos, identificadores, hashes, código e resultados numéricos podem ser organizados separadamente, respeitando a licença.

---

## 7. Auditoria da primeira captura

A captura `45662921` contém 1.135 frames com `instances.json`.

### 7.1 Filtros exploratórios

Os filtros visuais usados no piloto foram:

- área mínima da caixa 2D: 1.000 px²;
- lado mínimo: 20 px;
- cobertura máxima da caixa: 50% da imagem;
- candidato natural: categoria não genérica e única dentro do frame;
- categorias genéricas iniciais: `object`, `baseboard`, `wall`, `floor` e `ceiling`.

Esses limites foram exploratórios. Eles ainda precisam ser congelados formalmente antes da ampliação para 50 cenas.

### 7.2 Resultados da auditoria

| Contagem por frame | Mínimo | Q1 | Mediana | Q3 | Máximo | Média |
|---|---:|---:|---:|---:|---:|---:|
| Total | 1 | 7 | 12 | 19 | 45 | 14,02 |
| Geométrico | 1 | 7 | 12 | 19 | 45 | 14,02 |
| Oracle | 0 | 6 | 10 | 15 | 37 | 10,74 |
| Natural | 0 | 2 | 3 | 6 | 17 | 4,49 |

Cobertura observada:

- 732 frames tinham ao menos 8 candidatos oracle;
- 199 frames tinham ao menos 8 candidatos naturais;
- 866 frames tinham ao menos 6 candidatos oracle;
- 310 frames tinham ao menos 6 candidatos naturais.

O portão de densidade de relações foi atendido para o piloto.

### 7.3 Limitação encontrada na seleção automática

O primeiro ranking priorizou apenas a quantidade de candidatos. O frame mais bem ranqueado, `21859797734958`, tinha 21 candidatos oracle e 17 naturais, mas apresentava desfoque de movimento, objetos parcialmente cortados e muitas caixas sobrepostas.

Essa observação levou à inclusão da variância do Laplaciano como indicador de nitidez para ordenar candidatos nas capturas seguintes. A nitidez foi usada apenas para triagem visual, não como endpoint científico.

---

## 8. Ampliação para cinco capturas

Foram baixados e validados quatro TARs adicionais. Junto da captura inicial, o piloto passou a usar cinco IDs distintos.

| Captura | Tamanho local aproximado | Frames válidos | Mediana oracle | Mediana natural | Frames com natural ≥ 8 | Frame/bin selecionado | Oracle selecionado | Natural selecionado | Nitidez | Ambiente observado |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 45662921 | 920 MiB | 1.135 | 10 | 3 | 199 | `21859797734958` | 21 | 17 | — | cozinha |
| 45261179 | 181 MiB | 280 | 8 | 3 | 21 | bin 1 | 10 | 6 | 107,1 | banheiro |
| 42897545 | 238 MiB | 335 | 4 | 2 | 29 | bin 2 | 11 | 9 | 35,9 | banheiro |
| 45261615 | 369 MiB | 518 | 10 | 4 | 33 | bin 2 | 14 | 6 | 589,3 | sala/quarto |
| 47115543 | 631 MiB | 700 | 10 | 5 | 145 | bin 4 | 17 | 10 | 184,1 | escritório/sala de jantar |

[Não verificado] Os IDs representam capturas distintas e os frames mostram ambientes visualmente diferentes, mas não foi verificado se todas as capturas pertencem a imóveis fisicamente distintos. A unidade analítica conservadora deste piloto é a captura.

### 8.1 Objetos usados para construir pares

| Captura | IDs e categorias selecionados |
|---|---|
| 45662921 | 01 cabinet; 04 mug; 05 sink; 12 bowl; 13 glove; 14 chopping board; 15 plate holder; 16 utensil holder; 19 faucet |
| 45261179 | 01 exhaust fan; 02 toothbrush holder; 05 mirror; 06 recess light; 07 object; 08 switch board; 11 toothpaste |
| 42897545 | 01 cabinet; 02 sink; 03 countertop; 05 spray bottle; 06 mixer tap; 07 toilet paper roll; 09 tissue box; 10 watering can |
| 45261615 | 07 flower vase; 08 bottle; 09 toy; 12 tray; 13–16 cushions |
| 47115543 | 04 light fixture; 05 television; 06 laptop; 07 chair; 08 desk; 09 window; 11 mirror; 16 trash can |

No G-oracle, categorias repetidas, como as almofadas, permanecem utilizáveis porque cada imagem destaca explicitamente apenas as instâncias A e B.

---

## 9. Construção das relações e dos estímulos

### 9.1 Seleção das relações

Para cada captura:

1. calcularam-se todas as distâncias centro–centro entre os objetos selecionados;
2. relações menores que 0,25 m foram evitadas quando havia quantidade suficiente de alternativas;
3. 12 relações foram escolhidas para cobrir a faixa de distâncias disponível;
4. as relações foram ordenadas por distância;
5. índices ímpares foram atribuídos ao fold A e índices pares ao fold B;
6. cada fold recebeu seis relações distribuídas entre faixas curtas, médias e longas.

Faixas finais de verdade-terreno:

| Captura | Distância mínima | Distância máxima | Relações |
|---|---:|---:|---:|
| 42897545 | 0,266 m | 0,886 m | 12 |
| 45261179 | 0,623 m | 1,900 m | 12 |
| 45261615 | 0,355 m | 1,851 m | 12 |
| 45662921 | 0,285 m | 1,107 m | 12 |
| 47115543 | 0,354 m | 2,126 m | 12 |

### 9.2 Estímulo visual G-oracle

Cada relação gerou uma nova imagem:

- apenas o objeto A recebeu caixa vermelha e rótulo `A`;
- apenas o objeto B recebeu caixa azul e rótulo `B`;
- todas as outras caixas foram removidas;
- o valor da verdade-terreno não apareceu no arquivo nem no prompt.

A inspeção visual das 60 imagens foi concluída antes da rodada final.

### 9.3 Prompt congelado

```text
Object A is marked with a red box and object B with a blue box. Estimate the 3D Euclidean distance between the centers of their 3D bounding boxes, in meters. Return only JSON in this format: {"distance_m": number}
```

O prompt foi mantido idêntico em todas as consultas.

---

## 10. Testes técnicos anteriores à rodada final

### 10.1 Consulta de sanidade

A primeira consulta usou o par P01 da cozinha:

- verdade-terreno: 0,285 m;
- predição: 0,420 m;
- erro absoluto: 0,135 m;
- superestimação relativa: aproximadamente 47,4%;
- pico de memória informado: 4,108 GB;
- velocidade de geração informada: aproximadamente 38,5 tokens/s.

### 10.2 Repetibilidade observada

P01 foi executado novamente no lote da primeira cena e depois na rodada unificada. Nas execuções registradas com temperatura zero, a resposta permaneceu em 0,420 m.

Esse teste confirma a repetibilidade observada nessa configuração e nesse estímulo. Ele não substitui uma análise formal de determinismo em outras plataformas ou versões.

### 10.3 Primeiro resultado de uma cena

Antes do piloto com cinco capturas, a cozinha foi avaliada isoladamente:

- \(G^{\mathrm{CV}}=-0{,}733\);
- RMSE-log bruto: 0,322;
- RMSE-log corrigido: 0,424;
- fração descritiva de escala: 0,201;
- MAE: 0,130 m;
- MAPE: 28,3%.

Esse resultado mostrou que o fator aprendido em um fold podia piorar o outro. Por isso, a hipótese não foi julgada a partir de uma única cena e o piloto foi ampliado.

---

## 11. Execução unificada

### 11.1 Configuração

- capturas: `42897545`, `45261179`, `45261615`, `45662921`, `47115543`;
- consultas: 60;
- respostas analisáveis: 60 de 60;
- ordem: embaralhada com semente `20260814`;
- cada chamada foi independente, sem histórico conversacional;
- o modelo foi carregado uma única vez;
- saída pedida: JSON com a chave `distance_m`.

### 11.2 Tempo

- média por consulta: 4,833 s;
- mediana por consulta: 4,650 s;
- soma dos tempos de geração: 289,967 s;
- intervalo total registrado nos metadados: aproximadamente 4 min 51 s.

### 11.3 Integridade

O hash SHA-256 do manifesto calculado após o upload coincide com o valor registrado durante a execução:

```text
8042a906d9e86d579b165da63ce3973c3581bb997851991336146079034c4698
```

Também foram verificados:

- 60 IDs de pares únicos;
- 12 pares por captura;
- seis relações no fold A e seis no fold B de cada captura;
- correspondência exata da verdade-terreno entre manifesto e resultados;
- 60 predições numéricas positivas.

---

## 12. Métricas e validação cruzada

Para a cena \(s\), o modelo \(m\) e a relação \(k\), define-se:

\[
e_{skm}=\log \hat d_{skm}-\log d_{sk}.
\]

Em um conjunto de ajuste \(A_s\):

\[
\log\alpha_{sm}=-\frac{1}{|A_s|}\sum_{k\in A_s}e_{skm}.
\]

O fator \(\alpha\) multiplica as predições do fold retido. O ganho fora da amostra é:

\[
G_{sm}^{\mathrm{CV}}
=1-
\frac{\sum_{k\in T_s}(e_{skm}+\log\alpha_{sm})^2}
{\sum_{k\in T_s}e_{skm}^2}.
\]

Interpretação operacional:

- \(G^{\mathrm{CV}}>0\): a correção reduz o erro no fold retido;
- \(G^{\mathrm{CV}}=0\): a correção é neutra;
- \(G^{\mathrm{CV}}<0\): a correção piora o fold retido;
- \(G^{\mathrm{CV}}>0{,}5\): mais da metade do erro quadrático em log foi removida.

A decomposição descritiva no mesmo conjunto é:

\[
SS_{escala}=n\bar e^2,
\qquad
SS_{config}=\sum_k(e_k-\bar e)^2.
\]

Essa decomposição é secundária porque reutiliza as mesmas relações para estimar e descrever a escala. A evidência primária vem da validação cruzada.

---

## 13. Resultados por captura

Na tabela, \(\alpha_{B\rightarrow A}\) foi estimado no fold B e aplicado ao fold A; \(\alpha_{A\rightarrow B}\) foi estimado no fold A e aplicado ao fold B.

| Captura | GT (m) | RMSE-log bruto | RMSE-log corrigido | \(G^{CV}\) | \(\alpha_{B\rightarrow A}\) | \(\alpha_{A\rightarrow B}\) | Fração descritiva de escala | MAE | MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42897545 | 0,266–0,886 | 0,176 | 0,123 | **0,510** | 1,125 | 1,144 | 0,517 | 0,083 m | 13,0% |
| 45261179 | 0,623–1,900 | 0,367 | 0,359 | 0,038 | 0,929 | 0,799 | 0,165 | 0,349 m | 36,4% |
| 45261615 | 0,355–1,851 | 0,588 | 0,448 | **0,420** | 0,725 | 0,605 | 0,491 | 0,410 m | 69,3% |
| 45662921 | 0,285–1,107 | 0,322 | 0,424 | **−0,733** | 1,036 | 0,723 | 0,201 | 0,130 m | 28,3% |
| 47115543 | 0,354–2,126 | 0,247 | 0,260 | **−0,108** | 1,053 | 0,959 | 0,0004 | 0,242 m | 22,1% |

### 13.1 Leitura individual

**Captura 42897545.** A correção reduziu o RMSE-log de 0,176 para 0,123 e atingiu \(G^{CV}=0{,}510\). Os dois fatores estimados foram próximos, 1,125 e 1,144. É a evidência mais clara de escala compartilhada neste piloto.

**Captura 45261179.** O ganho combinado foi pequeno, \(G^{CV}=0{,}038\). Um sentido da validação cruzada teve ganho e o outro piorou. A correção foi quase neutra no conjunto da cena.

**Captura 45261615.** O ganho foi relevante, \(G^{CV}=0{,}420\), com redução do RMSE-log de 0,588 para 0,448. A cena tinha erro bruto alto e forte tendência de superestimação, corrigida por fatores menores que 1.

**Captura 45662921.** A correção piorou o resultado: \(G^{CV}=-0{,}733\). Os fatores dos folds, 1,036 e 0,723, apontaram em direções incompatíveis, evidenciando instabilidade entre as relações de ajuste.

**Captura 47115543.** O viés médio em log foi praticamente zero, mas os erros relacionais permaneceram. A fração descritiva de escala foi 0,0004 e o ganho cruzado foi negativo. Essa cena representa um caso de erro predominantemente configuracional.

---

## 14. Resultado agregado do piloto

O protocolo determina que as métricas sejam calculadas primeiro dentro da captura e depois agregadas com peso igual por captura.

| Métrica | Resultado |
|---|---:|
| Capturas | 5 |
| Relações | 60 |
| Média de \(G^{CV}\) por captura, endpoint primário | **0,025** |
| Mediana de \(G^{CV}\) | 0,038 |
| Mínimo / máximo de \(G^{CV}\) | −0,733 / 0,510 |
| Capturas com ganho positivo | 3 de 5 |
| Capturas com \(G^{CV}>0{,}5\) | 1 de 5 |
| \(G^{CV}\) calculado pelo SSE agrupado, secundário | 0,124 |
| RMSE-log agrupado bruto | 0,367 |
| RMSE-log agrupado corrigido | 0,344 |
| Fração descritiva média de escala | 0,275 |
| Fração descritiva mediana de escala | 0,201 |
| MAE médio por captura | 0,243 m |
| MAPE médio por captura | 33,8% |

O valor agrupado de 0,124 dá mais influência às cenas com maior soma de erro bruto. Por isso, ele é secundário. A média por captura, 0,025, segue a regra pré-especificada de peso igual entre cenas.

Não foi calculado intervalo de confiança para sustentar uma alegação populacional. Com cinco capturas, um bootstrap produziria uma descrição instável e não substituiria o piloto planejado com 50 cenas.

---

## 15. Padrão das respostas numéricas

O modelo produziu apenas 18 valores distintos nas 60 respostas. O valor `1.32` apareceu 15 vezes, ou 25% do total. Os cinco valores mais frequentes, `1.32`, `0.72`, `0.52`, `1.22` e `0.42`, somaram 40 respostas, ou 66,7% do total.

[Inferência] Essa concentração sugere que o modelo responde por níveis numéricos preferenciais em vez de formar uma estimativa contínua fina. A causa não foi identificada. Pode envolver priors aprendidos, arredondamento implícito, formato de resposta ou comportamento da versão quantizada. Um estudo de unidade, precisão pedida e modelo deve ser planejado como análise de robustez, sem substituir o prompt primário depois de observar os resultados.

---

## 16. Interpretação científica permitida neste estágio

### 16.1 O que os dados mostram diretamente

- a correção de escala pode generalizar entre relações em algumas cenas;
- essa generalização varia fortemente entre capturas;
- somente uma das cinco cenas ultrapassou o critério \(G^{CV}>0{,}5\);
- duas cenas tiveram ganho negativo;
- a média primária do piloto ficou próxima de zero;
- a componente descritiva de escala respondeu, em média, por menos de um terço do erro;
- 60 respostas foram obtidas e analisadas sem falha de parsing.

### 16.2 Interpretação provisória

[Inferência] Para o Qwen3-VL-4B-Instruct quantizado em 4 bits, a escala compartilhada parece ser uma propriedade dependente da cena, não um mecanismo estável presente em todas as imagens.

[Inferência] O resultado agregado atual favorece a continuação da linha como diagnóstico estrutural, mas não sustenta a formulação forte de que a escala domina o erro métrico de VLMs.

[Inferência] A diferença entre as cenas 42897545 e 47115543 mostra por que um ajuste retrospectivo em todas as relações seria enganoso: uma cena contém viés global transferível, enquanto a outra tem média praticamente correta e configuração residual.

### 16.3 O que permanece em aberto

- efeito do tamanho e da família do modelo;
- efeito da quantização;
- comportamento em G-natural;
- estabilidade em 50 capturas e três modelos;
- efeito de distância, categoria, oclusão e tamanho de caixa;
- dependência de focal, intrínsecos e enquadramento;
- efeito causal de âncoras físicas;
- generalização para outros datasets.

---

## 17. Estado dos portões

| Portão | Estado no checkpoint | Evidência |
|---|---|---|
| G0 — novidade | Aberto | posicionamento atualizado, mas auditoria sistemática ainda incompleta |
| G1 — dados | Passou para o piloto | anotações 3D consistentes, múltiplas relações e grounding verificável |
| G1b — identificabilidade | Evidência mista; decisão pendente | média \(G^{CV}=0{,}025\), três cenas positivas, duas negativas |
| G2 — componente dominante | Não atingido neste piloto | apenas uma cena com \(G^{CV}>0{,}5\); fração média de escala 0,275 |
| G3 — câmera | Não avaliado | Fase 3 |
| G4 — intervenção semântica | Não avaliado | Fase 3 |
| G5 — método | Não avaliado | M0/M1 ainda não implementados |
| G6 — utilidade do VLM | Não avaliado | requer baselines geométricos equivalentes |
| G7 — baseline geométrico | Não avaliado | comparação B3-oracle/B3-E2E pendente |

G1b e G2 não devem ser fechados definitivamente antes do piloto planejado de 50 cenas e três modelos.

---

## 18. Problemas técnicos encontrados e resposta adotada

### 18.1 Ambiente virtual inativo

Uma tentativa foi executada a partir de `/`, fora do diretório do projeto e sem `.venv`. O erro foi:

```text
ModuleNotFoundError: No module named 'mlx_vlm'
```

A sessão voltou para `~/scale-vs-configuration`, o ambiente foi reativado e o import foi verificado antes de continuar. Nenhum resultado parcial foi produzido nessa tentativa.

### 18.2 Erro sintático em script de anotação

Uma versão extensa do script apresentou `SyntaxError: f-string: unterminated string`. O script foi substituído por uma versão menor, com formatação `%`, e as quatro cenas foram anotadas corretamente.

### 18.3 Snapshot incompleto segundo o Hugging Face Hub

`snapshot_download(local_files_only=True)` recusou o cache por ausência de `.gitattributes` e `README.md`. Os arquivos necessários à inferência já estavam presentes. O carregamento passou a usar diretamente o diretório local do commit `2fd8dac...`, preservando a revisão exata sem baixar arquivos adicionais.

### 18.4 Quantidade não equivale a qualidade visual

O primeiro frame escolhido pelo número de candidatos tinha movimento e oclusão. A seleção seguinte passou a combinar quantidade, objetos internos e nitidez, seguida de inspeção manual.

### 18.5 Dívida de reprodutibilidade

Grande parte da preparação foi executada por blocos Python diretamente no terminal. Os resultados e manifestos foram preservados, mas os scripts de extração, auditoria, seleção, geração de pares e execução ainda precisam ser materializados como arquivos versionados antes do commit Git do checkpoint.

---

## 19. Estrutura de arquivos produzida

Estrutura lógica no computador de execução:

```text
scale-vs-configuration/
├── .venv/
├── .cache/huggingface/
├── ml-cubifyanything/
├── data/ca1m/
│   ├── ca1m-val-45662921.tar
│   ├── ca1m-val-45261179.tar
│   ├── ca1m-val-42897545.tar
│   ├── ca1m-val-45261615.tar
│   └── ca1m-val-47115543.tar
└── outputs/pilot/
    ├── oracle_pairs/
    ├── raw/
    ├── analysis/
    ├── capture_candidates/
    ├── selected_scenes/
    ├── multiscene_oracle/
    └── pilot5_oracle/
        ├── manifest.jsonl
        ├── metadata.json
        └── results.jsonl
```

Arquivos que devem permanecer fora do Git:

- `.venv/`;
- `.cache/`;
- TARs e imagens do CA-1M;
- pesos do modelo;
- estímulos contendo imagens do dataset, até revisão da licença.

Arquivos apropriados para versionamento privado e, quando permitido, público:

- scripts;
- manifesto com IDs e hashes;
- protocolo;
- metadados;
- resultados numéricos;
- métricas agregadas;
- documentação.

---

## 20. Integridade do pacote deste checkpoint

| Arquivo | SHA-256 |
|---|---|
| `manifest.jsonl` | `8042a906d9e86d579b165da63ce3973c3581bb997851991336146079034c4698` |
| `metadata.json` | `59f885b6c62e619e7a815dbc07e8b185a98a52a3224df26ed002194608ee0c88` |
| `protocol.json` | `43b88f65e4c305ecbc787421480e8c345fb3384a2c7092fd2a5a7fe540fa264d` |
| `qwen3vl4b_oracle_metrics.json` | `5749cb27dc457f16777e384dc5b86fd25a291271ef15079b0b8627ed06fb0d50` |
| `results.jsonl` | `088fa9d281f7520b4380f64c5f9ea0391078ce57fc89f7be8d256341de35c897` |

O `protocol.json` do pacote descreve as quatro capturas adicionadas depois da cozinha. O `manifest.jsonl` unificado é a fonte autoritativa para as cinco capturas e 60 relações.

---

## 21. Próximos passos recomendados

### Passo imediato: fechar o checkpoint técnico

1. transformar os blocos executados no terminal em scripts sob `scripts/`;
2. criar `README.md` com comandos reproduzíveis;
3. criar `.gitignore` para dados, cache, ambiente e imagens;
4. registrar as dependências com versões fixas;
5. versionar protocolo, manifestos, resultados e este documento;
6. criar a tag `v0.1-piloto-oracle` em um repositório próprio, separado do clone da Apple.

### Continuação da Fase 1

1. ampliar de 5 para 50 capturas;
2. manter 10–15 relações por captura;
3. avaliar três modelos representativos;
4. manter G-oracle como primeira condição;
5. congelar os filtros de visibilidade e a regra de seleção de pares antes da nova rodada;
6. agregar por captura, sem tratar frames adjacentes como unidades independentes;
7. calcular incerteza por bootstrap de captura somente com amostra maior;
8. decidir G1b e G2 depois da matriz de 50 cenas por três modelos.

### Depois do piloto ampliado

1. construir G-natural usando descrições não ambíguas;
2. comparar G-natural com G-oracle;
3. executar modelos maiores ou em precisão distinta;
4. testar um baseline geométrico com profundidade;
5. avançar para focal e âncoras somente depois de caracterizar a decomposição.

O treinamento do VLM fatorado M1 não é o próximo passo. Os dados atuais indicam que primeiro é necessário entender em quais cenas e modelos a escala é realmente transferível.

---

## 22. Critério de decisão após a Fase 1 completa

Três resultados continuam possíveis:

1. **Escala coerente e dominante:** a maioria das cenas e modelos apresenta ganho positivo alto e o critério de dominância é atingido.
2. **Escala coerente, mas não dominante:** a correção ajuda de forma estável, porém a configuração residual continua importante.
3. **Configuração dominante ou dependência forte da cena:** a correção não generaliza consistentemente, e a principal contribuição passa a ser o diagnóstico que impede confundir calibração com compreensão geométrica.

O checkpoint atual se aproxima provisoriamente do terceiro cenário no agregado, com evidência localizada do segundo e do primeiro em duas capturas. Essa classificação é provisória porque a amostra e o painel de modelos ainda estão abaixo do plano da Fase 1.

---

## Apêndice A — As 60 relações e respostas

| Cena | Par | Fold | Faixa | A | B | GT (m) | Pred. (m) | Erro abs. (m) |
|---|---:|:---:|:---:|---|---|---:|---:|---:|
| 42897545 | P01 | A | curta | 09 tissue box | 10 watering can | 0,266 | 0,220 | 0,046 |
| 42897545 | P02 | B | curta | 02 sink | 10 watering can | 0,402 | 0,420 | 0,018 |
| 42897545 | P03 | A | curta | 01 cabinet | 02 sink | 0,442 | 0,420 | 0,022 |
| 42897545 | P04 | B | curta | 07 toilet paper roll | 10 watering can | 0,464 | 0,320 | 0,144 |
| 42897545 | P05 | A | média | 01 cabinet | 10 watering can | 0,480 | 0,480 | 0,000 |
| 42897545 | P06 | B | média | 01 cabinet | 06 mixer tap | 0,583 | 0,620 | 0,037 |
| 42897545 | P07 | A | média | 06 mixer tap | 09 tissue box | 0,674 | 0,520 | 0,154 |
| 42897545 | P08 | B | média | 01 cabinet | 03 countertop | 0,688 | 0,620 | 0,068 |
| 42897545 | P09 | A | longa | 02 sink | 07 toilet paper roll | 0,813 | 0,720 | 0,093 |
| 42897545 | P10 | B | longa | 02 sink | 03 countertop | 0,820 | 0,720 | 0,100 |
| 42897545 | P11 | A | longa | 03 countertop | 06 mixer tap | 0,869 | 0,720 | 0,149 |
| 42897545 | P12 | B | longa | 03 countertop | 05 spray bottle | 0,886 | 0,720 | 0,166 |
| 45261179 | P01 | A | curta | 06 recess light | 07 object | 0,623 | 1,320 | 0,697 |
| 45261179 | P02 | B | curta | 01 exhaust fan | 06 recess light | 0,693 | 1,220 | 0,527 |
| 45261179 | P03 | A | curta | 06 recess light | 08 switch board | 0,794 | 1,320 | 0,526 |
| 45261179 | P04 | B | curta | 05 mirror | 07 object | 0,979 | 1,320 | 0,341 |
| 45261179 | P05 | A | média | 02 toothbrush holder | 05 mirror | 1,020 | 1,320 | 0,300 |
| 45261179 | P06 | B | média | 05 mirror | 06 recess light | 1,028 | 1,320 | 0,292 |
| 45261179 | P07 | A | média | 02 toothbrush holder | 07 object | 1,216 | 1,320 | 0,104 |
| 45261179 | P08 | B | média | 01 exhaust fan | 08 switch board | 1,459 | 1,320 | 0,139 |
| 45261179 | P09 | A | longa | 01 exhaust fan | 11 toothpaste | 1,533 | 1,520 | 0,013 |
| 45261179 | P10 | B | longa | 02 toothbrush holder | 06 recess light | 1,626 | 1,320 | 0,306 |
| 45261179 | P11 | A | longa | 06 recess light | 11 toothpaste | 1,687 | 1,320 | 0,367 |
| 45261179 | P12 | B | longa | 08 switch board | 11 toothpaste | 1,900 | 1,320 | 0,580 |
| 45261615 | P01 | A | curta | 15 cushion | 16 cushion | 0,355 | 1,220 | 0,865 |
| 45261615 | P02 | B | curta | 12 tray | 13 cushion | 0,419 | 1,320 | 0,901 |
| 45261615 | P03 | A | curta | 08 bottle | 12 tray | 0,588 | 1,220 | 0,632 |
| 45261615 | P04 | B | curta | 07 flower vase | 08 bottle | 0,596 | 0,820 | 0,224 |
| 45261615 | P05 | A | média | 07 flower vase | 12 tray | 0,735 | 1,220 | 0,485 |
| 45261615 | P06 | B | média | 14 cushion | 16 cushion | 0,755 | 0,920 | 0,165 |
| 45261615 | P07 | A | média | 12 tray | 15 cushion | 0,953 | 1,420 | 0,467 |
| 45261615 | P08 | B | média | 08 bottle | 14 cushion | 1,077 | 1,570 | 0,493 |
| 45261615 | P09 | A | longa | 09 toy | 15 cushion | 1,259 | 1,470 | 0,211 |
| 45261615 | P10 | B | longa | 07 flower vase | 14 cushion | 1,349 | 1,470 | 0,121 |
| 45261615 | P11 | A | longa | 07 flower vase | 15 cushion | 1,592 | 1,570 | 0,022 |
| 45261615 | P12 | B | longa | 07 flower vase | 16 cushion | 1,851 | 1,520 | 0,331 |
| 45662921 | P01 | A | curta | 12 bowl | 14 chopping board | 0,285 | 0,420 | 0,135 |
| 45662921 | P02 | B | curta | 13 glove | 16 utensil holder | 0,288 | 0,320 | 0,032 |
| 45662921 | P03 | A | curta | 05 sink | 19 faucet | 0,312 | 0,720 | 0,408 |
| 45662921 | P04 | B | curta | 16 utensil holder | 19 faucet | 0,345 | 0,420 | 0,075 |
| 45662921 | P05 | A | média | 15 plate holder | 19 faucet | 0,403 | 0,520 | 0,117 |
| 45662921 | P06 | B | média | 12 bowl | 16 utensil holder | 0,425 | 0,420 | 0,005 |
| 45662921 | P07 | A | média | 04 mug | 05 sink | 0,428 | 0,520 | 0,092 |
| 45662921 | P08 | B | média | 05 sink | 12 bowl | 0,573 | 0,520 | 0,053 |
| 45662921 | P09 | A | longa | 01 cabinet | 14 chopping board | 0,758 | 0,820 | 0,062 |
| 45662921 | P10 | B | longa | 01 cabinet | 19 faucet | 0,800 | 0,820 | 0,020 |
| 45662921 | P11 | A | longa | 01 cabinet | 04 mug | 0,842 | 1,020 | 0,178 |
| 45662921 | P12 | B | longa | 01 cabinet | 05 sink | 1,107 | 0,720 | 0,387 |
| 47115543 | P01 | A | curta | 06 laptop | 08 desk | 0,354 | 0,520 | 0,166 |
| 47115543 | P02 | B | curta | 05 television | 06 laptop | 0,368 | 0,520 | 0,152 |
| 47115543 | P03 | A | curta | 05 television | 07 chair | 0,582 | 0,720 | 0,138 |
| 47115543 | P04 | B | curta | 08 desk | 16 trash can | 0,691 | 0,720 | 0,029 |
| 47115543 | P05 | A | média | 05 television | 16 trash can | 0,954 | 1,220 | 0,266 |
| 47115543 | P06 | B | média | 04 light fixture | 08 desk | 1,263 | 1,320 | 0,057 |
| 47115543 | P07 | A | média | 08 desk | 11 mirror | 1,370 | 1,220 | 0,150 |
| 47115543 | P08 | B | média | 06 laptop | 09 window | 1,473 | 1,320 | 0,153 |
| 47115543 | P09 | A | longa | 05 television | 09 window | 1,554 | 1,320 | 0,234 |
| 47115543 | P10 | B | longa | 08 desk | 09 window | 1,730 | 1,320 | 0,410 |
| 47115543 | P11 | A | longa | 09 window | 16 trash can | 1,871 | 1,370 | 0,501 |
| 47115543 | P12 | B | longa | 09 window | 11 mirror | 2,126 | 1,480 | 0,646 |

---

## Apêndice B — Referências e artefatos relacionados

- plano atualizado: `plano_scale_vs_configuration_atualizado.md`;
- repositório e dados: [apple/ml-cubifyanything](https://github.com/apple/ml-cubifyanything);
- modelo-base: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct);
- modelo MLX: [mlx-community/Qwen3-VL-4B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit);
- mecanismo de inferência: [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm);
- entradas preservadas: `checkpoint_v0.1_inputs.zip`;
- saída recomendada deste checkpoint: `checkpoint_v0.1_piloto_oracle.md`.
