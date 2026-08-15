# Checkpoint técnico v0.3 — protocolo de seleção de cenas e relações

**Projeto:** Scale vs. Configuration  
**Data:** 15 de agosto de 2026  
**Branch de desenvolvimento:** `phase1/scene-selection-protocol`  
**Estado:** protocolo de desenvolvimento validado e pronto para congelamento antes do estudo confirmatório

## 1. Finalidade

Este checkpoint registra o protocolo usado para transformar capturas CA-1M em
estímulos de medição métrica com dois objetos marcados por caixas vermelha e
azul.

A etapa foi realizada antes da execução confirmatória dos modelos. Seu objetivo
é separar a escolha das cenas e relações dos resultados posteriores de
inferência.

## 2. Escopo do conjunto de desenvolvimento

Foram usadas cinco capturas CA-1M exclusivamente para desenvolver e auditar o
protocolo:

| Captura | Quadros auditados |
|---|---:|
| 42897545 | 335 |
| 45261179 | 280 |
| 45261615 | 518 |
| 45662921 | 1.135 |
| 47115543 | 700 |
| **Total** | **2.968** |

Essas cinco capturas devem permanecer fora do conjunto confirmatório.

## 3. Reconstrução da auditoria histórica

As regras da auditoria exploratória foram reconstruídas a partir dos arquivos
CA-1M e comparadas com os 1.833 registros históricos disponíveis.

Configuração reconstruída:

- caixas `box_2d_rend`;
- validade geométrica das instâncias;
- área de caixa de pelo menos 1.000 px²;
- largura e altura de pelo menos 20 px;
- cobertura máxima de 0,5 da imagem;
- categorias genéricas excluídas: `baseboard`, `ceiling`, `floor`,
  `object` e `wall`;
- categorias naturais únicas no quadro;
- legenda ignorada como requisito;
- objeto interior quando toda a caixa está a pelo menos 5 px da borda;
- nitidez calculada por Pillow RGB, conversão `cv2.COLOR_RGB2GRAY` e variância
  do Laplaciano em `CV_64F`.

Resultado da reconstrução: **1.833 de 1.833 quadros sem divergência**.

A quinta captura, 45662921, foi reprocessada com as mesmas regras, produzindo
1.135 registros adicionais.

## 4. Geração dos candidatos

A busca exaustiva considerou os 2.968 quadros das cinco capturas de
desenvolvimento.

- quadros candidatos após a regra de cena: 547;
- configurações de legibilidade avaliadas: 64;
- capturas capazes de fornecer 12 pares sob a configuração escolhida: 4 de 5;
- captura rejeitada: 45261615.

A configuração selecionada no conjunto de desenvolvimento foi:

| Critério | Valor |
|---|---:|
| distância 3D mínima | 0,25 m |
| fração visível mínima | 1,0 |
| IoU máximo entre caixas | 0,50 |
| interseção sobre a menor caixa | 0,80 |
| área mínima de cada caixa | 0,00375 da imagem |
| menor lado da caixa | 30 px |
| margem mínima da borda | 20 px |
| pares necessários por cena | 12 |

Nenhuma das 64 configurações forneceu 12 pares para todas as cinco capturas.
A configuração escolhida obteve cobertura máxima de quatro capturas e aplicou
os critérios mais estritos encontrados entre as soluções de melhor cobertura.

## 5. Seleção das relações

Para cada captura aceita:

1. escolhe-se o quadro com o maior número de pares que atendem aos filtros;
2. empates são resolvidos por posição no ranking e índice do quadro;
3. os pares são ordenados pela distância 3D;
4. a sequência é dividida em curta, média e longa;
5. são escolhidos quatro pares por faixa;
6. índices ímpares pertencem ao fold A e índices pares ao fold B.

O validador também exige:

- seis objetos diferentes ou mais por cena;
- no máximo cinco usos do mesmo objeto;
- razão entre maior e menor distância de pelo menos 2,5;
- relações não duplicadas;
- dois pares de cada fold dentro de cada faixa.

O limiar de razão 2,5 foi definido durante o desenvolvimento. Ele deve
permanecer congelado antes da avaliação confirmatória.

## 6. Resultado final

| Captura | Rank | Quadro | Pares | Distância | Razão | Objetos | Uso máximo |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42897545 | 2 | 100 | 12 | 0,266–0,886 m | 3,332 | 6 | 5 |
| 45261179 | 1 | 31 | 12 | 0,378–1,140 m | 3,013 | 6 | 5 |
| 45662921 | 2 | 946 | 12 | 0,288–0,775 m | 2,695 | 9 | 5 |
| 47115543 | 4 | 86 | 12 | 0,364–2,627 m | 7,217 | 7 | 4 |

Resultado agregado:

- cenas aceitas: 4;
- cenas rejeitadas: 1;
- relações finais: 48;
- relações por cena: 12;
- relações por faixa e cena: 4;
- relações por fold e cena: 6;
- respostas de modelo usadas nesta seleção: nenhuma.

SHA-256 do manifesto final:

```text
8347d700cf8874b511ca266644c8f5bca224b821c6e745b38dc8e61add610c44
```

## 7. Auditoria visual

A folha de contato final foi inspecionada após a seleção automática.

Foram verificados:

- presença das duas caixas em cada estímulo;
- legibilidade dos alvos;
- ausência de alvos cortados pela borda;
- ausência de caixas muito pequenas;
- identificação visual suficiente das regiões marcadas;
- correspondência entre a relação e a cena usada.

As quatro cenas selecionadas foram mantidas. A inspeção visual não modificou
pares individuais após a validação automática.

## 8. Validação automática

Comando:

```bash
python3 scripts/validate_pair_manifest.py \
  checkpoints/v0.3/pair_selection_manifest.jsonl \
  --protocol checkpoints/v0.3/pair_selection_protocol.json
```

Resultado esperado:

```text
Scenes: 4
Pairs: 48
Result: PASSED
```

O validador verifica estrutura, balanceamento, diversidade, amplitude métrica,
legibilidade, sobreposição e integridade do protocolo.

## 9. Artefatos congelados

| Arquivo | Conteúdo |
|---|---|
| `audit_reconstructed_metadata.json` | metadados da reconstrução de 1.833 quadros |
| `audit_45662921_metadata.json` | metadados da auditoria da quinta captura |
| `audit_rule_reconstruction.txt` | comparação das regras reconstruídas |
| `candidate_ranking_protocol.json` | regras da busca exaustiva de quadros |
| `pair_feasibility_protocol.json` | regras de construção e avaliação dos pares |
| `legibility_sweep_summary.json` | resumo do sweep e hash do arquivo completo |
| `pair_selection_manifest.jsonl` | 48 relações finais |
| `pair_selection_protocol.json` | parâmetros e decisões da seleção |
| `rendered_manifest.jsonl` | estímulos renderizados |
| `render_protocol.json` | parâmetros de renderização |
| `contact_sheet_all.png` | auditoria visual das 48 relações |
| `validation_report.txt` | saída do validador automático |
| `SHA256SUMS` | hashes dos artefatos do checkpoint |

Os arquivos intermediários extensos permanecem fora do Git e podem ser
reconstruídos pelos scripts versionados.

## 10. Limitações deste checkpoint

- O conjunto contém apenas cinco capturas de desenvolvimento.
- Uma das cinco capturas foi rejeitada pelo protocolo estrito.
- A inspeção visual foi realizada durante o desenvolvimento e não constitui
  avaliação cega independente.
- A taxa de aprovação nas 102 capturas restantes ainda não foi medida.
- O checkpoint não contém resultados confirmatórios de VLMs.
- Os limiares não devem ser reinterpretados como valores universais para outros
  datasets ou resoluções.

## 11. Próxima etapa

A próxima etapa é aplicar exatamente o protocolo congelado às capturas CA-1M
restantes, mantendo as cinco capturas de desenvolvimento excluídas.

A ordem correta é:

1. auditar novas capturas;
2. ranquear candidatos sem observar saídas dos modelos;
3. avaliar a viabilidade dos pares;
4. selecionar cenas e relações com os limiares congelados;
5. registrar rejeições;
6. congelar o conjunto confirmatório;
7. somente então executar os modelos.

Qualquer fallback necessário por insuficiência de cenas deve ser definido e
registrado antes da primeira inferência confirmatória.
