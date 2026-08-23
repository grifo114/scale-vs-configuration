# Checkpoint v0.4 — progresso confirmatório após 10 capturas

Este checkpoint congela o estado intermediário da construção do conjunto
confirmatório após a inspeção das dez primeiras capturas da ordem
predefinida.

Ele não representa o conjunto confirmatório completo nem contém resultados
de modelos. O alvo permanece em 50 cenas aceitas.

## Estado congelado

- capturas inspecionadas: 10;
- capturas aceitas: 6;
- capturas rejeitadas: 4;
- taxa observada de aceitação: 60%;
- relações selecionadas: 72;
- relações por cena aceita: 12;
- próxima captura: `42899459`;
- SHA-256 do log: `9bcdebfe8b39c553b22511aa975562af247544906b8c94f63950f797a6e12b29`;
- commit de origem: `73c11e7f92b74ae663a294640847402f5541e4fa`;
- branch de origem: `phase1/confirmatory-dataset`.

## Capturas aceitas

| Captura | Quadro | Pares estritos | Intervalo selecionado (m) | Razão | Objetos | Máx. usos |
|---|---:|---:|---:|---:|---:|---:|
| 42897561 | 91 | 19 | 0.483–1.434 | 2.970 | 7 | 5 |
| 47333898 | 278 | 100 | 0.290–1.579 | 5.453 | 8 | 5 |
| 42899679 | 848 | 43 | 0.534–6.236 | 11.687 | 8 | 5 |
| 43896330 | 1794 | 26 | 1.026–6.447 | 6.287 | 8 | 4 |
| 45260854 | 193 | 26 | 0.372–3.755 | 10.101 | 7 | 5 |
| 47334107 | 467 | 15 | 1.044–4.417 | 4.231 | 6 | 5 |

## Capturas rejeitadas

| Captura | Melhor suporte estrito | Motivo |
|---|---:|---|
| 42899698 | 7 | minimum_eligible_pairs |
| 45662942 | 6 | minimum_eligible_pairs |
| 48018382 | 0 | no_frame_candidates |
| 45662970 | 6 | minimum_eligible_pairs |

## Conteúdo

| Arquivo ou diretório | Finalidade |
|---|---|
| `summary.json` | resumo estruturado do checkpoint |
| `confirmatory_capture_log.jsonl` | decisões sequenciais e hashes dos artefatos |
| `accepted_pairs_manifest.jsonl` | 72 relações das seis cenas aceitas |
| `rendered_manifest.jsonl` | registros dos estímulos renderizados |
| `confirmatory_capture_protocol.json` | protocolo confirmatório congelado |
| `confirmatory_capture_order.csv` | ordem determinística das capturas |
| `confirmatory_capture_order_metadata.json` | metadados e hash da ordem |
| `development_capture_ids.txt` | capturas excluídas por uso no desenvolvimento |
| `selection_protocols/` | decisão numérica das dez capturas |
| `validation_reports/` | validação dos manifestos aceitos |
| `render_protocols/` | parâmetros de renderização |
| `contact_sheets/` | inspeção visual das seis cenas aceitas |
| `SHA256SUMS` | integridade dos arquivos do checkpoint |

## Escopo

Os relatórios de validação armazenados neste checkpoint tiveram apenas o
prefixo absoluto do diretório local removido. Os valores científicos não
foram alterados; os hashes dos relatórios originais permanecem registrados
em `confirmatory_capture_log.jsonl`.

Os TARs do CA-1M, as auditorias completas e os arquivos extensos de
viabilidade permanecem fora do Git. Seus caminhos, tamanhos e hashes estão
registrados em `confirmatory_capture_log.jsonl`.

A seleção deve continuar na ordem congelada, sem alteração dos limiares
durante a etapa confirmatória.
