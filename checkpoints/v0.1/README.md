# Checkpoint v0.1

Piloto técnico multiscena da condição G-oracle.

## Arquivos

- `manifest.jsonl`: manifesto unificado e autoritativo das 60 relações;
- `metadata.json`: ambiente, modelo, revisão e parâmetros da execução;
- `results.jsonl`: respostas brutas e predições das 60 consultas;
- `metrics_recomputed.json`: métricas recalculadas pelo script versionado;
- `scene_metrics.csv`: tabela resumida por captura;
- `protocol.json`: protocolo das quatro capturas adicionadas depois da cena inicial;
- `qwen3vl4b_oracle_metrics.json`: análise anterior da captura inicial 45662921.

O `protocol.json` não descreve sozinho as cinco capturas. Para a rodada
unificada, a fonte autoritativa é `manifest.jsonl`.

## Reprodução

A partir da raiz do projeto:

```bash
python3 scripts/analyze_oracle.py
```

## Integridade das entradas originais

```text
manifest.jsonl
8042a906d9e86d579b165da63ce3973c3581bb997851991336146079034c4698

metadata.json
59f885b6c62e619e7a815dbc07e8b185a98a52a3224df26ed002194608ee0c88

protocol.json
43b88f65e4c305ecbc787421480e8c345fb3384a2c7092fd2a5a7fe540fa264d

qwen3vl4b_oracle_metrics.json
5749cb27dc457f16777e384dc5b86fd25a291271ef15079b0b8627ed06fb0d50

results.jsonl
088fa9d281f7520b4380f64c5f9ea0391078ce57fc89f7be8d256341de35c897
```
