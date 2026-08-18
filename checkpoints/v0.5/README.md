# Checkpoint v0.5 - Final Confirmatory Dataset

This checkpoint freezes the final outcome of the confirmatory selection after
all 102 captures in the predefined order were
inspected.

The pool was exhausted with 49 accepted scenes,
one fewer than the target of 50. Under the
frozen stopping rule, the thresholds were not relaxed and no capture outside
the pool was used as a replacement.

## Final Outcome

- captures inspected: 102;
- captures accepted: 49;
- numerical rejections: 51;
- visual rejections: 2;
- selected relations: 588;
- final acceptance rate: 48.039%;
- visually rejected captures: `47115525`, `47204605`;
- official log SHA-256: `ed8957558dd677a885e2ab753527974050e64f583098562969b8ca56bf1bb59c`;
- source commit: `93dbeabce0a1a5ecc1c6de79d8c761c1f68f17e2`;
- source branch: `phase1/confirmatory-dataset`.

## Contents

| File or directory | Purpose |
|---|---|
| `summary.json` | structured summary of the final outcome |
| `confirmatory_capture_log.jsonl` | 102 sequential decisions and artifact hashes |
| `accepted_pairs_manifest.jsonl` | 588 relations from the 49 accepted scenes |
| `rendered_manifest.jsonl` | rendered manifests for the accepted scenes |
| `confirmatory_capture_protocol.json` | frozen confirmatory protocol |
| `confirmatory_visual_qc_amendment_v1.json` | predeclared rule for visual-QC failures |
| `confirmatory_capture_order.csv` | complete deterministic capture order |
| `confirmatory_capture_order_metadata.json` | capture-order metadata and hash |
| `development_capture_ids.txt` | captures excluded because they were used in development |
| `selection_protocols/` | numerical decisions for all 102 captures |
| `validation_reports/` | validation reports for the 49 accepted scenes |
| `render_protocols/` | rendering parameters for the accepted scenes |
| `contact_sheets/` | visual evidence for the 49 accepted scenes |
| `visual_rejections/` | pairs, renderings, and contact sheets for the two visual-QC failures |
| `SHA256SUMS` | integrity checksums for every checkpoint file |

## Scope and Portability

The CA-1M TAR archives, full frame audits, candidate lists, and large
pair-feasibility files remain outside Git. Their paths, sizes, and hashes are
preserved in the official log.

Only the absolute local project-root prefix was removed from the validation
reports copied into this checkpoint. The scientific values were not changed,
and the hashes of the original reports remain in the log.

This checkpoint contains no model results and does not represent a 50-scene
sample. The final confirmatory dataset contains 49 scenes and 588 relations.
