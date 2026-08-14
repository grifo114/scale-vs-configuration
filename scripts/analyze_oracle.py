#!/usr/bin/env python3
"""Recalcula as metricas do piloto G-oracle por cena.

O endpoint primario e calculado dentro de cada cena por validacao cruzada
entre os folds A e B. A agregacao final atribui o mesmo peso a cada cena.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any


DEFAULT_CHECKPOINT = Path("checkpoints/v0.1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa manifest.jsonl e results.jsonl do piloto G-oracle."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Diretorio com manifest.jsonl, results.jsonl e metadata.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Saida JSON. Padrao: <checkpoint-dir>/metrics_recomputed.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Saida CSV por cena. Padrao: <checkpoint-dir>/scene_metrics.csv",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERRO: {message}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"arquivo nao encontrado: {path}")
    except json.JSONDecodeError as exc:
        fail(f"JSON invalido em {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"a raiz de {path} precisa ser um objeto JSON")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        fail(f"arquivo nao encontrado: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"JSONL invalido em {path}, linha {line_number}: {exc}")
        if not isinstance(value, dict):
            fail(f"a linha {line_number} de {path} nao e um objeto JSON")
        rows.append(value)
    if not rows:
        fail(f"arquivo sem registros: {path}")
    return rows


def require_number(row: dict[str, Any], field: str, pair_id: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{pair_id}: campo {field!r} nao e numerico")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{pair_id}: campo {field!r} nao e finito")
    return number


def fit_log_scale(rows: list[dict[str, Any]]) -> tuple[float, float]:
    log_alpha = -fmean(float(row["_log_error"]) for row in rows)
    return log_alpha, math.exp(log_alpha)


def compute_gain(raw_sse: float, corrected_sse: float, label: str) -> float:
    if raw_sse <= 0.0:
        fail(f"G_cv indefinido porque o erro bruto e zero em {label}")
    return 1.0 - corrected_sse / raw_sse


def validate_inputs(
    manifest: list[dict[str, Any]],
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    manifest_path: Path,
) -> str:
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    recorded_hash = metadata.get("manifest_sha256")
    if recorded_hash and manifest_hash != recorded_hash:
        fail(
            "hash do manifesto diverge dos metadados: "
            f"calculado={manifest_hash}, registrado={recorded_hash}"
        )

    manifest_by_pair: dict[str, dict[str, Any]] = {}
    for row in manifest:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            fail("registro do manifesto sem pair_id")
        if pair_id in manifest_by_pair:
            fail(f"pair_id duplicado no manifesto: {pair_id}")
        manifest_by_pair[pair_id] = row

    result_ids = [str(row.get("pair_id", "")) for row in results]
    if any(not pair_id for pair_id in result_ids):
        fail("registro de resultado sem pair_id")
    if len(result_ids) != len(set(result_ids)):
        fail("ha pair_id duplicado nos resultados")
    if set(result_ids) != set(manifest_by_pair):
        missing = sorted(set(manifest_by_pair) - set(result_ids))
        extra = sorted(set(result_ids) - set(manifest_by_pair))
        fail(f"manifesto e resultados divergem; ausentes={missing}, extras={extra}")

    expected_queries = metadata.get("number_of_queries")
    if expected_queries is not None and int(expected_queries) != len(results):
        fail(
            f"metadata.number_of_queries={expected_queries}, "
            f"mas foram encontrados {len(results)} resultados"
        )
    valid_responses = metadata.get("valid_responses")
    if valid_responses is not None and int(valid_responses) != len(results):
        fail(
            f"metadata.valid_responses={valid_responses}, "
            f"mas foram encontrados {len(results)} resultados"
        )

    for row in results:
        pair_id = str(row["pair_id"])
        expected = manifest_by_pair[pair_id]
        ground_truth = require_number(row, "ground_truth_m", pair_id)
        expected_gt = require_number(expected, "ground_truth_m", pair_id)
        if not math.isclose(ground_truth, expected_gt, rel_tol=0.0, abs_tol=1e-12):
            fail(f"{pair_id}: ground truth diverge do manifesto")
        prediction = require_number(row, "predicted_distance_m", pair_id)
        if ground_truth <= 0.0 or prediction <= 0.0:
            fail(f"{pair_id}: ground truth e predicao precisam ser positivos")
        if str(row.get("scene_id")) != str(expected.get("scene_id")):
            fail(f"{pair_id}: scene_id diverge do manifesto")
        if row.get("fold") != expected.get("fold"):
            fail(f"{pair_id}: fold diverge do manifesto")
        row["_log_error"] = math.log(prediction / ground_truth)

    return manifest_hash


def analyze_scene(scene_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fold_counts = Counter(str(row.get("fold")) for row in rows)
    if set(fold_counts) != {"A", "B"}:
        fail(f"cena {scene_id}: folds esperados A e B; observado {dict(fold_counts)}")
    if fold_counts["A"] != fold_counts["B"]:
        fail(f"cena {scene_id}: folds desequilibrados; observado {dict(fold_counts)}")

    fold_metrics: dict[str, dict[str, Any]] = {}
    scene_raw_squared: list[float] = []
    scene_corrected_squared: list[float] = []

    for test_fold in ("A", "B"):
        training_fold = "B" if test_fold == "A" else "A"
        training = [row for row in rows if row["fold"] == training_fold]
        testing = [row for row in rows if row["fold"] == test_fold]
        log_alpha, alpha = fit_log_scale(training)
        raw_errors = [float(row["_log_error"]) for row in testing]
        corrected_errors = [error + log_alpha for error in raw_errors]
        raw_sse = sum(error**2 for error in raw_errors)
        corrected_sse = sum(error**2 for error in corrected_errors)
        scene_raw_squared.extend(error**2 for error in raw_errors)
        scene_corrected_squared.extend(error**2 for error in corrected_errors)

        fold_metrics[test_fold] = {
            "training_fold": training_fold,
            "test_fold": test_fold,
            "log_alpha": log_alpha,
            "alpha": alpha,
            "raw_sse_log": raw_sse,
            "corrected_sse_log": corrected_sse,
            "raw_rmse_log": math.sqrt(raw_sse / len(testing)),
            "corrected_rmse_log": math.sqrt(corrected_sse / len(testing)),
            "g_cv": compute_gain(
                raw_sse, corrected_sse, f"cena {scene_id}, fold {test_fold}"
            ),
        }

    raw_sse = sum(scene_raw_squared)
    corrected_sse = sum(scene_corrected_squared)
    errors = [float(row["_log_error"]) for row in rows]
    mean_log_error = fmean(errors)
    ss_scale = len(errors) * mean_log_error**2
    ss_configuration = sum((error - mean_log_error) ** 2 for error in errors)
    ss_total = ss_scale + ss_configuration

    ground_truths = [float(row["ground_truth_m"]) for row in rows]
    predictions = [float(row["predicted_distance_m"]) for row in rows]
    prediction_counts = Counter(predictions)
    alpha_test_a = float(fold_metrics["A"]["alpha"])
    alpha_test_b = float(fold_metrics["B"]["alpha"])

    return {
        "scene_id": scene_id,
        "number_of_relations": len(rows),
        "ground_truth_min_m": min(ground_truths),
        "ground_truth_max_m": max(ground_truths),
        "fold_metrics": fold_metrics,
        "raw_rmse_log": math.sqrt(raw_sse / len(rows)),
        "corrected_rmse_log": math.sqrt(corrected_sse / len(rows)),
        "g_cv": compute_gain(raw_sse, corrected_sse, f"cena {scene_id}"),
        "mean_log_error": mean_log_error,
        "descriptive_alpha": math.exp(-mean_log_error),
        "ss_scale": ss_scale,
        "ss_configuration": ss_configuration,
        "scale_fraction": ss_scale / ss_total if ss_total else 0.0,
        "mae_m": fmean(abs(pred - gt) for pred, gt in zip(predictions, ground_truths)),
        "mape": fmean(
            abs(pred - gt) / gt for pred, gt in zip(predictions, ground_truths)
        ),
        "fold_alpha_ratio_max_min": (
            max(alpha_test_a, alpha_test_b) / min(alpha_test_a, alpha_test_b)
        ),
        "number_of_unique_predictions": len(prediction_counts),
        "prediction_counts": {
            f"{value:.6g}": count for value, count in sorted(prediction_counts.items())
        },
        "_raw_squared": scene_raw_squared,
        "_corrected_squared": scene_corrected_squared,
    }


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    manifest_path = checkpoint_dir / "manifest.jsonl"
    results_path = checkpoint_dir / "results.jsonl"
    metadata_path = checkpoint_dir / "metadata.json"
    output_json = args.output_json or checkpoint_dir / "metrics_recomputed.json"
    output_csv = args.output_csv or checkpoint_dir / "scene_metrics.csv"

    manifest = read_jsonl(manifest_path)
    results = read_jsonl(results_path)
    metadata = read_json(metadata_path)
    manifest_hash = validate_inputs(manifest, results, metadata, manifest_path)

    scene_ids = sorted({str(row["scene_id"]) for row in results})
    expected_scenes = metadata.get("number_of_scenes")
    if expected_scenes is not None and int(expected_scenes) != len(scene_ids):
        fail(
            f"metadata.number_of_scenes={expected_scenes}, "
            f"mas foram encontradas {len(scene_ids)} cenas"
        )

    scene_metrics = [
        analyze_scene(
            scene_id,
            [row for row in results if str(row["scene_id"]) == scene_id],
        )
        for scene_id in scene_ids
    ]

    all_raw_squared = [
        value for row in scene_metrics for value in row.pop("_raw_squared")
    ]
    all_corrected_squared = [
        value for row in scene_metrics for value in row.pop("_corrected_squared")
    ]
    g_values = [float(row["g_cv"]) for row in scene_metrics]
    scale_fractions = [float(row["scale_fraction"]) for row in scene_metrics]
    prediction_counts = Counter(
        float(row["predicted_distance_m"]) for row in results
    )
    elapsed = [require_number(row, "elapsed_seconds", str(row["pair_id"])) for row in results]

    aggregate = {
        "aggregation_rule": (
            "Compute within scene first, then aggregate scenes with equal weight."
        ),
        "number_of_scenes": len(scene_metrics),
        "number_of_relations": len(results),
        "mean_scene_g_cv": fmean(g_values),
        "median_scene_g_cv": median(g_values),
        "minimum_scene_g_cv": min(g_values),
        "maximum_scene_g_cv": max(g_values),
        "scenes_with_positive_g_cv": sum(value > 0.0 for value in g_values),
        "scenes_with_g_cv_above_half": sum(value > 0.5 for value in g_values),
        "pooled_g_cv": compute_gain(
            sum(all_raw_squared), sum(all_corrected_squared), "conjunto agrupado"
        ),
        "pooled_raw_rmse_log": math.sqrt(sum(all_raw_squared) / len(results)),
        "pooled_corrected_rmse_log": math.sqrt(
            sum(all_corrected_squared) / len(results)
        ),
        "mean_scene_scale_fraction": fmean(scale_fractions),
        "median_scene_scale_fraction": median(scale_fractions),
        "mean_scene_mae_m": fmean(float(row["mae_m"]) for row in scene_metrics),
        "mean_scene_mape": fmean(float(row["mape"]) for row in scene_metrics),
        "number_of_unique_predictions": len(prediction_counts),
        "prediction_counts": {
            f"{value:.6g}": count for value, count in sorted(prediction_counts.items())
        },
        "mean_elapsed_seconds": fmean(elapsed),
        "median_elapsed_seconds": median(elapsed),
        "total_elapsed_seconds": sum(elapsed),
    }

    output = {
        "integrity": {
            "manifest_sha256_computed": manifest_hash,
            "manifest_sha256_recorded": metadata.get("manifest_sha256"),
            "manifest_hash_matches": manifest_hash == metadata.get("manifest_sha256"),
            "valid_responses": len(results),
        },
        "metadata": metadata,
        "scene_metrics": scene_metrics,
        "aggregate": aggregate,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "scene_id",
        "number_of_relations",
        "ground_truth_min_m",
        "ground_truth_max_m",
        "raw_rmse_log",
        "corrected_rmse_log",
        "g_cv",
        "descriptive_alpha",
        "scale_fraction",
        "mae_m",
        "mape",
        "fold_alpha_ratio_max_min",
        "number_of_unique_predictions",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in scene_metrics:
            writer.writerow({field: row[field] for field in csv_fields})

    print("Metricas por cena:")
    for row in scene_metrics:
        print(
            f"{row['scene_id']} | "
            f"G_cv={row['g_cv']:.6f} | "
            f"RMSE-log={row['raw_rmse_log']:.6f} -> "
            f"{row['corrected_rmse_log']:.6f} | "
            f"escala={row['scale_fraction']:.6f} | "
            f"MAE={row['mae_m']:.6f} m | "
            f"MAPE={100 * row['mape']:.3f}%"
        )

    print("\nAgregado primario:")
    print(f"cenas={aggregate['number_of_scenes']}")
    print(f"relacoes={aggregate['number_of_relations']}")
    print(f"media_G_cv={aggregate['mean_scene_g_cv']:.9f}")
    print(f"mediana_G_cv={aggregate['median_scene_g_cv']:.9f}")
    print(f"cenas_G_cv_positivo={aggregate['scenes_with_positive_g_cv']}")
    print(f"cenas_G_cv_maior_que_0.5={aggregate['scenes_with_g_cv_above_half']}")
    print(f"\nJSON: {output_json}")
    print(f"CSV:  {output_csv}")


if __name__ == "__main__":
    main()
