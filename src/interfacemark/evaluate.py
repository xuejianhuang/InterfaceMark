"""Audit and evaluate a completed Terminal InterfaceMark experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .core import VARIANTS
from .records import atomic_json, read_json


def _binary_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    values = np.concatenate([negative, positive])
    ranks = rankdata(values, method="average")
    n_negative = len(negative)
    n_positive = len(positive)
    positive_rank_sum = float(ranks[n_negative:].sum())
    statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2
    return float(statistic / (n_negative * n_positive))


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _threshold(scores: np.ndarray, target_fpr: float) -> float:
    allowed = int(math.floor(target_fpr * len(scores)))
    ordered = np.sort(scores)
    if allowed <= 0:
        return float(ordered[-1])
    return float(ordered[-allowed - 1])


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _audit(
    root: Path,
    split: str,
    expected: int,
    config_hash: str,
) -> list[dict[str, Any]]:
    image_names = {"clean"} if split == "calibration" else {
        "clean",
        *VARIANTS,
    }
    rows: list[dict[str, Any]] = []
    for index in range(expected):
        shard = root / "splits" / split / f"{index:05d}"
        summary = shard / "summary.json"
        if not summary.exists():
            raise RuntimeError(f"missing summary: {summary}")
        row = read_json(summary)
        if row.get("config_hash") != config_hash:
            raise RuntimeError(f"config hash mismatch: {summary}")
        if int(row.get("index", -1)) != index:
            raise RuntimeError(f"index mismatch: {summary}")
        if not _finite(row):
            raise RuntimeError(f"non-finite record: {summary}")
        actual = {path.stem for path in shard.glob("*.png")}
        if actual != image_names:
            raise RuntimeError(
                f"image inventory mismatch at {shard}: "
                f"expected={sorted(image_names)}, actual={sorted(actual)}"
            )
        if set(row["scores"]) != image_names:
            raise RuntimeError(f"score inventory mismatch: {summary}")
        rows.append(row)
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != expected:
        raise RuntimeError(f"duplicate seeds in {split}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--target-fpr",
        type=float,
        help="override the target FPR stored in the resolved configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input
    config = read_json(root / "config.resolved.json")
    config_hash = str(config["config_hash"])
    target_fpr = (
        float(args.target_fpr)
        if args.target_fpr is not None
        else float(config["target_fpr"])
    )
    calibration = _audit(
        root,
        "calibration",
        int(config["calibration_size"]),
        config_hash,
    )
    test = _audit(root, "test", int(config["test_size"]), config_hash)
    calibration_seeds = {int(row["seed"]) for row in calibration}
    test_seeds = {int(row["seed"]) for row in test}
    if calibration_seeds & test_seeds:
        raise RuntimeError("calibration and test seeds overlap")

    calibration_scores = np.asarray(
        [row["scores"]["clean"]["correct"] for row in calibration],
        dtype=np.float64,
    )
    test_clean = np.asarray(
        [row["scores"]["clean"]["correct"] for row in test],
        dtype=np.float64,
    )
    decision_threshold = _threshold(calibration_scores, target_fpr)
    observed_fpr_count = int(np.sum(test_clean > decision_threshold))

    methods: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        positive = np.asarray(
            [row["scores"][variant]["correct"] for row in test],
            dtype=np.float64,
        )
        wrong_clean = np.asarray(
            [row["scores"]["clean"]["wrong"] for row in test],
            dtype=np.float64,
        )
        wrong_positive = np.asarray(
            [row["scores"][variant]["wrong"] for row in test],
            dtype=np.float64,
        )
        tpr_count = int(np.sum(positive > decision_threshold))
        quality = [row["paired_quality"][variant] for row in test]
        result = {
            "auc": _binary_auc(test_clean, positive),
            "tpr_at_calibrated_fpr": tpr_count / len(positive),
            "tpr_wilson_95": _wilson(tpr_count, len(positive)),
            "observed_test_fpr": observed_fpr_count / len(test_clean),
            "observed_test_fpr_wilson_95": _wilson(
                observed_fpr_count,
                len(test_clean),
            ),
            "wrong_key_auc": _binary_auc(wrong_clean, wrong_positive),
            "mean_psnr": float(np.mean([item["psnr"] for item in quality])),
            "median_psnr": float(np.median([item["psnr"] for item in quality])),
            "mean_rgb_mse": float(
                np.mean([item["rgb_mse"] for item in quality])
            ),
            "mean_seconds_per_test_sample": float(
                np.mean([row["seconds"] for row in test])
            ),
            "peak_cuda_gib": float(max(row["peak_cuda_gib"] for row in test)),
        }
        methods[variant] = result
        csv_rows.append({"variant": variant, **result})

    summary = {
        "schema_version": 1,
        "event": "interfacemark_evaluation_complete",
        "config_hash": config_hash,
        "counts": {
            "calibration": len(calibration),
            "test": len(test),
        },
        "target_fpr": target_fpr,
        "threshold": decision_threshold,
        "threshold_rule": (
            "correct-key analytic projection > independent clean-calibration "
            "order-statistic threshold"
        ),
        "calibration_activation": float(
            np.mean(calibration_scores > decision_threshold)
        ),
        "methods": methods,
    }
    atomic_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        scalar_fields = [
            "variant",
            "auc",
            "tpr_at_calibrated_fpr",
            "observed_test_fpr",
            "wrong_key_auc",
            "mean_psnr",
            "median_psnr",
            "mean_rgb_mse",
            "mean_seconds_per_test_sample",
            "peak_cuda_gib",
        ]
        writer = csv.DictWriter(handle, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    (root / "EVALUATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
