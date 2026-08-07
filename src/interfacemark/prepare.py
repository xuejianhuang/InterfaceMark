"""Prepare a reproducible Terminal InterfaceMark experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core import (
    VARIANTS,
    fixed_rms,
    inject_terminal,
    shuffled,
    tail_displacement,
    unit_carrier,
)
from .records import (
    atomic_json,
    atomic_torch,
    load_prompts,
    load_yaml,
    sha256,
    stable_hash,
)
from .sd15 import (
    decode,
    environment_record,
    generate_terminal,
    latent_shape,
    load_pipeline,
    rgb_mse,
    sample_noise,
)


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"missing required configuration key: {key}")
    return config[key]


def _make_specs(
    prompts: list[dict[str, str]],
    design_size: int,
    calibration_size: int,
    test_size: int,
    split_seed: int,
    base_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    order = torch.randperm(len(prompts), generator=generator).tolist()
    shuffled_prompts = [prompts[index] for index in order]
    bounds = (0, len(prompts) // 3, 2 * len(prompts) // 3, len(prompts))
    pools = {
        "design": shuffled_prompts[bounds[0] : bounds[1]],
        "calibration": shuffled_prompts[bounds[1] : bounds[2]],
        "test": shuffled_prompts[bounds[2] : bounds[3]],
    }
    sizes = {
        "design": design_size,
        "calibration": calibration_size,
        "test": test_size,
    }
    offsets = {"design": 0, "calibration": 1_000_000, "test": 2_000_000}
    specs: dict[str, list[dict[str, Any]]] = {}
    for split, size in sizes.items():
        pool = pools[split]
        specs[split] = []
        for index in range(size):
            prompt = pool[index % len(pool)]
            specs[split].append(
                {
                    "split": split,
                    "index": index,
                    "seed": base_seed + offsets[split] + index,
                    "prompt": prompt["Prompt"],
                    "category": prompt.get("Category", ""),
                    "challenge": prompt.get("Challenge", ""),
                    "prompt_occurrence": index // len(pool),
                }
            )
    return specs


def _quality_match(
    pipeline: Any,
    specs: list[dict[str, Any]],
    shape: tuple[int, ...],
    carrier: torch.Tensor,
    quality_tail_values: list[float],
    candidate_source_values: list[float],
    steps: int,
    guidance: float,
    grid_size: int,
) -> dict[str, Any]:
    candidates = np.linspace(
        float(np.quantile(candidate_source_values, 0.05)),
        float(np.quantile(candidate_source_values, 0.95)),
        int(grid_size),
    )
    tail_mse: list[float] = []
    candidate_mse: list[list[float]] = [[] for _ in candidates]
    with torch.inference_mode():
        for spec, tail_delta in zip(specs, quality_tail_values, strict=True):
            base = sample_noise(shape, int(spec["seed"]))
            terminal = generate_terminal(
                pipeline,
                str(spec["prompt"]),
                base,
                steps,
                guidance,
            )
            latents = [
                terminal,
                inject_terminal(terminal, carrier, tail_delta),
                *[
                    inject_terminal(terminal, carrier, float(value))
                    for value in candidates
                ],
            ]
            images = decode(pipeline, torch.cat(latents))
            tail_mse.append(rgb_mse(images[0], images[1]))
            for index in range(len(candidates)):
                candidate_mse[index].append(rgb_mse(images[0], images[index + 2]))
    target = float(np.median(tail_mse))
    candidate_medians = [float(np.median(values)) for values in candidate_mse]
    selected = int(np.argmin(np.abs(np.asarray(candidate_medians) - target)))
    return {
        "metric": "median_paired_rgb_mse",
        "candidate_displacements": [float(value) for value in candidates],
        "candidate_median_mse": candidate_medians,
        "tail_median_mse": target,
        "selected_index": selected,
        "gamma": float(candidates[selected]),
        "selected_median_mse": candidate_medians[selected],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="override the output path stored in the YAML configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    output = args.output or Path(str(_required(raw, "output")))
    model = Path(str(_required(raw, "model")))
    prompt_path = Path(str(_required(raw, "prompts")))
    output.mkdir(parents=True, exist_ok=True)

    parameters = {
        "steps": int(raw.get("steps", 30)),
        "guidance": float(raw.get("guidance", 7.5)),
        "resolution": int(raw.get("resolution", 512)),
        "tail_probability": float(raw.get("tail_probability", 2.5e-7)),
        "carrier_seed": int(raw.get("carrier_seed", 51001)),
        "wrong_carrier_seed": int(raw.get("wrong_carrier_seed", 51002)),
        "shuffle_seed": int(raw.get("shuffle_seed", 51003)),
        "split_seed": int(raw.get("split_seed", 2026072801)),
        "base_seed": int(raw.get("base_seed", 50_000_000)),
        "design_size": int(raw.get("design_size", 640)),
        "quality_design_count": int(raw.get("quality_design_count", 64)),
        "quality_grid_size": int(raw.get("quality_grid_size", 9)),
        "calibration_size": int(raw.get("calibration_size", 5000)),
        "test_size": int(raw.get("test_size", 5000)),
        "target_fpr": float(raw.get("target_fpr", 0.01)),
    }
    if not 2 <= parameters["quality_design_count"] <= parameters["design_size"]:
        raise ValueError(
            "quality_design_count must be at least two and no larger than design_size"
        )

    prompts = load_prompts(prompt_path)
    specs = _make_specs(
        prompts,
        parameters["design_size"],
        parameters["calibration_size"],
        parameters["test_size"],
        parameters["split_seed"],
        parameters["base_seed"],
    )

    pipeline = load_pipeline(model)
    shape = latent_shape(pipeline, parameters["resolution"])
    carrier = unit_carrier(shape, parameters["carrier_seed"])
    wrong_carrier = unit_carrier(shape, parameters["wrong_carrier_seed"])

    design_tail = [
        float(
            tail_displacement(
                sample_noise(shape, int(spec["seed"])),
                carrier,
                parameters["tail_probability"],
            )[0]
        )
        for spec in specs["design"]
    ]
    test_tail = [
        float(
            tail_displacement(
                sample_noise(shape, int(spec["seed"])),
                carrier,
                parameters["tail_probability"],
            )[0]
        )
        for spec in specs["test"]
    ]
    test_shuffled, permutation = shuffled(
        test_tail,
        parameters["shuffle_seed"],
    )

    started = time.perf_counter()
    quality = _quality_match(
        pipeline,
        specs["design"][: parameters["quality_design_count"]],
        shape,
        carrier,
        design_tail[: parameters["quality_design_count"]],
        design_tail,
        parameters["steps"],
        parameters["guidance"],
        parameters["quality_grid_size"],
    )

    config_core = {
        "schema_version": 1,
        "method": "InterfaceMark",
        "channel": "frozen_sd15_terminal_latent",
        "variants": list(VARIANTS),
        "model": str(model.resolve()),
        "prompts": str(prompt_path.resolve()),
        "prompt_sha256": sha256(prompt_path),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": sha256(args.config),
        "output": str(output.resolve()),
        "latent_shape": list(shape),
        **parameters,
        "sizes": {name: len(rows) for name, rows in specs.items()},
        "environment": environment_record(),
    }
    config_hash = stable_hash(config_core)
    resolved = {
        **config_core,
        "config_hash": config_hash,
        "specs": specs,
    }
    controls = {
        "schema_version": 1,
        "config_hash": config_hash,
        "tail_coupled": {"test_displacements": test_tail},
        "shuffled_tail": {
            "test_displacements": test_shuffled,
            "permutation": permutation,
        },
        "fixed_rms": {"gamma": fixed_rms(design_tail)},
        "fixed_quality": quality,
        "design_tail_displacements": design_tail,
    }
    atomic_json(output / "config.resolved.json", resolved)
    atomic_json(output / "controls.json", controls)
    atomic_torch(
        output / "carriers.pt",
        {
            "config_hash": config_hash,
            "correct": carrier,
            "wrong": wrong_carrier,
        },
    )
    (output / "PREPARE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "prepare_complete",
                "output": str(output),
                "config_hash": config_hash,
                "variants": list(VARIANTS),
                "fixed_rms_gamma": controls["fixed_rms"]["gamma"],
                "fixed_quality_gamma": quality["gamma"],
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
