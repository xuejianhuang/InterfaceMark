"""Generate resumable calibration/test shards for Terminal InterfaceMark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .core import VARIANTS, inject_terminal, project
from .records import atomic_json, read_json
from .sd15 import (
    decode,
    encode,
    generate_terminal,
    load_pipeline,
    paired_psnr,
    peak_cuda_gib,
    rgb_mse,
    sample_noise,
)


def _expected_images(split: str) -> set[str]:
    if split == "calibration":
        return {"clean.png"}
    return {"clean.png", *(f"{name}.png" for name in VARIANTS)}


def _complete(shard: Path, split: str, config_hash: str) -> bool:
    summary = shard / "summary.json"
    if not summary.exists():
        return False
    if {path.name for path in shard.glob("*.png")} != _expected_images(split):
        return False
    try:
        return read_json(summary).get("config_hash") == config_hash
    except Exception:
        return False


def _all_complete(root: Path, split: str, count: int, config_hash: str) -> bool:
    return all(
        _complete(root / "splits" / split / f"{index:05d}", split, config_hash)
        for index in range(count)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("calibration", "test"),
        required=True,
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--count",
        type=int,
        help="number of samples; omit to process the rest of the split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input
    config = read_json(root / "config.resolved.json")
    controls = read_json(root / "controls.json")
    carriers = torch.load(root / "carriers.pt", map_location="cpu", weights_only=True)
    config_hash = str(config["config_hash"])
    if controls.get("config_hash") != config_hash:
        raise RuntimeError("controls/config hash mismatch")
    if carriers.get("config_hash") != config_hash:
        raise RuntimeError("carrier/config hash mismatch")

    specs = config["specs"][args.split]
    stop = len(specs) if args.count is None else min(
        len(specs),
        args.start + args.count,
    )
    if args.start < 0 or args.start >= len(specs) or stop <= args.start:
        raise ValueError("requested sample range is empty or out of bounds")

    pipeline = load_pipeline(config["model"])
    shape = tuple(int(value) for value in config["latent_shape"])
    carrier = carriers["correct"]
    wrong = carriers["wrong"]
    torch.cuda.reset_peak_memory_stats()

    for spec in specs[args.start:stop]:
        index = int(spec["index"])
        shard = root / "splits" / args.split / f"{index:05d}"
        if _complete(shard, args.split, config_hash):
            continue
        started = time.perf_counter()
        noise = sample_noise(shape, int(spec["seed"]))
        with torch.inference_mode():
            terminal = generate_terminal(
                pipeline,
                str(spec["prompt"]),
                noise,
                int(config["steps"]),
                float(config["guidance"]),
            )
            if args.split == "calibration":
                deltas: dict[str, float] = {}
                names = ["clean"]
                latents = [terminal]
            else:
                deltas = {
                    "tail_coupled": float(
                        controls["tail_coupled"]["test_displacements"][index]
                    ),
                    "shuffled_tail": float(
                        controls["shuffled_tail"]["test_displacements"][index]
                    ),
                    "fixed_rms": float(controls["fixed_rms"]["gamma"]),
                    "fixed_quality": float(controls["fixed_quality"]["gamma"]),
                }
                names = ["clean", *VARIANTS]
                latents = [
                    terminal,
                    *[
                        inject_terminal(terminal, carrier, deltas[name])
                        for name in VARIANTS
                    ],
                ]
            images = decode(pipeline, torch.cat(latents))
            observed = encode(pipeline, images).float().cpu()

        shard.mkdir(parents=True, exist_ok=True)
        scores: dict[str, dict[str, float]] = {}
        quality: dict[str, dict[str, float]] = {}
        for offset, (name, image) in enumerate(zip(names, images, strict=True)):
            image.save(shard / f"{name}.png")
            state = observed[offset : offset + 1]
            scores[name] = {
                "correct": float(project(state, carrier)),
                "wrong": float(project(state, wrong)),
            }
            if name != "clean":
                quality[name] = {
                    "psnr": paired_psnr(images[0], image),
                    "rgb_mse": rgb_mse(images[0], image),
                }
        row: dict[str, Any] = {
            "schema_version": 1,
            "event": "interfacemark_sample_complete",
            "config_hash": config_hash,
            **spec,
            "deltas": deltas,
            "scores": scores,
            "paired_quality": quality,
            "seconds": time.perf_counter() - started,
            "peak_cuda_gib": peak_cuda_gib(),
        }
        atomic_json(shard / "summary.json", row)
        print(
            json.dumps(
                {
                    "event": "sample_complete",
                    "split": args.split,
                    "index": index,
                    "seconds": row["seconds"],
                }
            ),
            flush=True,
        )

    if _all_complete(root, args.split, len(specs), config_hash):
        (root / f"{args.split.upper()}_COMPLETE").write_text(
            "complete\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
