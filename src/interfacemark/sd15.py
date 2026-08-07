"""Frozen Stable Diffusion 1.5 channel used by the released experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def load_pipeline(model: str | Path) -> StableDiffusionPipeline:
    """Load a local fp16 SD1.5 checkpoint without changing any model weights."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the released SD1.5 protocol")
    pipeline = StableDiffusionPipeline.from_pretrained(
        str(model),
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def latent_shape(
    pipeline: StableDiffusionPipeline,
    resolution: int = 512,
) -> tuple[int, int, int, int]:
    channels = int(pipeline.unet.config.in_channels)
    side = int(resolution // pipeline.vae_scale_factor)
    return (1, channels, side, side)


def sample_noise(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator)


def generate_terminal(
    pipeline: StableDiffusionPipeline,
    prompt: str | list[str],
    noise: torch.Tensor,
    steps: int,
    guidance: float,
) -> torch.Tensor:
    """Run the frozen sampler and return its final latent before VAE decoding."""
    return pipeline(
        prompt=prompt,
        latents=noise.to(device="cuda", dtype=torch.float16),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        output_type="latent",
    ).images


def decode(
    pipeline: StableDiffusionPipeline,
    latents: torch.Tensor,
) -> list[Image.Image]:
    scaling = float(pipeline.vae.config.scaling_factor)
    decoded = pipeline.vae.decode(latents / scaling, return_dict=False)[0]
    return pipeline.image_processor.postprocess(decoded, output_type="pil")


def encode(
    pipeline: StableDiffusionPipeline,
    images: list[Image.Image],
) -> torch.Tensor:
    """Deterministically VAE-encode RGB images using the posterior mode."""
    pixels = pipeline.image_processor.preprocess(images).to(
        device=pipeline._execution_device,
        dtype=pipeline.vae.dtype,
    )
    scaling = float(pipeline.vae.config.scaling_factor)
    return pipeline.vae.encode(pixels).latent_dist.mode() * scaling


def rgb_mse(first: Image.Image, second: Image.Image) -> float:
    x = np.asarray(first, dtype=np.float32) / 255.0
    y = np.asarray(second, dtype=np.float32) / 255.0
    return float(np.mean(np.square(x - y)))


def paired_psnr(first: Image.Image, second: Image.Image) -> float:
    return float(-10.0 * math.log10(max(rgb_mse(first, second), 1e-12)))


def peak_cuda_gib() -> float:
    return float(torch.cuda.max_memory_allocated() / 2**30)


def environment_record() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
