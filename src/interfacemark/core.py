"""Mathematical core of InterfaceMark.

All four released variants modify only the terminal latent. They share the
same secret unit carrier and analytic projection detector; only the scalar
displacement law differs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.stats import norm


VARIANTS = (
    "tail_coupled",
    "shuffled_tail",
    "fixed_rms",
    "fixed_quality",
)


def unit_carrier(shape: Sequence[int], seed: int) -> torch.Tensor:
    """Create a deterministic unit-norm Gaussian carrier with a batch axis."""
    if len(shape) < 2 or int(shape[0]) != 1:
        raise ValueError("carrier shape must start with a singleton batch axis")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    carrier = torch.randn(tuple(int(value) for value in shape), generator=generator)
    norm_value = carrier.flatten(1).norm(dim=1)
    if bool((norm_value <= 0).any()):
        raise RuntimeError("sampled a zero-norm carrier")
    return carrier / norm_value.view(-1, *([1] * (carrier.ndim - 1)))


def project(states: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
    """Return the keyed scalar projection for every state in a batch."""
    if states.ndim != carrier.ndim:
        raise ValueError("states and carrier must have the same rank")
    if tuple(states.shape[1:]) != tuple(carrier.shape[1:]):
        raise ValueError(
            f"shape mismatch: states={tuple(states.shape)}, "
            f"carrier={tuple(carrier.shape)}"
        )
    aligned = carrier.to(device=states.device, dtype=torch.float32)
    return states.float().flatten(1) @ aligned.flatten()


def tail_target(projection_value: torch.Tensor, probability: float) -> torch.Tensor:
    """Map N(0,1) projections into the keyed upper-tail conditional law.

    For A~N(0,1), this implements

        SF(T_p(A)) = p * SF(A),

    where SF is the standard-normal survival function. Consequently T_p(A)
    follows N(0,1) conditioned on the upper tail whose probability is p.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    before = projection_value.detach().double().cpu().numpy()
    survival = norm.sf(before)
    scaled = np.clip(
        float(probability) * survival,
        np.finfo(np.float64).tiny,
        1.0 - np.finfo(np.float64).eps,
    )
    result = torch.from_numpy(np.asarray(norm.isf(scaled))).to(
        device=projection_value.device,
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("tail transport produced a non-finite target")
    return result


def tail_displacement(
    base_noise: torch.Tensor,
    carrier: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Compute the positive sample-dependent Tail-coupled displacement."""
    before = project(base_noise, carrier).double()
    displacement = (tail_target(before, probability) - before).float()
    if not bool(torch.isfinite(displacement).all()) or bool(
        (displacement <= 0).any()
    ):
        raise RuntimeError("tail displacement must be finite and positive")
    return displacement


def inject_terminal(
    terminal_latent: torch.Tensor,
    carrier: torch.Tensor,
    displacement: torch.Tensor | float,
) -> torch.Tensor:
    """Shift each terminal latent along the secret carrier."""
    if isinstance(displacement, (float, int)):
        delta = torch.full(
            (terminal_latent.shape[0],),
            float(displacement),
            device=terminal_latent.device,
            dtype=torch.float32,
        )
    else:
        delta = displacement.to(
            device=terminal_latent.device,
            dtype=torch.float32,
        ).reshape(-1)
    if len(delta) == 1 and terminal_latent.shape[0] > 1:
        delta = delta.expand(terminal_latent.shape[0])
    if len(delta) != terminal_latent.shape[0]:
        raise ValueError("one displacement is required per terminal latent")
    view_shape = (-1,) + (1,) * (terminal_latent.ndim - 1)
    shifted = (
        terminal_latent.float()
        + delta.view(view_shape) * carrier.to(terminal_latent.device).float()
    )
    return shifted.to(terminal_latent.dtype)


def fixed_rms(tail_displacements: Sequence[float]) -> float:
    """Return the fixed displacement with matched RMS magnitude."""
    values = np.asarray(tail_displacements, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("tail_displacements must be a non-empty vector")
    return float(np.sqrt(np.mean(np.square(values))))


def shuffled(values: Sequence[float], seed: int) -> tuple[list[float], list[int]]:
    """Permute tail displacements without changing their empirical marginal."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(len(values), generator=generator).tolist()
    return [float(values[index]) for index in permutation], permutation
