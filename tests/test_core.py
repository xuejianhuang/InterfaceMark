from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

from interfacemark.core import (
    fixed_rms,
    inject_terminal,
    project,
    shuffled,
    tail_displacement,
    unit_carrier,
)


def test_carrier_has_unit_norm_and_is_reproducible() -> None:
    first = unit_carrier((1, 4, 8, 8), seed=17)
    second = unit_carrier((1, 4, 8, 8), seed=17)
    assert torch.equal(first, second)
    assert torch.allclose(first.flatten().norm(), torch.tensor(1.0), atol=1e-6)


def test_tail_transport_obeys_survival_identity() -> None:
    carrier = unit_carrier((1, 1, 1, 4), seed=3)
    base = torch.tensor(
        [
            [[[0.1, -0.2, 0.3, -0.4]]],
            [[[1.0, 0.5, -0.5, -1.0]]],
        ]
    )
    probability = 2.5e-3
    before = project(base, carrier)
    delta = tail_displacement(base, carrier, probability)
    after = before + delta
    np.testing.assert_allclose(
        norm.sf(after.numpy()),
        probability * norm.sf(before.numpy()),
        rtol=1e-5,
        atol=1e-12,
    )
    assert bool((delta > 0).all())


def test_terminal_injection_adds_exact_keyed_displacement() -> None:
    carrier = unit_carrier((1, 2, 4, 4), seed=9)
    terminal = torch.randn((3, 2, 4, 4))
    delta = torch.tensor([0.5, 1.0, 2.0])
    before = project(terminal, carrier)
    after = project(inject_terminal(terminal, carrier, delta), carrier)
    assert torch.allclose(after - before, delta, atol=1e-5)


def test_control_laws_preserve_their_stated_constraints() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    permuted, indices = shuffled(values, seed=11)
    assert sorted(permuted) == values
    assert sorted(indices) == list(range(len(values)))
    assert np.isclose(fixed_rms(values), np.sqrt(np.mean(np.square(values))))
