"""InterfaceMark terminal-interface watermarking."""

from .core import (
    VARIANTS,
    inject_terminal,
    project,
    tail_displacement,
    unit_carrier,
)

__all__ = [
    "VARIANTS",
    "inject_terminal",
    "project",
    "tail_displacement",
    "unit_carrier",
]

__version__ = "0.1.0"
