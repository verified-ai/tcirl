# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
"""The `flax.nnx` layer helper the deep agent builds its network from."""

from __future__ import annotations

import flax.nnx as nnx


def make_linear(in_features: int, out_features: int, rngs: nnx.Rngs) -> nnx.Linear:
    """A dense layer at JAX's default initialization (Lecun normal, fan-in, truncated)."""
    return nnx.Linear(in_features, out_features, rngs=rngs)
