# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false
"""The deep baseline's network and optimizer.

A plain multilayer perceptron with ReLU hidden layers and a linear head, which is what the paper
describes ("a two-layer MLP (128 neurons each)"). The training rule lives in `dqn_core`: Double DQN's
target (van Hasselt et al., 2016) with a squared error and no gradient clipping. `double_q`, `huber`
and `max_grad_norm` are config switches.
"""

from __future__ import annotations

from typing import override

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from tcirl.rl.networks import make_linear


class DqnModel(nnx.Module):
    """`Q(obs) -> (num_actions,)`: an MLP with `hidden_sizes` ReLU layers and a linear head."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        if not hidden_sizes:
            raise ValueError("hidden_sizes must name at least one hidden layer")
        widths = (obs_dim, *hidden_sizes)
        # `nnx.List`, not a plain list: flax needs the container declared as data for the layers
        # inside it to be part of the module's pytree, and so of the vmapped parameter stack.
        self.hidden = nnx.List(
            [make_linear(widths[i], widths[i + 1], rngs) for i in range(len(hidden_sizes))]
        )
        self.head = make_linear(hidden_sizes[-1], num_actions, rngs)

    @override
    def __call__(self, obs: jax.Array) -> jax.Array:
        """`obs (B, obs_dim)` -> `(B, num_actions)`."""
        h = obs
        for layer in self.hidden:
            h = nnx.relu(layer(h))
        return self.head(h)


def make_dqn_model(
    *,
    obs_dim: int,
    num_actions: int,
    hidden_sizes: tuple[int, ...],
    key: jax.Array,
) -> tuple[nnx.GraphDef[DqnModel], nnx.GraphState]:
    """Build one `DqnModel` and split it into `(graphdef, params)`.

    The params are a plain pytree, so `jax.vmap` stacks them across seeds and `optax` updates them
    without this module knowing how many layers there are.
    """
    model = DqnModel(
        rngs=nnx.Rngs(key),
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_sizes=hidden_sizes,
    )
    graphdef, params = nnx.split(model)
    return graphdef, params  # pyright: ignore[reportReturnType]


def make_optimizer(
    learning_rate: float, max_grad_norm: float | None
) -> optax.GradientTransformation:
    """Adam at a constant rate, preceded by global-norm gradient clipping when one is set.

    `max_grad_norm=None` disables clipping.

    The rate is constant; a network has no per-entry step size.
    """
    if not learning_rate > 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    adam = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is None:
        return adam
    if not max_grad_norm > 0.0:
        raise ValueError(f"max_grad_norm must be positive or None, got {max_grad_norm}")
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), adam)


def huber_or_squared(pred: jax.Array, target: jax.Array, *, delta: float, huber: bool) -> jax.Array:
    """Per-row loss: Huber at `delta` when `huber`, else the squared residual."""
    if huber:
        return optax.huber_loss(pred, target, delta=delta)
    return (pred - target) ** 2


def global_norm(tree: nnx.GraphState) -> jax.Array:
    """The global L2 norm of a gradient pytree, logged as `train/grad_norm`."""
    return jnp.asarray(optax.tree.norm(tree))
