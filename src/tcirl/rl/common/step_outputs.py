"""The per-step training quantities every iteration primitive emits, and nothing else.

`qrm_core.make_iteration` (tabular) and `dqn_core.make_dqn_iteration` (deep) are different
algorithms over different state, but the training curves are the same six
quantities in both. Declaring them as one protocol is what lets `qrm_logging.log_batch`, the
`BatchLogger`, the recorder, the registry and the exporter serve both arms with no second path.

`BatchOutputs` (the tabular one) carries automaton fields on top; it satisfies this protocol
because a NamedTuple's fields are read-only properties. The deep iteration has no automaton, so it
emits the concrete `StepOutputs`.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax


class TrainStepOutputs(Protocol):
    """Per-step per-seed arrays, `(num_runs, batch_steps)` after vmap + scan."""

    @property
    def reward(self) -> jax.Array:
        """float32, the true env reward this transition paid."""
        ...

    @property
    def td_abs(self) -> jax.Array:
        """float32, the absolute TD error of this step's update; NaN where none ran."""
        ...

    @property
    def q_max(self) -> jax.Array:
        """float32, the max Q over the legal actions at the state acted from."""
        ...

    @property
    def episode_over(self) -> jax.Array:
        """bool, the episode-boundary bit (termination or horizon truncation)."""
        ...

    @property
    def ep_length(self) -> jax.Array:
        """float32, the env step count at an `episode_over` step, else 0."""
        ...

    @property
    def ep_return(self) -> jax.Array:
        """float32, the finished episode's UNDISCOUNTED return at an `episode_over` step, else 0.

        Emitted once per episode, at the step that ends it, exactly like `ep_length`. That is what
        makes `sum(ep_return) / count(episode_over)` over a window the mean return of the episodes
        that FINISHED in it, with no reward from a partial episode in the numerator.
        """
        ...

    @property
    def ep_disc_return(self) -> jax.Array:
        """float32, the finished episode's DISCOUNTED return at an `episode_over` step, else 0.

        Discounted from the episode's own first step. It is the TRUNCATED discounted return and not
        the objective: both learners bootstrap through the horizon truncation, so they maximise the
        infinite-horizon sum, of which this keeps the first `max_episode_steps` terms.
        """
        ...

    @property
    def label(self) -> jax.Array:
        """int32, the alphabet symbol this transition emitted."""
        ...


class StepOutputs(NamedTuple):
    """The protocol's concrete carrier, for an iteration with no automaton to report."""

    reward: jax.Array
    td_abs: jax.Array
    q_max: jax.Array
    episode_over: jax.Array
    ep_length: jax.Array
    ep_return: jax.Array
    ep_disc_return: jax.Array
    label: jax.Array
