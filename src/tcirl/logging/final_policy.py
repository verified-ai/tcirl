# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false
"""The final policy of a run, saved once per seed so a question about it needs no retrain.

Every training loop returns the policy its last evaluation point measured: a Q-table and the DFA
it is indexed by for the tabular learner, and network parameters for the DQN. For TCIRL that is the
pair bound BEFORE the final batch's re-inference (`agents/tabular.py`), so the file holds what the
paper's final numbers were measured on, and the final policy can be analysed without retraining. Each run writes `checkpoints/final_policy.npz` into its own directory, beside
`metric_curves/`.

One file, one convention: every array carries the SEED axis first and the same seed count, the
same bank the recorder uses, so seed `i` here is seed `i` of every metric of the run. The file is
a plain `.npz` so an analysis reads it with numpy alone; the DQN needs a template state to become a
model again, which `dqn_state_from_arrays` does from a fresh `init_per_seed_params`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tcirl.agents.tabular import TabularParams
from tcirl.automata.dfa import DFA
from tcirl.automata.plain_dfa import PlainDFA
from tcirl.logging.wandb import WandbLogger

FINAL_POLICY_SUBDIR = "checkpoints"
FINAL_POLICY_FILE = "final_policy.npz"
_ALGO_KEY = "algo"
_DQN_PREFIX = "params"


class FinalPolicy(NamedTuple):
    """What `load_final_policy` returns: the algorithm's name and its per-seed arrays."""

    algo: str
    arrays: dict[str, np.ndarray]

    @property
    def num_seeds(self) -> int:
        return next(iter(self.arrays.values())).shape[0]


def final_policy_path(run_dir: Path) -> Path:
    return run_dir / FINAL_POLICY_SUBDIR / FINAL_POLICY_FILE


def save_final_policy(
    run_dir: Path | None,
    logger: WandbLogger,
    arrays: Mapping[str, np.ndarray | jax.Array],
    *,
    algo: str,
) -> Path | None:
    """Write the run's final policy and register the file with the run.

    `run_dir` is the run's artifact directory, `logger.run_dir` in a CLI and any directory another
    caller chooses; `None` means no run is open and nothing persists, as for the recorder. Every array
    must carry the seed axis first and agree on the seed count, or the file would let a reader pair
    one seed's table with another seed's hypothesis.
    """
    if run_dir is None:
        return None
    host = {key: np.asarray(jax.device_get(value)) for key, value in arrays.items()}
    seeds = {key: value.shape[0] if value.ndim else -1 for key, value in host.items()}
    if len(set(seeds.values())) != 1 or -1 in seeds.values():
        raise ValueError(f"every array must carry the same seed axis first, got {seeds}")
    if _ALGO_KEY in host:
        raise ValueError(f"{_ALGO_KEY!r} is reserved for the algorithm's name")
    path = final_policy_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {**host, _ALGO_KEY: np.array(algo)}
    np.savez_compressed(path, allow_pickle=True, **payload)
    logger.save_file(path, base_path=run_dir)
    return path


def load_final_policy(run_dir: Path) -> FinalPolicy:
    """Read a run's final policy back; raises when the run has none."""
    path = final_policy_path(run_dir)
    with np.load(path) as data:
        algo = str(data[_ALGO_KEY])
        arrays = {key: data[key] for key in data.files if key != _ALGO_KEY}
    return FinalPolicy(algo=algo, arrays=arrays)


# ---------------------------------------------------------------------------------------------
# What each agent's training returns, as the arrays the file holds.
# ---------------------------------------------------------------------------------------------


def tabular_arrays(q_tables: jax.Array, automata: Sequence[DFA]) -> dict[str, np.ndarray]:
    """The tabular learner's per-seed Q-table and DFA, the pair its greedy policy reads.

    The DFA is TCIRL's hypothesis, or the cause DFA or the one-state DFA of Q-learning. The DFAs can
    differ in size across seeds, so each is padded to the largest one with self-loop filler states
    (`PlainDFA.from_dfa`), and `hyp_num_states` says how many states of each seed's `hyp_delta` are
    real. State 0 is initial, as in every DFA of this repository. That width is the largest DFA and
    not the table's automaton axis, the axis before the action axis, which the run sizes at its
    largest possible DFA; every state a DFA can reach is below its own `hyp_num_states`, so the two
    widths never index a wrong value.
    """
    q = np.asarray(jax.device_get(q_tables))
    if q.shape[0] != len(automata):
        raise ValueError(
            f"{q.shape[0]} Q-tables against {len(automata)} DFAs; one run is one seed bank"
        )
    max_states = max(dfa.num_states for dfa in automata)
    plain = [PlainDFA.from_dfa(dfa, max_states) for dfa in automata]
    return {
        "q_table": q,
        "hyp_delta": np.stack([np.asarray(p.delta) for p in plain]),
        "hyp_accepting": np.stack([np.asarray(p.accepting) for p in plain]),
        "hyp_num_states": np.array([dfa.num_states for dfa in automata], dtype=np.int32),
    }


def tabular_params(policy: FinalPolicy, *, cause_dfa: DFA, cause_known: bool) -> TabularParams:
    """The per-seed parameters of a saved tabular policy, as `tcirl.agents.tabular.TabularPolicy` reads them.

    A Q-learning checkpoint written by release v1.0 holds no DFA: that learner read the cause
    state, when the run observed it, as the last observable component, and its table has an axis of
    size 1 before the actions. Its DFA is then `cause_dfa` when `cause_known`, whose state the
    observed cause axis holds, and otherwise the one-state DFA over the cause's alphabet, whose state
    that axis of size 1 holds.
    """
    q_table = policy.arrays["q_table"]
    if "hyp_delta" in policy.arrays:
        delta = policy.arrays["hyp_delta"]
    else:
        if policy.algo != "q_learning" or q_table.shape[-2] != 1:
            raise ValueError(
                f"a {policy.algo} checkpoint with the table shape {q_table.shape} holds no DFA"
            )
        automaton = cause_dfa if cause_known else DFA.trivial_hypothesis(cause_dfa.num_symbols)
        if cause_known:
            q_table = q_table[..., 0, :]
        delta = np.broadcast_to(
            np.asarray(PlainDFA.from_dfa(automaton, automaton.num_states).delta),
            (policy.num_seeds, automaton.num_states, automaton.num_symbols),
        )
    return TabularParams(q_table=jnp.asarray(q_table), delta=jnp.asarray(delta))


def _leaf_key(path: jax.tree_util.KeyPath) -> str:
    return f"{_DQN_PREFIX}{jax.tree_util.keystr(path)}"


def dqn_arrays(params: nnx.GraphState) -> dict[str, np.ndarray]:
    """The DQN's per-seed parameters, one array per leaf of the state, keyed by the leaf's path.

    A `GraphState` is a pytree whose leaves are the parameter arrays, stacked on the seed axis by
    `init_per_seed_params`, so `jax.tree_util` reads them without any knowledge of the network.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {_leaf_key(path): np.asarray(jax.device_get(leaf)) for path, leaf in leaves}


def dqn_state_from_arrays(
    template: nnx.GraphState, arrays: Mapping[str, np.ndarray]
) -> nnx.GraphState:
    """A state shaped like `template` whose every leaf is the saved array.

    `template` is a fresh `init_per_seed_params(...)` for the same architecture and seed count; it
    supplies the tree structure the flat file does not carry. A leaf the file lacks, an extra key
    it carries, or a leaf of another shape (another width or seed count) is an error rather than a
    silent partial restore, because the graph definition would run the wrong network unnoticed.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    wanted = [_leaf_key(path) for path, _ in leaves]
    missing = [key for key in wanted if key not in arrays]
    extra = [key for key in arrays if key.startswith(_DQN_PREFIX) and key not in wanted]
    if missing or extra:
        raise ValueError(
            f"the saved DQN leaves do not match the template: missing {missing}, extra {extra}"
        )
    mismatched = {
        key: (tuple(arrays[key].shape), tuple(np.shape(leaf)))
        for key, (_, leaf) in zip(wanted, leaves, strict=True)
        if tuple(arrays[key].shape) != tuple(np.shape(leaf))
    }
    if mismatched:
        raise ValueError(
            f"saved leaf shapes differ from the template's (saved, template): {mismatched}"
        )
    restored: nnx.GraphState = jax.tree_util.tree_unflatten(
        treedef, [jax.numpy.asarray(arrays[key]) for key in wanted]
    )
    return restored
