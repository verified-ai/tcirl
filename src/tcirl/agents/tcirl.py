"""TCIRL's cause inference, which the shared tabular learner (`tcirl.agents.tabular`) runs on its hypotheses.

Each seed keeps its own hypothesis cause DFA and its own causal sample, a prefix tree of the
episodes that refuted an earlier hypothesis with the marker certificates they carry. Re-inference
is counterexample-driven: once per `stop_interval`, every seed whose hypothesis an episode of the
interval refuted synthesizes a smallest DFA consistent with its whole sample by SAT, and its Q-table
moves onto the new hypothesis by the configured value transfer.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.agents.tabular import TabularCarry, TabularOutputs
from tcirl.alphabet import OrderedAlphabet
from tcirl.automata.dfa import DFA, force_self_loops
from tcirl.config import ValueTransfer
from tcirl.inference.causal_sample import MutableCausalSample
from tcirl.inference.markers import certificates, compress_certificates, is_counterexample
from tcirl.inference.solver import learn_cause_dfa
from tcirl.inference.transfer import kept_update_share, transfer_rows, transfer_sources
from tcirl.learning.trace import filter_symbol_from_trace
from tcirl.logging.wandb import WandbLogger
from tcirl.visualization import DFASummaryInfo


class CauseInference:
    """The per-seed causal samples and SAT re-inference of TCIRL, an `AutomatonInference`.

    `label_projection` maps each environment label to the symbol of the cause alphabet, the
    alphabet of the hypotheses. With `self_loop_symbol`, the search removes that symbol, which the
    cause is declared never to read, from every trace and adds its self-loops back to each
    synthesized hypothesis.

    At a re-inference every update count restarts at 0, because no update has used the new
    hypothesis yet. Under the value transfer rules "exact" and "counterexample" each state of the new
    hypothesis takes the Q-values of an old state (`tcirl.inference.transfer`); under "none" the
    whole table restarts from `q_init`, and the seed's exploration schedule restarts with it, so
    that the new table is learned as the first one was.
    """

    def __init__(
        self,
        *,
        num_runs: int,
        alphabet: OrderedAlphabet[str],
        label_projection: np.ndarray,
        x_r_obs_ids: frozenset[int],
        x_a_obs_ids: frozenset[int],
        self_loop_symbol: int | None,
        max_states: int,
        initial_rejecting: bool,
        symmetry_breaking: bool,
        value_transfer: ValueTransfer,
        q_init: float,
        logger: WandbLogger,
    ) -> None:
        self._alphabet = alphabet
        self._label_projection = label_projection
        self._x_r_obs_ids = x_r_obs_ids
        self._x_a_obs_ids = x_a_obs_ids
        self._self_loop_symbol = self_loop_symbol
        self._max_states = max_states
        self._initial_rejecting = initial_rejecting
        self._symmetry_breaking = symmetry_breaking
        self._value_transfer: ValueTransfer = value_transfer
        self._q_init = q_init
        self._logger = logger
        self._samples = [MutableCausalSample(alphabet.num_symbols) for _ in range(num_runs)]
        # The labels of each seed's episode in progress, which a re-inference replays to find the
        # new hypothesis's state.
        self._pending: list[list[int]] = [[] for _ in range(num_runs)]
        # The seeds whose hypothesis an episode refuted since the last re-inference. A
        # `stop_interval` can span several batches, so this spans them too.
        self._refuted: set[int] = set()
        self._reinferences = [0] * num_runs
        # The share of each seed's update counts before its last re-inference that lie in the rows
        # of the old states whose values some new state took.
        self._kept_updates = [0.0] * num_runs
        self._logged_hypotheses = 0
        self.initial_hypotheses = [DFA.trivial_hypothesis(alphabet.num_symbols)] * num_runs
        self._log_hypothesis(self.initial_hypotheses[0])

    def _log_hypothesis(self, dfa: DFA) -> None:
        """Record a hypothesis of seed 0."""
        self._logger.log_automaton_summary(
            DFASummaryInfo(
                dfa=dfa,
                alphabet=self._alphabet,
                formula_str=None,
                perf=None,
                encoder=None,
                live_states=None,
            ),
            subdir="data/hypotheses",
            filename=f"hyp_{self._logged_hypotheses}.txt",
            render_if=20,
        )
        self._logged_hypotheses += 1

    def observe(self, outputs: TabularOutputs, automata: Sequence[DFA]) -> None:
        labels = np.asarray(outputs.label)
        episode_over = np.asarray(outputs.episode_over)
        num_runs, batch_steps = labels.shape
        for s in range(num_runs):
            for t in range(batch_steps):
                self._pending[s].append(int(labels[s, t]))
                if not episode_over[s, t]:
                    continue
                # The trace holds the labels of the sampled steps, the terminating one included;
                # the self-loops of a terminal state are never sampled, so no label follows it.
                trace = tuple(self._pending[s])
                self._pending[s] = []
                # The certificates come from the full trace, as the theory reads it: a marker
                # certifies the prefix in front of it.
                certs = certificates(trace, self._x_r_obs_ids, self._x_a_obs_ids)
                symbols = tuple(int(self._label_projection[label]) for label in trace)
                # The declared self-loop symbol leaves the cause in its state, so each certificate
                # moves to the compressed form of the prefix it certifies.
                if self._self_loop_symbol is not None:
                    symbols, keep = filter_symbol_from_trace(symbols, self._self_loop_symbol)
                    certs = compress_certificates(certs, keep)
                if is_counterexample(symbols, certs, automata[s]):
                    self._samples[s].add_trace(symbols, certs)
                    self._refuted.add(s)

    def revise(
        self, carry: TabularCarry, automata: list[DFA]
    ) -> tuple[TabularCarry, list[DFA]] | None:
        refuted, self._refuted = sorted(self._refuted), set()
        if not refuted:
            return None
        hypotheses = list(automata)
        q_tables, counts = carry.q_table, carry.update_count
        exploration_start, state = carry.exploration_start, carry.automaton_state
        reset_counts = jnp.zeros(counts.shape[1:], dtype=counts.dtype)
        # The hypothesis axis of one seed's table, after the observable axes.
        axis = q_tables.ndim - 3
        for s in refuted:
            sample = self._samples[s].to_causal_sample()
            result = learn_cause_dfa(
                sample,
                max_states=self._max_states,
                initial_rejecting=self._initial_rejecting,
                symmetry_breaking=self._symmetry_breaking,
            )
            if result is None:
                raise ValueError(
                    f"Failed to learn a consistent cause DFA for seed {s} with "
                    f"max_states={self._max_states}."
                )
            if self._self_loop_symbol is not None:
                result = force_self_loops(result, self._self_loop_symbol)
            self._reinferences[s] += 1
            sources = transfer_sources(hypotheses[s], result, sample, self._value_transfer)
            if sources:
                # States beyond the new hypothesis start from `q_init`.
                self._kept_updates[s] = kept_update_share(counts[s], sources, axis=axis)
                q_tables = q_tables.at[s].set(
                    transfer_rows(
                        q_tables[s],
                        sources,
                        num_states=result.num_states,
                        axis=axis,
                        fill=self._q_init,
                    )
                )
            else:
                self._kept_updates[s] = 0.0
                q_tables = q_tables.at[s].set(jnp.full_like(q_tables[s], self._q_init))
                exploration_start = exploration_start.at[s].set(carry.step[s])
            counts = counts.at[s].set(reset_counts)
            hypotheses[s] = result
            replayed = 0
            for label in self._pending[s]:
                replayed = result.step(replayed, int(self._label_projection[label]))
            state = state.at[s].set(replayed)
            if s == 0:
                self._log_hypothesis(result)
        return (
            carry._replace(
                q_table=q_tables,
                update_count=counts,
                exploration_start=exploration_start,
                automaton_state=state,
            ),
            hypotheses,
        )

    def metrics(self, automata: Sequence[DFA]) -> dict[str, jax.Array]:
        return {
            "inference/traces": jnp.array(
                [sample.num_traces for sample in self._samples], dtype=jnp.float32
            ),
            "inference/hypothesis_states": jnp.array(
                [dfa.num_states for dfa in automata], dtype=jnp.float32
            ),
            "inference/reinferences": jnp.array(self._reinferences, dtype=jnp.float32),
            "inference/kept_updates": jnp.array(self._kept_updates, dtype=jnp.float32),
        }
