"""Traffic signal control environment: 3-intersection pipeline x hidden cause DFA.

Model: 3 intersections (q0, q1, q2) in a left-to-right pipeline with a
single agent-controlled red light.  The agent picks which intersection is
blocked; the other two are green.  Cars enter only at q0 and exit only from
q2.

Full MDP state (q0, q1, q2, blocked, cooldown, cause_state, obs_label):
  qi = queue length at intersection i, in {0..q_max}, starting at queue_init
  blocked = agent-selected red-light intersection, in {0, 1, 2}
  cooldown = steps to hold, in {0..cooldown_max}: cooldown_max after an advance, 0 before the next
  cause_state = hidden cause DFA state
  obs_label = most recent combined observation label (hidden bookkeeping)

Observable dimensions come before hidden ones in EnvState.indices.
  cause_known=True  -> state_shape = (q_max+1, q_max+1, q_max+1, 3, cooldown_max+1, |C|)
  cause_known=False -> state_shape = (q_max+1, q_max+1, q_max+1, 3, cooldown_max+1)

Actions: 0 = hold, 1 = advance.

Hard constraint (the mandatory hold, a cooldown timer):
  Advance while cooldown > 0 => immediate termination with reward 0.
  cooldown_max = 1, so the fastest legal cycle is advance-hold-advance-hold
  (period 2).

Soft constraint (cause DFA):
  The cause formula F(h,a,h,a,h,a) fires exactly when the agent executes
  three consecutive hold-advance pairs without inserting an extra hold.
  The advance-hold policy triggers it after 6 steps.  To avoid the cause,
  the agent must break the pattern by holding twice in a row
  (e.g. h,a,h,a,h,h,a,...), sacrificing one advance opportunity.

Step mechanics:
  1. Termination check: advance while cooldown > 0 => terminated.
  2. Update blocked and cooldown.
     Advance: blocked = (blocked+1) % 3, cooldown = cooldown_max.
     Hold:    cooldown = max(0, cooldown-1).
  3. Cause DFA step on action label.
  4. Traffic flow at green intersections, downstream first, from the queues before the step:
     For each GREEN intersection i a draw X_i offers cars to move on:
       REJECT: X_i ~ Geom(p_rej_depart).
       ACCEPT: X_i ~ Bernoulli(p_acc_depart).
     Blocked intersection: no flow.
     D = min(X_2, q2) cars leave the system from q2.
     min(X_1, q1, room of q2 after D) cars move from q1 to q2.
     min(X_0, q0, room of q1 after its outflow) cars move from q0 to q1.
  5. New arrivals at q0 (leftmost only), up to the room of q0 after its outflow:
     REJECT: a draw ~ Bernoulli(p_rej_arrive).  At most 1 car.
     ACCEPT: a draw ~ Geom(p_acc_arrive).  Can be 0, 1, 2, ...
     A = the cars that enter q0.
  Geom(p) is the number of consecutive Bernoulli(p) successes before the first failure, capped at
  q_max, which is the number of failures before the first success of Bernoulli(1 - p) trials.
  No car is lost and no queue exceeds q_max, so the queues before and after a step, with the
  red light after it, determine A, D and every flow by conservation of cars.
  6. Dense reward: 1.0 - max(q0,q1,q2) / q_max, or 0.0 on termination.
     The max-based reward penalises the worst bottleneck, not the average.
     Any static block position drives the queue before it to q_max (reward 0).
  7. Combined obs label: action + arrival/departure indicator, a function of the transition.
     multi_arr iff A >= 2 (ACCEPT-exclusive).
     multi_dep iff D >= 2 (REJECT-exclusive).
     Otherwise: base label (ambiguous).

Phase dynamics summary:
  REJECT (cause DFA not accepting):
    - Departures: Geom per green intersection -> high throughput, D>=2 possible.
    - Arrivals: Bernoulli -> at most 1 new car, A>=2 impossible.
  ACCEPT (cause DFA accepting, absorbing):
    - Departures: Bernoulli per green intersection -> at most 1 exit, D>=2 impossible.
    - Arrivals: Geom -> can burst multiple cars, A>=2 possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.alphabet import OrderedAlphabet
from tcirl.automata.dfa import DFA, validate_cause_dfa
from tcirl.rl.common.tabular import EnvState
from tcirl.specifications.traffic_signal import TrafficSignalSpec, build_cause_formula

if TYPE_CHECKING:
    # Only for the annotation: `tcirl.config` imports this module for its validator.
    from tcirl.config import TrafficSignalConfig

# The state has exactly three queue axes, and the step moves cars q0 -> q1 -> q2 -> exit.
NUM_INTERSECTIONS = 3

# Action indices.
HOLD = 0
ADVANCE = 1


def _geom_count(key: jax.Array, p: float, max_trials: int) -> jax.Array:
    """Count consecutive Bernoulli(p) successes, up to max_trials."""
    uniforms = jax.random.uniform(key, (max_trials,))
    successes = (uniforms < p).astype(jnp.int32)
    return jnp.sum(jnp.cumprod(successes))


def _draw_cars(key: jax.Array, p: float, *, bursty: bool, q_max: int) -> jax.Array:
    """Cars moving in one step: a geometric count when the phase bursts, else a Bernoulli.

    `bursty` is a Python bool on the environment, so the choice resolves when the step is traced
    and neither draw is computed twice.
    """
    if bursty:
        return _geom_count(key, p, q_max)
    return (jax.random.uniform(key) < p).astype(jnp.int32)


# Which of the four flow probabilities draws a geometric COUNT rather than a Bernoulli.
# Phase 1 bursts its departures and drips its arrivals, and Phase 2 does the opposite.
# That pairing makes the two markers phase-exclusive, because `multi_dep` needs two departures in a step and `multi_arr` needs two arrivals.
BURSTY_DRAWS: frozenset[str] = frozenset({"p_rej_depart", "p_acc_arrive"})


def validate_traffic_signal_parameters(
    *,
    p_rej_depart: float,
    p_rej_arrive: float,
    p_acc_depart: float,
    p_acc_arrive: float,
    q_max: int,
    cooldown_max: int,
    queue_init: int,
) -> None:
    """Refuse a numerical parameter outside the range the pipeline dynamics need."""
    # A geometric count with p == 1 never stops, so a bursty flow probability is open at one, and a Bernoulli admits one.
    for name, value in (
        ("p_rej_depart", p_rej_depart),
        ("p_rej_arrive", p_rej_arrive),
        ("p_acc_depart", p_acc_depart),
        ("p_acc_arrive", p_acc_arrive),
    ):
        if name in BURSTY_DRAWS:
            if not (0 < value < 1):
                raise ValueError(
                    f"{name} draws a geometric count here, so it must be in (0, 1), got {value}"
                )
        elif not (0 < value <= 1):
            raise ValueError(f"{name} draws a Bernoulli here, so it must be in (0, 1], got {value}")
    # Every draw is capped at q_max cars, and both markers need two cars in one step.
    if q_max < 2:
        raise ValueError(f"q_max must be >= 2, or neither marker can fire, got {q_max}")
    if cooldown_max < 1:
        raise ValueError(f"cooldown_max must be >= 1, got {cooldown_max}")
    if not (0 <= queue_init <= q_max):
        raise ValueError(f"queue_init must be in [0, q_max={q_max}], got {queue_init}")


@dataclass(frozen=True, slots=True, kw_only=True)
class TrafficSignalEnv:
    """3-intersection pipeline x hidden-cause-DFA with dense reward."""

    cause_dfa: DFA
    action_alphabet: OrderedAlphabet[str]
    obs_alphabet: OrderedAlphabet[str]
    cause_known: bool
    cause_dfa_size: int
    q_max: int
    cooldown_max: int
    queue_init: int
    p_rej_depart: float
    p_rej_arrive: float
    p_acc_depart: float
    p_acc_arrive: float
    # One action-alphabet symbol per action index: hold reads the empty symbol, advance the other.
    action_labels: jax.Array  # (num_actions,) int32
    # Precomputed JAX arrays
    cause_delta: jax.Array  # (num_cause_states, action_num_symbols) int32
    cause_accepting: jax.Array  # (num_cause_states,) bool
    # Label mapping arrays for combined obs labels
    label_with_multi_arr: jax.Array  # (action_num_symbols,) int32
    label_with_multi_dep: jax.Array  # (action_num_symbols,) int32
    label_base_obs: jax.Array  # (action_num_symbols,) int32
    combined_to_action: jax.Array  # (obs_num_symbols,) int32

    @classmethod
    def create(
        cls,
        *,
        cause_dfa: DFA,
        action_alphabet: OrderedAlphabet[str],
        obs_alphabet: OrderedAlphabet[str],
        cause_known: bool,
        q_max: int,
        cooldown_max: int,
        queue_init: int,
        p_rej_depart: float,
        p_rej_arrive: float,
        p_acc_depart: float,
        p_acc_arrive: float,
        action_to_obs_base: dict[int, int],
        action_to_obs_multi_arr: dict[int, int],
        action_to_obs_multi_dep: dict[int, int],
        combined_to_action_map: dict[int, int],
    ) -> TrafficSignalEnv:
        """Build a TrafficSignalEnv from a pre-built cause DFA and alphabets."""
        validate_traffic_signal_parameters(
            p_rej_depart=p_rej_depart,
            p_rej_arrive=p_rej_arrive,
            p_acc_depart=p_acc_depart,
            p_acc_arrive=p_acc_arrive,
            q_max=q_max,
            cooldown_max=cooldown_max,
            queue_init=queue_init,
        )

        validate_cause_dfa(cause_dfa)

        # Precompute cause DFA transition table as JAX array.
        cause_table = np.zeros((cause_dfa.num_states, cause_dfa.num_symbols), dtype=np.int32)
        for (state, symbol), next_state in cause_dfa.transitions.items():
            cause_table[state, symbol] = next_state
        cause_delta_arr = jnp.array(cause_table, dtype=jnp.int32)

        cause_accepting_np = np.array(
            [s in cause_dfa.accepting_states for s in range(cause_dfa.num_states)]
        )
        cause_accepting_arr = jnp.array(cause_accepting_np, dtype=jnp.bool_)

        # Precompute label mapping arrays.
        action_n = action_alphabet.num_symbols
        obs_n = obs_alphabet.num_symbols

        multi_arr_a = np.zeros(action_n, dtype=np.int32)
        multi_dep_a = np.zeros(action_n, dtype=np.int32)
        base_arr = np.zeros(action_n, dtype=np.int32)
        for aid in range(action_n):
            multi_arr_a[aid] = action_to_obs_multi_arr[aid]
            multi_dep_a[aid] = action_to_obs_multi_dep[aid]
            base_arr[aid] = action_to_obs_base[aid]

        c2a_arr = np.zeros(obs_n, dtype=np.int32)
        for oid in range(obs_n):
            c2a_arr[oid] = combined_to_action_map[oid]

        labels = [
            action_alphabet.encode(frozenset()),
            action_alphabet.encode(frozenset({"advance"})),
        ]
        action_labels = jnp.asarray(labels, dtype=jnp.int32)

        return cls(
            cause_dfa=cause_dfa,
            action_alphabet=action_alphabet,
            obs_alphabet=obs_alphabet,
            cause_known=cause_known,
            cause_dfa_size=cause_dfa.num_states,
            q_max=q_max,
            cooldown_max=cooldown_max,
            queue_init=queue_init,
            p_rej_depart=p_rej_depart,
            p_rej_arrive=p_rej_arrive,
            p_acc_depart=p_acc_depart,
            p_acc_arrive=p_acc_arrive,
            action_labels=action_labels,
            cause_delta=cause_delta_arr,
            cause_accepting=cause_accepting_arr,
            label_with_multi_arr=jnp.array(multi_arr_a, dtype=jnp.int32),
            label_with_multi_dep=jnp.array(multi_dep_a, dtype=jnp.int32),
            label_base_obs=jnp.array(base_arr, dtype=jnp.int32),
            combined_to_action=jnp.array(c2a_arr, dtype=jnp.int32),
        )

    @classmethod
    def from_config(
        cls, config: TrafficSignalConfig, spec: TrafficSignalSpec, *, cause_known: bool
    ) -> TrafficSignalEnv:
        """Build the environment that a config's `env` section describes, over the cause and alphabets of `spec`."""
        if spec.cause_formula != build_cause_formula(config.difficulty):
            raise ValueError(
                f"spec holds the cause {spec.cause_formula}, but the config's difficulty "
                f"{config.difficulty} names {build_cause_formula(config.difficulty)}"
            )
        return cls.create(
            cause_dfa=spec.cause_dfa,
            action_alphabet=spec.action_alphabet,
            obs_alphabet=spec.obs_alphabet,
            cause_known=cause_known,
            q_max=config.q_max,
            cooldown_max=config.cooldown_max,
            queue_init=config.queue_init,
            p_rej_depart=config.p_rej_depart,
            p_rej_arrive=config.p_rej_arrive,
            p_acc_depart=config.p_acc_depart,
            p_acc_arrive=config.p_acc_arrive,
            action_to_obs_base=spec.action_to_obs_base,
            action_to_obs_multi_arr=spec.action_to_obs_multi_arr,
            action_to_obs_multi_dep=spec.action_to_obs_multi_dep,
            combined_to_action_map=spec.combined_to_action,
        )

    @property
    def name(self) -> str:
        return "traffic_signal"

    @property
    def cause_index(self) -> int:
        """Position of the cause DFA state in EnvState.indices."""
        return 5

    @property
    def q0_index(self) -> int:
        return 0

    @property
    def q1_index(self) -> int:
        return 1

    @property
    def q2_index(self) -> int:
        return 2

    @property
    def blocked_index(self) -> int:
        return 3

    @property
    def cooldown_index(self) -> int:
        return 4

    @property
    def obs_label_index(self) -> int:
        """Position of the observation label in EnvState.indices."""
        return 6

    @property
    def state_shape(self) -> tuple[int, ...]:
        """Observable state dimensions for Q-table indexing."""
        q_levels = self.q_max + 1
        cd_levels = self.cooldown_max + 1
        if self.cause_known:
            return (q_levels, q_levels, q_levels, NUM_INTERSECTIONS, cd_levels, self.cause_dfa_size)
        return (q_levels, q_levels, q_levels, NUM_INTERSECTIONS, cd_levels)

    @property
    def num_actions(self) -> int:
        return 2

    def reset(self, key: jax.Array) -> EnvState:
        del key
        q0 = jnp.int32(self.queue_init)
        q1 = jnp.int32(self.queue_init)
        q2 = jnp.int32(self.queue_init)
        blocked = jnp.int32(0)
        cooldown = jnp.int32(0)
        cause = jnp.int32(0)
        initial_label = jnp.int32(self.obs_alphabet.encode(frozenset()))
        indices = (q0, q1, q2, blocked, cooldown, cause, initial_label)
        return EnvState(
            indices=indices,
            step_count=jnp.int32(0),
            label=initial_label,
        )

    def render_map(self) -> str:
        return self.render(self.reset(jax.random.key(0)))

    def step(
        self, state: EnvState, action: jax.Array, key: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        q0, q1, q2 = state.indices[0], state.indices[1], state.indices[2]
        blocked = state.indices[self.blocked_index]
        cooldown = state.indices[self.cooldown_index]
        cause = state.indices[self.cause_index]

        is_advance = action != HOLD

        # 1. Termination: advance while cooldown > 0.
        terminated = jnp.bool_(is_advance & (cooldown > 0))

        # 2. Update blocked and cooldown.
        new_blocked = jnp.asarray(jnp.where(is_advance, (blocked + 1) % NUM_INTERSECTIONS, blocked))
        new_cooldown = jnp.asarray(
            jnp.where(
                is_advance,
                jnp.int32(self.cooldown_max),
                jnp.maximum(cooldown - 1, jnp.int32(0)),
            )
        )

        # 3. Cause DFA step on action label.
        action_label = jnp.asarray(self.action_labels[action])
        # Phase is determined by the CURRENT cause state (before DFA step).
        # Transition probabilities p_phase(q)(s, a) depend on q, not q'.
        in_accepting = self.cause_accepting[cause]
        new_cause = self.cause_delta[cause, action_label]

        # 4. Traffic flow at green intersections, downstream first.
        # REJECT: Geom(p_rej_depart) cars offered at each green light (high throughput).
        # ACCEPT: Bernoulli(p_acc_depart) -- at most 1 car (bottleneck).
        # A car moves only out of the queue it was in before the step and only into room, so no car is lost.
        is_green_0 = new_blocked != 0
        is_green_1 = new_blocked != 1
        is_green_2 = new_blocked != 2

        k0r, k0a, k1r, k1a, k2r, k2a, k_arr_r, k_arr_a = jax.random.split(key, 8)

        def offered(rej_key: jax.Array, acc_key: jax.Array) -> jax.Array:
            """The cars a green light offers to move on: the draw of the active phase."""
            rej = _draw_cars(
                rej_key, self.p_rej_depart, bursty="p_rej_depart" in BURSTY_DRAWS, q_max=self.q_max
            )
            acc = _draw_cars(
                acc_key, self.p_acc_depart, bursty="p_acc_depart" in BURSTY_DRAWS, q_max=self.q_max
            )
            return jnp.asarray(jnp.where(in_accepting, acc, rej))

        q_max = jnp.int32(self.q_max)
        zero = jnp.int32(0)
        # Intersection 2: q2 -> exit (departures D); the exit always has room.
        flow_2 = jnp.asarray(jnp.where(is_green_2, jnp.minimum(offered(k2r, k2a), q2), zero))
        # Intersection 1: q1 -> q2, into the room that q2 has after its departures.
        flow_1 = jnp.minimum(jnp.minimum(offered(k1r, k1a), q1), q_max - (q2 - flow_2))
        flow_1 = jnp.asarray(jnp.where(is_green_1, flow_1, zero))
        # Intersection 0: q0 -> q1, into the room that q1 has after its outflow.
        flow_0 = jnp.minimum(jnp.minimum(offered(k0r, k0a), q0), q_max - (q1 - flow_1))
        flow_0 = jnp.asarray(jnp.where(is_green_0, flow_0, zero))

        departures = flow_2  # D = cars exiting the system

        # 5. New arrivals at q0 (leftmost only), into the room that q0 has after its outflow.
        rej_arrivals = _draw_cars(
            k_arr_r, self.p_rej_arrive, bursty="p_rej_arrive" in BURSTY_DRAWS, q_max=self.q_max
        )
        acc_arrivals = _draw_cars(
            k_arr_a, self.p_acc_arrive, bursty="p_acc_arrive" in BURSTY_DRAWS, q_max=self.q_max
        )
        drawn_arrivals = jnp.where(in_accepting, acc_arrivals, rej_arrivals)
        arrivals = jnp.asarray(
            jnp.minimum(drawn_arrivals, q_max - (q0 - flow_0))
        )  # A = cars that enter q0

        new_q0 = q0 - flow_0 + arrivals
        new_q1 = q1 - flow_1 + flow_0
        new_q2 = q2 - flow_2 + flow_1

        # 6. Reward: penalises the worst bottleneck (max queue), not the sum.
        # This makes any static block position terrible (blocked queue -> max
        # -> reward 0) and incentivises cycling all three intersections.
        max_queue = jnp.maximum(new_q0, jnp.maximum(new_q1, new_q2))
        dense_reward = jnp.float32(1.0) - max_queue.astype(jnp.float32) / jnp.float32(self.q_max)
        reward = jnp.asarray(jnp.where(terminated, jnp.float32(0.0), dense_reward))

        # 7. Combined obs label: action + multi_arr / multi_dep / base.
        combined_label = jnp.asarray(
            jnp.where(
                arrivals >= 2,
                self.label_with_multi_arr[action_label],
                jnp.where(
                    departures >= 2,
                    self.label_with_multi_dep[action_label],
                    self.label_base_obs[action_label],
                ),
            )
        )

        new_indices = (
            new_q0,
            new_q1,
            new_q2,
            new_blocked,
            new_cooldown,
            new_cause,
            combined_label,
        )
        new_state = EnvState(
            indices=new_indices,
            step_count=state.step_count + 1,
            label=combined_label,
        )
        return new_state, reward, terminated

    def action_mask(self, state: EnvState) -> jax.Array:
        # An illegal advance ends the episode instead of being masked, so every action is available.
        del state
        return jnp.ones(self.num_actions, dtype=jnp.bool_)

    def render(self, state: EnvState) -> str:
        q0 = int(state.indices[self.q0_index])
        q1 = int(state.indices[self.q1_index])
        q2 = int(state.indices[self.q2_index])
        blocked = int(state.indices[self.blocked_index])
        cooldown = int(state.indices[self.cooldown_index])
        cause = int(state.indices[self.cause_index])
        return (
            "traffic signal MDP\n"
            f"  queues = [{q0}, {q1}, {q2}] (max {self.q_max})\n"
            f"  blocked (red) = intersection {blocked}\n"
            f"  cooldown = {cooldown}/{self.cooldown_max}\n"
            f"  cause = {cause}"
        )
