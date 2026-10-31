from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from tcirl.alphabet import OrderedAlphabet

# Action deltas: UP=0, DOWN=1, LEFT=2, RIGHT=3; plain tuples, because a module that builds a JAX array at import time starts a JAX backend.
_DR = (-1, 1, 0, 0)
_DC = (0, 0, -1, 1)

_NUM_ACTIONS = 4

# Per-layer characters. A cell's full description is the union of its characters
# across every layer (see `GridWorld.create`): walls/slip/stick/start are transition
# dynamics; lowercase letters are atomic propositions that accumulate into the cell's
# label set.
_WALL = "X"
_START = "S"
_EMPTY = " "
_SLIP_LOW = "."
_SLIP_HIGH = ":"
_STICK_LOW = ","
_STICK_HIGH = ";"


@dataclass(frozen=True, slots=True, kw_only=True)
class GridWorld:
    """A gridworld MDP with walls, AP-labeled cells, and per-cell slip/stick.

    Each cell carries a *label*: the set of atomic propositions on it (possibly the
    empty set), encoded via an `OrderedAlphabet`. Labels are what the cause DFA reads
    and are distinct from the cell's transition dynamics -- being a wall,
    the slip probability, the stick probability, and being a start cell are dynamics,
    not part of the label.

    `slip_probs` randomizes the action on a move; `stick_probs` keeps the agent in
    place (a geometric "try to leave" RV). Both are per-cell and read from the agent's
    current cell each step. `starts` lists the start cells in row-major order.
    """

    walls: jax.Array
    labels: jax.Array
    slip_probs: jax.Array
    stick_probs: jax.Array
    layers: tuple[tuple[str, ...], ...]
    starts: jax.Array
    height: int
    width: int
    no_label: int

    @classmethod
    def create(
        cls,
        path: Path | str,
        *,
        alphabet: OrderedAlphabet[str],
        slip_prob_low: float = 0.0,
        slip_prob_high: float = 0.0,
        stick_prob_low: float = 0.0,
        stick_prob_high: float = 0.0,
    ) -> GridWorld:
        """Load a layered gridworld from a text file.

        The file holds one or more equal-sized layers separated by a line of `-`
        (length == grid width). A cell's attributes are the union of its characters
        across layers: walls OR together, atomic-proposition letters accumulate into
        the cell's label set, and slip/stick characters select a probability. The
        characters are `X` (wall), `S` (start), ` ` (nothing), `a`-`z` (atomic
        proposition, must be in `alphabet`), `.`/`:` (low/high slip), `,`/`;`
        (low/high stick). Incompatible layers raise `ValueError`. A wall cell may
        also carry slip/stick characters in other layers (inert under a wall, so a
        slip layer can paint over walls for readability); APs and starts may not.
        """
        layers = _split_layers(Path(path).read_text())
        height = len(layers[0])
        width = len(layers[0][0])

        alphabet_props = set(alphabet.props)
        no_label = alphabet.encode([])

        walls_list: list[list[bool]] = []
        labels_list: list[list[int]] = []
        slip_list: list[list[float]] = []
        stick_list: list[list[float]] = []
        starts: list[tuple[int, int]] = []

        slip_of = {_SLIP_LOW: slip_prob_low, _SLIP_HIGH: slip_prob_high}
        stick_of = {_STICK_LOW: stick_prob_low, _STICK_HIGH: stick_prob_high}

        for r in range(height):
            wall_row: list[bool] = []
            label_row: list[int] = []
            slip_row: list[float] = []
            stick_row: list[float] = []
            for c in range(width):
                chars = [layer[r][c] for layer in layers]
                if _WALL in chars:
                    # Slip/stick chars are inert under a wall (the agent can never stand
                    # there), so allow them: a slip layer may paint over walls to make the
                    # dynamics regions readable. APs and starts under a wall stay errors.
                    allowed = (_WALL, _EMPTY, *slip_of, *stick_of)
                    bad = [ch for ch in chars if ch not in allowed]
                    if bad:
                        raise ValueError(
                            f"wall cell at ({r}, {c}) also carries {bad!r}; a wall may only "
                            f"be '{_WALL}', '{_EMPTY}', or a slip/stick dynamics character"
                        )
                    wall_row.append(True)
                    label_row.append(no_label)
                    slip_row.append(0.0)
                    stick_row.append(0.0)
                    continue

                aps: set[str] = set()
                slip_vals: set[float] = set()
                stick_vals: set[float] = set()
                for ch in chars:
                    if ch == _EMPTY:
                        continue
                    if ch == _START:
                        if (r, c) not in starts:
                            starts.append((r, c))
                    elif ch in slip_of:
                        slip_vals.add(slip_of[ch])
                    elif ch in stick_of:
                        stick_vals.add(stick_of[ch])
                    elif ch.islower() and ch.isascii() and ch.isalpha():
                        if ch not in alphabet_props:
                            raise ValueError(
                                f"atomic proposition '{ch}' at ({r}, {c}) is not in the alphabet"
                            )
                        aps.add(ch)
                    else:
                        raise ValueError(f"unknown character '{ch}' at ({r}, {c})")

                if len(slip_vals) > 1:
                    raise ValueError(f"conflicting slip probabilities at ({r}, {c}): {slip_vals}")
                if len(stick_vals) > 1:
                    raise ValueError(f"conflicting stick probabilities at ({r}, {c}): {stick_vals}")

                try:
                    label = alphabet.encode(sorted(aps))
                except KeyError:
                    raise ValueError(
                        f"label {{{', '.join(sorted(aps))}}} at ({r}, {c}) is not in the alphabet"
                    ) from None

                wall_row.append(False)
                label_row.append(label)
                slip_row.append(slip_vals.pop() if slip_vals else 0.0)
                stick_row.append(stick_vals.pop() if stick_vals else 0.0)
            walls_list.append(wall_row)
            labels_list.append(label_row)
            slip_list.append(slip_row)
            stick_list.append(stick_row)

        if not starts:
            raise ValueError("no start position 'S' found in grid")

        return cls(
            walls=jnp.array(walls_list, dtype=jnp.bool_),
            labels=jnp.array(labels_list, dtype=jnp.int32),
            slip_probs=jnp.array(slip_list, dtype=jnp.float32),
            stick_probs=jnp.array(stick_list, dtype=jnp.float32),
            layers=tuple(tuple(layer) for layer in layers),
            starts=jnp.array(sorted(starts), dtype=jnp.int32),
            height=height,
            width=width,
            no_label=no_label,
        )

    @property
    def num_actions(self) -> int:
        return _NUM_ACTIONS


def _split_layers(text: str) -> list[list[str]]:
    """Split a map file into layers (lists of row strings) on `-`-only separator lines."""
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError("empty grid file")

    width = len(lines[0])
    layers: list[list[str]] = [[]]
    for line in lines:
        if line and set(line) == {"-"}:
            if len(line) != width:
                raise ValueError(f"separator line has length {len(line)}, expected {width}")
            layers.append([])
            continue
        if len(line) != width:
            raise ValueError(f"row {line!r} has length {len(line)}, expected {width}")
        layers[-1].append(line)

    layers = [layer for layer in layers if layer]
    height = len(layers[0])
    for li, layer in enumerate(layers):
        if len(layer) != height:
            raise ValueError(f"layer {li} has height {len(layer)}, expected {height}")
    return layers


def gridworld_step(
    gw: GridWorld,
    row: jax.Array,
    col: jax.Array,
    action: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Take a step in the gridworld. Returns (new_row, new_col, label).

    Two independent draws stack: with probability `stick_probs[cell]` the agent stays
    put; otherwise it moves under `action`, which with probability `slip_probs[cell]`
    is replaced by a uniformly random action. Walls and bounds block the move.
    """
    stick_key, slip_key, dir_key = jax.random.split(key, 3)

    did_stick = jax.random.uniform(stick_key) < gw.stick_probs[row, col]
    did_slip = jax.random.uniform(slip_key) < gw.slip_probs[row, col]
    random_action = jax.random.randint(dir_key, (), 0, _NUM_ACTIONS)
    effective_action = jnp.where(did_slip, random_action, action)

    new_row = jnp.clip(row + jnp.asarray(_DR, dtype=jnp.int32)[effective_action], 0, gw.height - 1)
    new_col = jnp.clip(col + jnp.asarray(_DC, dtype=jnp.int32)[effective_action], 0, gw.width - 1)

    hit_wall = gw.walls[new_row, new_col]
    moved_row = jnp.where(hit_wall, row, new_row)
    moved_col = jnp.where(hit_wall, col, new_col)

    final_row = jnp.where(did_stick, row, moved_row)
    final_col = jnp.where(did_stick, col, moved_col)

    label = gw.labels[final_row, final_col]
    return final_row, final_col, label


def render_gridworld(gw: GridWorld, agent_row: int, agent_col: int) -> str:
    """Render every layer with the agent drawn as '@' (agent wins ties over the cell)."""
    sep = "-" * gw.width
    blocks: list[str] = []
    for layer in gw.layers:
        rows: list[str] = []
        for r, row_chars in enumerate(layer):
            cells = list(row_chars)
            if r == agent_row:
                cells[agent_col] = "@"
            rows.append("".join(cells))
        blocks.append("\n".join(rows))
    return ("\n" + sep + "\n").join(blocks)
