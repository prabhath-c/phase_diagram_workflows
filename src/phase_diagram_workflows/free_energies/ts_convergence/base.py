from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def ts_overlap_criterion(
    forward_energy_diff: Sequence[float],
    backward_energy_diff: Sequence[float],
) -> float:
    """Compute the forward/backward hysteresis of a reversible-scaling pair.

    Compares the first point of the forward integration path against the
    last point of the backward integration path. This is the only one of
    calphy's two hysteresis criteria that is actually used to decide
    convergence. Kept separate from `check_ts_overlap` so the raw value can
    be computed once and judged against different tolerances later, e.g.
    when building a criterion table to plot and tune the threshold on.

    Parameters
    ----------
    forward_energy_diff : Sequence[float]
        Energy difference array for one forward TI repetition (e.g.
        ``df["forward_energy_diff"].iloc[0][0]``).
    backward_energy_diff : Sequence[float]
        Energy difference array for the corresponding backward TI repetition.

    Returns
    -------
    float
        Absolute mismatch, in the same units as the energy difference arrays.
    """
    return abs(backward_energy_diff[-1] - forward_energy_diff[0])


def check_ts_overlap(
    forward_energy_diff: Sequence[float],
    backward_energy_diff: Sequence[float],
    tolerance: float,
) -> bool:
    """Check whether a reversible-scaling forward/backward pair overlaps.

    Parameters
    ----------
    forward_energy_diff : Sequence[float]
        Energy difference array for one forward TI repetition (e.g.
        ``df["forward_energy_diff"].iloc[0][0]``).
    backward_energy_diff : Sequence[float]
        Energy difference array for the corresponding backward TI repetition.
    tolerance : float
        Maximum allowed absolute mismatch, in the same units as the energy
        difference arrays.

    Returns
    -------
    bool
        True if the forward/backward paths overlap within tolerance.
    """
    return bool(ts_overlap_criterion(forward_energy_diff, backward_energy_diff) <= tolerance)


def step_bracket(
    t_low: float,
    t_high: float,
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
) -> Tuple[float, float]:
    """Narrow a temperature bracket by independently stepping either bound.

    Parameters
    ----------
    t_low : float
        Current lower temperature bound.
    t_high : float
        Current upper temperature bound.
    step_lower : Optional[float], optional
        If given, raise the lower bound by this amount.
    step_upper : Optional[float], optional
        If given, lower the upper bound by this amount.

    Returns
    -------
    Tuple[float, float]
        The narrowed ``(t_low, t_high)`` bracket.

    Raises
    ------
    ValueError
        If neither step is given, or if the resulting bracket is empty or
        inverted.
    """
    if step_lower is None and step_upper is None:
        raise ValueError("At least one of step_lower or step_upper must be given.")

    new_t_low = t_low + step_lower if step_lower is not None else t_low
    new_t_high = t_high - step_upper if step_upper is not None else t_high

    if new_t_low >= new_t_high:
        raise ValueError(
            f"Stepping the bracket collapsed it: ({t_low}, {t_high}) -> "
            f"({new_t_low}, {new_t_high})"
        )

    return new_t_low, new_t_high


def resolve_current_bracket(
    initial_bracket: Tuple[float, float],
    tried_brackets: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Recompute the current temperature bracket from prior attempts.

    Stateless by design: rather than storing which bound has been narrowed,
    the current bracket is always the most restrictive lower bound and the
    most restrictive upper bound seen across the initial bracket and every
    previously tried bracket. A bound that has never been stepped simply
    never appears more restrictive than the initial value.

    Parameters
    ----------
    initial_bracket : Tuple[float, float]
        The ``(t_low, t_high)`` bracket a sweep started from.
    tried_brackets : List[Tuple[float, float]]
        Every ``(t_low, t_high)`` bracket already attempted, in any order,
        however they were discovered (e.g. by scanning working directory
        names for a matching prefix).

    Returns
    -------
    Tuple[float, float]
        The current ``(t_low, t_high)`` bracket.

    Raises
    ------
    ValueError
        If the lower and upper bounds were narrowed independently, in
        separate attempts, far enough that combining their most restrictive
        values produces an inverted/empty bracket that no single attempt
        actually ran. Raised here rather than silently returned, since the
        caller (`refine_temperature_bracket`) submits this bracket for real.
    """
    lows = [initial_bracket[0]] + [b[0] for b in tried_brackets]
    highs = [initial_bracket[1]] + [b[1] for b in tried_brackets]
    t_low, t_high = max(lows), min(highs)

    if t_low >= t_high:
        raise ValueError(
            f"Combining independently narrowed bounds produced an invalid "
            f"bracket ({t_low}, {t_high}); the lower and upper bounds were "
            f"narrowed past each other across different attempts: {tried_brackets}"
        )

    return t_low, t_high
