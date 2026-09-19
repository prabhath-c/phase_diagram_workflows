"""
Point-defect formation energy bookkeeping.

Ported from the (previously inlined, duplicated across every
`all_{vacancies,substitutionals,interstitials}.ipynb`) "Formation energies and
save" cell -- the chemical-potential arithmetic is identical for every defect
type given `E_defect`, `E_pristine`, and how many atoms of each element the
defect added or removed (`delta_n`); only *how* `delta_n` is derived differs
per type, which is what the small `delta_n_for_*` helpers below are for.

Element-agnostic but still a two-element (binary-system) formula: one element
is `pinned_element` (its chemical potential held at its elemental reference,
`mu0[pinned_element]`), the other is `swept_element` (the one the returned
`slope`/`intercept` describe a formation-energy sweep over, via
`Δμ_swept = mu_swept - mu0[swept_element]`). Which two symbols these are is
entirely up to the caller (e.g. "Al"/"Mg" today, "Ti"/"Al" tomorrow) --
nothing here is hardcoded to a specific pair.

Two related but distinct linear forms are computed together, both from the
same `E_defect`/`E_pristine`/`delta_n`/`mu0` inputs -- see
`compute_formation_energy`'s docstring for the full derivation:

- `slope`/`slope_pinned`/`intercept`: this package's own 0K/dilute-limit
  convention (both sweep directions -- sweeping `swept_element` with
  `pinned_element` held fixed, and the mirror case).
- `landau`: the convention a `landau.phases.pointdefects.PointDefectedPhase`
  needs, where `Δμ ≡ mu_swept - mu_pinned` is anchored to the *floating*
  `mu_pinned` of whichever phase is actually stable at each point of a T-c
  diagram, not to the fixed `mu0[pinned_element]`. Computing this alongside
  the rest costs nothing extra (no additional energy calculation, just
  different bookkeeping of the same two numbers) and is required to
  eventually feed a defect-formation-energy record into `landau`. Grouped
  under its own `landau` key in the returned record so it's unambiguous
  which fields are landau's input and which are this package's own.
"""

from __future__ import annotations

from typing import Any, Dict


def delta_n_for_vacancy(removed_element: str) -> Dict[str, int]:
    """`{removed_element: -1}` for removing one atom of `removed_element`."""
    return {removed_element: -1}


def delta_n_for_substitution(from_element: str, to_element: str) -> Dict[str, int]:
    """`{from_element: -1, to_element: +1}` for substituting `from_element` -> `to_element`.

    If `from_element == to_element` the two contributions cancel, returning `{}`
    (no-op substitution -- not a real defect, but handled consistently rather
    than silently reporting a spurious +1).
    """
    delta: Dict[str, int] = {}
    delta[to_element] = delta.get(to_element, 0) + 1
    delta[from_element] = delta.get(from_element, 0) - 1
    return {el: n for el, n in delta.items() if n != 0}


def delta_n_for_interstitial(element: str) -> Dict[str, int]:
    """`{element: +1}` for inserting one interstitial atom of `element`."""
    return {element: 1}


def compute_formation_energy(
    E_defect: float,
    E_pristine: float,
    delta_n: Dict[str, int],
    mu0: Dict[str, float],
    pinned_element: str,
    swept_element: str,
) -> Dict[str, Any]:
    """
    Compute a point defect's formation-energy record from its relaxed energy
    and its per-element atom-count change relative to the pristine host.

    The general formula is::

        E_form = E_defect - E_pristine - delta_n[pinned_element] * mu_pinned
                                        - delta_n[swept_element] * mu_swept

    Evaluated two ways, both linear in a single swept chemical potential:

    1. **This package's convention** (`slope`, `intercept`, `slope_pinned`):
       `mu_pinned` held at its elemental reference `mu0[pinned_element]`,
       sweeping `Δμ_swept = mu_swept - mu0[swept_element]`::

           E_form(Δμ_swept) = intercept + slope * Δμ_swept
           slope     = -delta_n[swept_element]
           intercept = E_defect - E_pristine - delta_n[pinned_element] * mu0[pinned_element]
                                              - delta_n[swept_element] * mu0[swept_element]

       `slope_pinned = -delta_n[pinned_element]` is the mirror convention
       (sweeping `Δμ_pinned` instead, `mu_swept` held fixed) -- same
       `intercept`, since both sweeps start from the same point (every
       element at its own `mu0`).

    2. **landau's convention** (`landau["intercept"]`): landau's
       `Δμ ≡ mu_swept - mu_pinned` is anchored to the *floating*
       `mu_pinned` of whichever phase is stable at each point of a T-c
       diagram, not to the fixed `mu0[pinned_element]`::

           landau["intercept"] = E_defect - E_pristine - delta_n_total * mu0[pinned_element]
           delta_n_total = delta_n[pinned_element] + delta_n[swept_element]

       landau's slope is identical to `slope` above (no separate value).
       This is **exact** when `delta_n_total == 0` (a swap -- substitution/
       antisite defects: the `mu_pinned`-dependence cancels algebraically,
       independent of where `mu_pinned` actually sits elsewhere in the
       diagram). For vacancies/interstitials (`delta_n_total != 0`) it
       carries the same `mu_pinned ≈ mu0[pinned_element]` approximation
       `intercept` always makes -- nothing worse, just not exact.
       `landau["exact"]` flags which case a given record is, so nothing
       downstream has to guess.

    Parameters
    ----------
    E_defect : float
        Relaxed total energy of the defect structure.
    E_pristine : float
        Relaxed total energy of the pristine host (same size/potential).
    delta_n : dict of str to int
        Change in atom count per element, e.g. `{"Ti": -1}` (vacancy) or
        `{"Al": -1, "Mg": 1}` (substitution). Every key must be
        `pinned_element` and/or `swept_element` -- this is a two-element
        formula; a `delta_n` entry for anything else is rejected rather than
        silently dropped, which would otherwise omit that atom's
        chemical-potential contribution from the result without warning.
        Elements with zero change may simply be omitted.
    mu0 : dict of str to float
        Elemental reference chemical potential (energy per atom of the pure
        elemental phase) for `pinned_element` and `swept_element`.
    pinned_element : str
        The element whose chemical potential is held at its elemental
        reference for the primary `slope`/`intercept`, and that landau's
        `Δμ` is measured from.
    swept_element : str
        The other element -- the one `slope`/`intercept` describe a sweep
        over, and that landau's `Δμ` is measured towards.

    Returns
    -------
    dict with keys:
        `E_defect`, `delta_n` (as given), `pinned_element`, `swept_element`,
        `delta_n_pinned`, `delta_n_swept`, `delta_n_total`, `mu0_pinned`,
        `mu0_swept`, `slope`, `slope_pinned`, `intercept`, and a nested
        `landau` dict with `intercept` and `exact`.

    Raises
    ------
    ValueError
        If `delta_n` has a key other than `pinned_element`/`swept_element`.
    KeyError
        If `mu0` is missing an entry for `pinned_element` or `swept_element`.
    """
    unexpected = set(delta_n) - {pinned_element, swept_element}
    if unexpected:
        raise ValueError(
            f"delta_n has element(s) {sorted(unexpected)} that are neither pinned_element "
            f"({pinned_element!r}) nor swept_element ({swept_element!r}) -- "
            f"compute_formation_energy is a two-element formula, so every delta_n key must "
            f"be one of the two, or its chemical-potential contribution would be silently "
            f"omitted from the result."
        )
    for el in (pinned_element, swept_element):
        if el not in mu0:
            raise KeyError(f"mu0 is missing an entry for {el!r} (pinned_element/swept_element).")

    delta_n_pinned = delta_n.get(pinned_element, 0)
    delta_n_swept = delta_n.get(swept_element, 0)
    mu0_pinned = mu0[pinned_element]
    mu0_swept = mu0[swept_element]
    delta_n_total = delta_n_pinned + delta_n_swept

    slope = -delta_n_swept
    slope_pinned = -delta_n_pinned
    intercept = E_defect - E_pristine - delta_n_pinned * mu0_pinned - delta_n_swept * mu0_swept

    landau_intercept = E_defect - E_pristine - delta_n_total * mu0_pinned
    landau_exact = delta_n_total == 0

    return {
        "E_defect": E_defect,
        "delta_n": dict(delta_n),
        "pinned_element": pinned_element,
        "swept_element": swept_element,
        "delta_n_pinned": delta_n_pinned,
        "delta_n_swept": delta_n_swept,
        "delta_n_total": delta_n_total,
        "mu0_pinned": mu0_pinned,
        "mu0_swept": mu0_swept,
        "slope": slope,
        "slope_pinned": slope_pinned,
        "intercept": intercept,
        "landau": {
            "intercept": landau_intercept,
            "exact": landau_exact,
        },
    }
