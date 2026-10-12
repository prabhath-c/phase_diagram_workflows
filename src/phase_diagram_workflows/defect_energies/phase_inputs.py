"""
Read what the earlier steps of a phase saved: the relaxed unit cell with the energies per atom of the pure
elements, and the symmetry-unique sites. Plain functions of two file paths, so no layout of any notebook is built in.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ase import Atoms


@dataclass(frozen=True)
class PhaseInputs:
    """``unit_cell``: the relaxed unit cell; ``mu0``: energy per atom of each pure element (eV);
    ``atomic`` and ``interstitial``: the symmetry-unique orbits found on the unit cell; ``potential``: the
    potential the saved results are from."""

    unit_cell: Atoms
    mu0: Dict[str, float]
    atomic: List[dict]
    interstitial: List[dict]
    potential: Optional[str]


def load_phase_inputs(
    pristine_pickle: Union[str, Path], sublattices_pickle: Union[str, Path], phase: str, potential: Optional[str] = None
) -> PhaseInputs:
    """Inputs of `phase` from the pickle with the relaxed unit cells and the pickle with their sublattices.

    The pristine pickle needs ``table`` (row per phase, with a ``structure`` column) and ``mu0_<element>`` entries; the
    sublattices pickle needs ``sublattices[phase]["atomic" | "interstitial"]``. If `potential` is given, both pickles
    must have been made with it (their ``potential`` entry), so results of two potentials cannot be mixed.

    Raises
    ------
    ValueError
        If `potential` is given and a pickle is from another one.
    KeyError
        If `phase` is not in the pickles.
    """
    with open(pristine_pickle, "rb") as handle:
        pristine = pickle.load(handle)
    with open(sublattices_pickle, "rb") as handle:
        sublattices = pickle.load(handle)
    if potential is not None:
        for name, found in (("pristine", pristine), ("sublattices", sublattices)):
            if found.get("potential") != potential:
                raise ValueError(f"the {name} pickle is from {found.get('potential')!r}, not from {potential!r}.")
    found = sublattices["sublattices"][phase]
    return PhaseInputs(
        unit_cell=pristine["table"].loc[phase, "structure"],
        mu0={key[len("mu0_"):]: float(value) for key, value in pristine.items() if key.startswith("mu0_")},
        atomic=found["atomic"],
        interstitial=found["interstitial"],
        potential=pristine.get("potential"),
    )
