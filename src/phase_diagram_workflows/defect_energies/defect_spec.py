"""
Describe a point defect for ``converge_formation_energy`` from an orbit found by the sublattice discovery.

All members of an orbit are equivalent by symmetry, so the first one represents it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ase import Atoms

from phase_diagram_workflows.structures.point_defects import (
    add_pristine,
    create_interstitial,
    create_substitution,
    create_vacancy,
    delta_n_for_interstitial,
    delta_n_for_substitution,
    delta_n_for_vacancy,
)

KINDS = ("vacancy", "substitution", "interstitial")


def defect_from_orbit(orbit: Dict[str, Any], kind: str, element: Optional[str] = None) -> Dict[str, Any]:
    """The defect dict of one representative of `orbit`, with a ``label`` for file names.

    Parameters
    ----------
    orbit : dict
        An entry of ``discover_atomic_sublattices`` (for a vacancy or a substitution) or of
        ``discover_interstitial_sublattices`` (for an interstitial), found on the unit cell.
    kind : {"vacancy", "substitution", "interstitial"}
    element : str, optional
        The element that replaces the atom (substitution) or is inserted (interstitial). Not used for a vacancy.

    Returns
    -------
    dict
        ``{"type": "vacancy", "atom_index": i}``, ``{"type": "substitution", "atom_index": i, "to_element": ...}`` or
        ``{"type": "interstitial", "position": [x, y, z], "element": ...}``, plus ``label``: element first and the
        site after it, as in ``Al_vac_4a``, ``Mg_on_Al_4a`` or ``Mg_i_int_8c``.

    Raises
    ------
    ValueError
        If `kind` is unknown, the orbit is of the wrong sort for it, or a required `element` is missing
        (or, for a substitution, is the atom's own element).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}.")

    if kind == "interstitial":
        if "cart_positions" not in orbit:
            raise ValueError("An interstitial needs an interstitial orbit (one with 'cart_positions').")
        if element is None:
            raise ValueError("An interstitial needs `element`, the atom to insert.")
        return {
            "type": "interstitial",
            "position": [float(x) for x in orbit["cart_positions"][0]],
            "element": element,
            "label": f"{element}_i_{orbit['label']}",
        }

    if "atom_indices" not in orbit:
        raise ValueError(f"A {kind} needs an atomic orbit (one with 'atom_indices').")
    species = orbit["species"]
    site = orbit["label"].split("_", 1)[1]            # "Al_4a" -> "4a", "Al_18f_3" -> "18f_3"
    index = int(orbit["atom_indices"][0])
    if kind == "vacancy":
        return {"type": "vacancy", "atom_index": index, "label": f"{species}_vac_{site}"}
    if element is None:
        raise ValueError("A substitution needs `element`, the atom that replaces the original.")
    if element == species:
        raise ValueError(f"Substituting {species} by {element} is not a defect.")
    return {"type": "substitution", "atom_index": index, "to_element": element, "label": f"{element}_on_{species}_{site}"}


def build_defect(supercell: Atoms, unit_cell: Atoms, defect: Dict[str, Any]) -> Tuple[Atoms, Dict[str, int]]:
    """The supercell with the defect, and the change in atom counts it makes (``delta_n``).

    `defect` is a dict of ``defect_from_orbit``; its atom index or void position is that of `unit_cell` and is placed in
    the first tile of `supercell` (= ``unit_cell.repeat(...)``), which starts at the origin.
    """
    kind = defect.get("type")
    if kind not in KINDS:
        raise ValueError(f"defect['type'] must be one of {KINDS}, got {kind!r}.")
    container = add_pristine(atoms=supercell, unique_id="pristine")

    if kind == "interstitial":
        element = defect["element"]
        container = create_interstitial(container, sublattice=[defect["position"]], element=element, site_ids=[0])
        delta_n = delta_n_for_interstitial(element)
    else:
        index = int(defect["atom_index"])
        host_element = unit_cell[index].symbol   # the atom of the first tile, which has the unit cell's index
        if kind == "vacancy":
            container = create_vacancy(container, atom_ids=[index])
            delta_n = delta_n_for_vacancy(host_element)
        else:
            container = create_substitution(container, to_element=defect["to_element"], atom_ids=[index])
            delta_n = delta_n_for_substitution(host_element, defect["to_element"])
    return container.get_defect_structures()[0]["structure"], delta_n
