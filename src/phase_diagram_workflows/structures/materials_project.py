from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from ase import Atoms


def _get_element_fractions(atoms: Atoms, elements: Sequence[str]) -> Dict[str, float]:
    """Compute the fraction of each element among a fixed set of elements.

    Restricted to `elements` (rather than every species present) so that a
    structure with incidental extra species still yields fractions that sum
    to 1 over the elements of interest.

    Parameters
    ----------
    atoms : Atoms
        Structure to analyze.
    elements : Sequence[str]
        Elements to compute fractions for, e.g. ["Al", "Mg"].

    Returns
    -------
    Dict[str, float]
        Mapping from each element in `elements` to its fraction.

    Raises
    ------
    ValueError
        If none of `elements` are present in `atoms`.
    """
    symbols = atoms.get_chemical_symbols()
    counts = {el: symbols.count(el) for el in elements}
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"No atoms of the specified elements {list(elements)} found.")
    return {el: count / total for el, count in counts.items()}


def get_materials_project_df(
    chemsys: str,
    api_key: str,
    include_pure: bool = True,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch all Materials Project entries for a chemical system.

    Parameters
    ----------
    chemsys : str
        Chemical system, e.g. "Al-Mg".
    api_key : str
        Materials Project API key. Passed through explicitly rather than
        read from the environment, so callers control credential handling.
    include_pure : bool
        If True, also fetch the pure single-element systems making up
        `chemsys` (e.g. "Al", "Mg"), so mixing energies can be anchored to
        them later.
    fields : Optional[List[str]]
        Fields to request from the MP API. None requests all fields.

    Returns
    -------
    pd.DataFrame
        One row per material, deduplicated by material_id.
    """
    from mp_api.client import MPRester

    systems = [chemsys]
    if include_pure:
        systems += chemsys.split("-")
    systems = sorted(set(systems))

    docs_all = []
    with MPRester(api_key) as mpr:
        for system in systems:
            docs = mpr.materials.summary.search(chemsys=system, fields=fields)
            docs_all.extend(docs)

    rows = [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in docs_all]
    df = pd.DataFrame(rows)

    if "material_id" in df.columns:
        df = df.drop_duplicates(subset="material_id").reset_index(drop=True)

    return df


def build_structures_dataframe(mp_df: pd.DataFrame, elements: Sequence[str]) -> pd.DataFrame:
    """Convert a raw Materials Project dataframe into a structure-ready dataframe.

    Adds ASE Atoms, crystal system / space group, and per-element composition
    fractions, so the result can be fed directly into energy calculation and
    convex-hull analysis (`phase_diagram_workflows.phase_diagram.convex_hull`).

    Parameters
    ----------
    mp_df : pd.DataFrame
        Output of `get_materials_project_df`; must contain 'material_id',
        'structure' (pymatgen Structure.as_dict()), and 'symmetry' columns.
    elements : Sequence[str]
        The two elements defining the composition axis, e.g. ["Al", "Mg"].
        The fraction of `elements[1]` becomes the 'x' column.

    Returns
    -------
    pd.DataFrame
        `mp_df` plus columns: structure_ase, crystal_system, space_group,
        material_id_underscore, fractions, x.

    Raises
    ------
    ValueError
        If `elements` does not contain exactly 2 elements.
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    if len(elements) != 2:
        raise ValueError(f"build_structures_dataframe expects exactly 2 elements, got {list(elements)}")

    df = mp_df.copy()
    df["structure_ase"] = df["structure"].apply(lambda s: AseAtomsAdaptor.get_atoms(Structure.from_dict(s)))
    df["crystal_system"] = df["symmetry"].apply(lambda s: s["crystal_system"] if s else None)
    df["space_group"] = df["symmetry"].apply(lambda s: s["symbol"] if s else None)
    df["material_id_underscore"] = df["material_id"].apply(lambda s: s.replace("-", "_"))
    df["fractions"] = df["structure_ase"].apply(lambda atoms: _get_element_fractions(atoms, elements))
    df["x"] = df["fractions"].apply(lambda f: f[elements[1]])

    return df


def append_structure(
    df: pd.DataFrame,
    atoms: Atoms,
    elements: Sequence[str],
    material_id: str,
    formula_pretty: Optional[str] = None,
    **extra_columns: Any,
) -> pd.DataFrame:
    """Append an externally sourced structure to a structures dataframe.

    Useful for structures that are not indexed by Materials Project (e.g. a
    hand-built supercell read from a file), so they still participate in
    energy calculation and convex-hull analysis alongside MP entries.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe produced by `build_structures_dataframe` (or with a
        compatible schema).
    atoms : Atoms
        The structure to append.
    elements : Sequence[str]
        Same two elements used to build `df`'s composition axis.
    material_id : str
        Identifier for the new row; also used, with '-' replaced by '_',
        for material_id_underscore.
    formula_pretty : Optional[str]
        Chemical formula label. Defaults to `atoms.get_chemical_formula()`.
    **extra_columns
        Additional column values to set on the new row (e.g.
        energy_per_atom), for columns not otherwise populated here.

    Returns
    -------
    pd.DataFrame
        `df` with the new row appended.
    """
    fractions = _get_element_fractions(atoms, elements)
    row = {
        "material_id": material_id,
        "material_id_underscore": material_id.replace("-", "_"),
        "formula_pretty": formula_pretty or atoms.get_chemical_formula(),
        "structure_ase": atoms,
        "crystal_system": None,
        "space_group": None,
        "fractions": fractions,
        "x": fractions[elements[1]],
        **extra_columns,
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
