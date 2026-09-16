"""
Vacancy, substitution, and interstitial creation on top of a StructureContainer.

Each defect type has two entry points:

- `create_vacancy` / `create_substitution` / `create_interstitial` -- add one
  defect structure on top of a single parent. Selection is either explicit
  (`atom_ids=`/`site_ids=`, exact sites) or random (`n=` sites, optionally
  `seed=`-reproducible). Exactly one of the two selection modes must be given.
- `create_vacancy_batch` / `create_substitution_batch` / `create_interstitial_batch`
  -- apply the same operation across multiple existing parent structures
  (`target_indices`) in the container, fanning out into one or several new
  structures per parent.

This merges what was originally twelve separate functions (a `_from_ids` and
a `_from_seed` variant of each of the six above) into six: the `_from_ids`
and `_from_seed` bodies differed only in how the target site(s) were chosen,
with the rest of each function (parent resolution, event recording, storing
the result) identical, so picking the selection mode is now a parameter
rather than a separate function name.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from .container import (
    UID_KEY,
    StructureContainer,
    _protected_uids_from_events,
    _resolve_parent,
    append_atom_with_uid,
    ensure_uids,
    next_uid,
    validate_atoms_arrays,
)


def _parent_kwargs(container: StructureContainer, parent_idx: int) -> dict:
    """
    Translate an absolute row index into the `parent_defect_index`/`input_structure`
    kwargs the singular `create_*` functions expect: a defect index if that row is
    itself a defect, else the pristine structure object (`_resolve_parent` only accepts
    a defect index or a structure, not a pristine row index directly).
    """
    entry = container._structures[parent_idx]
    if entry["is_pristine"]:
        return {"parent_defect_index": None, "input_structure": entry["structure"]}
    return {"parent_defect_index": parent_idx, "input_structure": None}


# ============================================================================
# Vacancies
# ============================================================================


def create_vacancy(
    structure_container: StructureContainer,
    atom_ids: Optional[List[int]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    vacancy_element: Optional[Union[str, List[str]]] = None,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create one or more vacancies, either at explicit sites or randomly sampled.

    Exactly one of `atom_ids` (explicit) or `n` (random) must be given.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    atom_ids : list of int or None
        Explicit atom indices to remove. Mutually exclusive with `n`.
    n : int or None
        Number of vacancies to create by random sampling. Mutually exclusive
        with `atom_ids`.
    seed : int or None
        Random seed for reproducibility (random mode only).
    vacancy_element : str, list of str, or None
        Random mode only. Element to remove (e.g. 'Al'); `None` samples any
        element. Pass a list to fix the per-vacancy elements (e.g.
        `['Al', 'Mg']` removes one Al and one Mg); when a list is given, `n`
        must equal its length.
    parent_defect_index : int or None
        Index of a defect structure to build on (`None` = use default parent).
    input_structure : Atoms or None
        Structure to use as parent (`None` = use container default).
    forbid_uids : list of int or None
        UIDs to exclude from removal.
    protect_history : bool
        If True, also forbid every UID touched by a prior defect event on
        this structure's lineage.

    Returns
    -------
    StructureContainer with the new vacancy structure added.

    Raises
    ------
    ValueError
        If neither or both of `atom_ids`/`n` are given, or (random mode) if
        there aren't enough candidates for the request.
    IndexError
        If any `atom_ids` value is out of range (explicit mode).
    """
    if (atom_ids is None) == (n is None):
        raise ValueError("Provide exactly one of atom_ids (explicit sites) or n (random selection).")

    container = structure_container
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)
    atoms = ensure_uids(parent["structure"]).copy()
    existing_events = parent["events"].copy()
    pristine_idx = container._find_pristine_index(parent_idx)

    validate_atoms_arrays(atoms)
    uids = atoms.arrays[UID_KEY].astype(int)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)

    forbid = set() if forbid_uids is None else set(map(int, forbid_uids))
    if protect_history:
        forbid |= _protected_uids_from_events(existing_events)

    if atom_ids is not None:
        for idx in atom_ids:
            if not (0 <= idx < len(atoms)):
                raise IndexError(f"Vacancy index {idx} is out of range for a structure with {len(atoms)} atoms.")
        pick_idx = [idx for idx in atom_ids if int(uids[idx]) not in forbid]
        if len(pick_idx) == 0:
            raise ValueError("No valid indices after applying forbid_uids filters.")
    else:
        rng = np.random.default_rng(seed if seed is not None else 0)
        if isinstance(vacancy_element, list):
            if n != len(vacancy_element):
                raise ValueError(f"n={n} must equal len(vacancy_element)={len(vacancy_element)} when vacancy_element is a list")
            pick_idx = []
            used: set = set()
            for elem in vacancy_element:
                cand = [i for i in range(len(atoms)) if syms[i] == elem and int(uids[i]) not in forbid and i not in used]
                if len(cand) == 0:
                    raise ValueError(f"No candidates left for element '{elem}'")
                chosen = int(rng.choice(cand))
                pick_idx.append(chosen)
                used.add(chosen)
        else:
            if vacancy_element is None:
                cand = [i for i in range(len(atoms)) if int(uids[i]) not in forbid]
            else:
                cand = [i for i in range(len(atoms)) if syms[i] == vacancy_element and int(uids[i]) not in forbid]
            if len(cand) < n:
                raise ValueError(f"Not enough candidates: need {n}, have {len(cand)}")
            pick_idx = rng.choice(cand, size=int(n), replace=False).tolist()

    new_events = [
        {
            "type": "vacancy",
            "removed_element": str(syms[i]),
            "site_uid": int(uids[i]),
            "site_pos0": atoms.positions[i].tolist(),
            "pos_at_removal": atoms.positions[i].tolist(),
        }
        for i in pick_idx
    ]

    for i in sorted(map(int, pick_idx), reverse=True):
        del atoms[i]
    validate_atoms_arrays(atoms)

    operation_str = f"vacancy[{len(pick_idx)}]" if len(pick_idx) > 1 else f"vacancy[{new_events[0]['site_uid']}]"

    metadata = {"parent_index": parent_idx, "pristine_index": pristine_idx}
    if atom_ids is None:
        metadata["seed"] = seed

    container.add_defect(
        atoms=atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=existing_events + new_events,
        metadata=metadata,
    )
    return container


def create_vacancy_batch(
    structure_container: StructureContainer,
    target_indices: List[int],
    atom_ids: Optional[List[int]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    vacancy_element: Optional[Union[str, List[str]]] = None,
    separate_structures: bool = True,
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
    n_structures: int = 1,
) -> StructureContainer:
    """
    Apply `create_vacancy` across multiple existing parent structures.

    Exactly one of `atom_ids` (explicit) or `n` (random) must be given, as in
    `create_vacancy`.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    target_indices : list of int
        Absolute indices of parent structures to modify.
    atom_ids : list of int or None
        Explicit mode: atom indices to remove for each target.
    n : int or None
        Random mode: number of vacancies to create per structure.
    seed : int or None
        Random mode: base random seed; each generated structure uses a
        distinct, incrementing seed (`seed + i`).
    vacancy_element : str, list of str, or None
        Random mode only; see `create_vacancy`.
    separate_structures : bool
        Explicit mode only. True (default): one new structure per atom id
        per target. False: one new structure per target containing every
        `atom_ids` vacancy at once.
    forbid_uids : list of int or None
        UIDs to exclude from removal.
    protect_history : bool
        See `create_vacancy`.
    n_structures : int
        Random mode only. Number of structures to generate from each parent
        (default 1), each with its own incrementing seed.

    Returns
    -------
    StructureContainer with all new structures added.

    See Also
    --------
    create_vacancy : Single-parent version; see its docstring for the
        explicit/random selection modes.
    """
    if (atom_ids is None) == (n is None):
        raise ValueError("Provide exactly one of atom_ids (explicit sites) or n (random selection).")

    container = structure_container

    if atom_ids is not None:
        for parent_idx in target_indices:
            pk = _parent_kwargs(container, parent_idx)
            id_groups = [[aid] for aid in atom_ids] if separate_structures else [atom_ids]
            for ids in id_groups:
                container = create_vacancy(
                    structure_container=container,
                    atom_ids=ids,
                    forbid_uids=forbid_uids,
                    protect_history=protect_history,
                    **pk,
                )
    else:
        structure_counter = 0
        for parent_idx in target_indices:
            pk = _parent_kwargs(container, parent_idx)
            for _ in range(n_structures):
                structure_seed = seed + structure_counter if seed is not None else None
                container = create_vacancy(
                    structure_container=container,
                    n=n,
                    seed=structure_seed,
                    vacancy_element=vacancy_element,
                    forbid_uids=forbid_uids,
                    protect_history=protect_history,
                    **pk,
                )
                structure_counter += 1

    return container


# ============================================================================
# Substitutions
# ============================================================================


def create_substitution(
    structure_container: StructureContainer,
    to_element: str,
    atom_ids: Optional[List[int]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    from_element: Optional[str] = None,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
) -> StructureContainer:
    """
    Create one or more substitutions, either at explicit sites or randomly sampled.

    Exactly one of `atom_ids` (explicit) or `n` (random) must be given.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    to_element : str
        Element to substitute with.
    atom_ids : list of int or None
        Explicit atom indices to substitute (the element being replaced is
        read from the structure). Mutually exclusive with `n`.
    n : int or None
        Number of substitutions to create by random sampling. Requires
        `from_element`. Mutually exclusive with `atom_ids`.
    seed : int or None
        Random seed for reproducibility (random mode only).
    from_element : str or None
        Random mode only, required: element to replace.
    parent_defect_index : int or None
        Index of a defect structure to build on (`None` = use default parent).
    input_structure : Atoms or None
        Structure to use as parent (`None` = use container default).
    forbid_uids : list of int or None
        UIDs to exclude from substitution.
    protect_history : bool
        If True, also forbid every UID touched by a prior defect event on
        this structure's lineage.

    Returns
    -------
    StructureContainer with the new substitution structure added.

    Raises
    ------
    ValueError
        If neither or both of `atom_ids`/`n` are given, if `n` is given
        without `from_element`, or if there aren't enough candidates.
    IndexError
        If any `atom_ids` value is out of range (explicit mode).
    """
    if (atom_ids is None) == (n is None):
        raise ValueError("Provide exactly one of atom_ids (explicit sites) or n (random selection).")
    if n is not None and from_element is None:
        raise ValueError("from_element is required when selecting substitution sites randomly (n=...).")

    container = structure_container
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)
    atoms = ensure_uids(parent["structure"]).copy()
    existing_events = parent["events"].copy()
    pristine_idx = container._find_pristine_index(parent_idx)

    uids = atoms.arrays[UID_KEY].astype(int)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)

    forbid = set() if forbid_uids is None else set(map(int, forbid_uids))
    if protect_history:
        forbid |= _protected_uids_from_events(existing_events)

    if atom_ids is not None:
        for idx in atom_ids:
            if not (0 <= idx < len(atoms)):
                raise IndexError(f"Substitution index {idx} is out of range for a structure with {len(atoms)} atoms.")
        pick_idx = [idx for idx in atom_ids if int(uids[idx]) not in forbid]
        if len(pick_idx) == 0:
            raise ValueError("No valid indices after applying forbid_uids filters.")
    else:
        rng = np.random.default_rng(seed if seed is not None else 0)
        cand = [i for i in range(len(atoms)) if syms[i] == from_element and int(uids[i]) not in forbid]
        if n > len(cand):
            raise ValueError(f"Not enough candidates to substitute {from_element}->{to_element}: need {n}, have {len(cand)}")
        pick_idx = rng.choice(cand, size=int(n), replace=False).tolist()

    new_events = [
        {
            "type": "substitution",
            "from": str(syms[i]),
            "to": str(to_element),
            "atom_uid": int(uids[i]),
            "site_uid": int(uids[i]),
            "site_pos0": atoms.positions[i].tolist(),
            "pos_at_creation": atoms.positions[i].tolist(),
        }
        for i in pick_idx
    ]

    for i in pick_idx:
        atoms[i].symbol = to_element
    validate_atoms_arrays(atoms)

    if len(pick_idx) > 1:
        operation_str = f"substitution[{len(pick_idx)}:{new_events[0]['from']}->{to_element}]"
    else:
        operation_str = f"substitution[{new_events[0]['from']}->{to_element}]"

    metadata = {"parent_index": parent_idx, "pristine_index": pristine_idx}
    if atom_ids is None:
        metadata["seed"] = seed

    container.add_defect(
        atoms=atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=existing_events + new_events,
        metadata=metadata,
    )
    return container


def create_substitution_batch(
    structure_container: StructureContainer,
    target_indices: List[int],
    to_element: str,
    atom_ids: Optional[List[int]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    from_element: Optional[str] = None,
    separate_structures: bool = True,
    forbid_uids: Optional[List[int]] = None,
    protect_history: bool = False,
    n_structures: int = 1,
) -> StructureContainer:
    """
    Apply `create_substitution` across multiple existing parent structures.

    Exactly one of `atom_ids` (explicit) or `n` (random, requires
    `from_element`) must be given, as in `create_substitution`.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    target_indices : list of int
        Absolute indices of parent structures to modify.
    to_element : str
        Element to substitute with.
    atom_ids : list of int or None
        Explicit mode: atom indices to substitute for each target.
    n : int or None
        Random mode: number of substitutions to create per structure.
    seed : int or None
        Random mode: base random seed; each generated structure uses a
        distinct, incrementing seed (`seed + i`).
    from_element : str or None
        Random mode only, required: element to replace.
    separate_structures : bool
        Explicit mode only. True (default): one new structure per atom id
        per target. False: one new structure per target containing every
        `atom_ids` substitution at once.
    forbid_uids : list of int or None
        UIDs to exclude from substitution.
    protect_history : bool
        See `create_substitution`.
    n_structures : int
        Random mode only. Number of structures to generate from each parent.

    Returns
    -------
    StructureContainer with all new structures added.

    See Also
    --------
    create_substitution : Single-parent version; see its docstring for the
        explicit/random selection modes.
    """
    if (atom_ids is None) == (n is None):
        raise ValueError("Provide exactly one of atom_ids (explicit sites) or n (random selection).")

    container = structure_container

    if atom_ids is not None:
        for parent_idx in target_indices:
            pk = _parent_kwargs(container, parent_idx)
            id_groups = [[aid] for aid in atom_ids] if separate_structures else [atom_ids]
            for ids in id_groups:
                container = create_substitution(
                    structure_container=container,
                    atom_ids=ids,
                    to_element=to_element,
                    forbid_uids=forbid_uids,
                    protect_history=protect_history,
                    **pk,
                )
    else:
        structure_counter = 0
        for parent_idx in target_indices:
            pk = _parent_kwargs(container, parent_idx)
            for _ in range(n_structures):
                structure_seed = seed + structure_counter if seed is not None else None
                container = create_substitution(
                    structure_container=container,
                    n=n,
                    to_element=to_element,
                    from_element=from_element,
                    seed=structure_seed,
                    forbid_uids=forbid_uids,
                    protect_history=protect_history,
                    **pk,
                )
                structure_counter += 1

    return container


# ============================================================================
# Interstitials
# ============================================================================


def create_interstitial(
    structure_container: StructureContainer,
    sublattice: "np.ndarray",
    element: str,
    site_ids: Optional[List[int]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
) -> StructureContainer:
    """
    Create one or more interstitials from a candidate sublattice, either at
    explicit sites or randomly sampled.

    Exactly one of `site_ids` (explicit) or `n` (random) must be given.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    sublattice : (N, 3) array-like
        Cartesian coordinates (angstrom) of candidate interstitial sites --
        e.g. the output of `get_voronoi_interstitial_sites`.
    element : str
        Chemical symbol of the atom to insert (e.g. 'Mg').
    site_ids : list of int or None
        Explicit indices into `sublattice` to occupy. Mutually exclusive with `n`.
    n : int or None
        Number of interstitial atoms to insert by random sampling (without
        replacement) from `sublattice`. Mutually exclusive with `site_ids`.
    seed : int or None
        Random seed for reproducibility (random mode only).
    parent_defect_index : int or None
        Index of a defect structure to build on (`None` = use pristine).
    input_structure : Atoms or None
        Explicit parent structure (alternative to `parent_defect_index`).

    Returns
    -------
    StructureContainer with the new interstitial structure added.

    Raises
    ------
    ValueError
        If neither or both of `site_ids`/`n` are given, if `sublattice`
        doesn't have shape (N, 3), or if there aren't enough candidate sites.
    IndexError
        If any `site_ids` value is out of range for `sublattice`.
    """
    if (site_ids is None) == (n is None):
        raise ValueError("Provide exactly one of site_ids (explicit sites) or n (random selection).")

    container = structure_container
    parent_idx = _resolve_parent(container, parent_defect_index, input_structure)
    parent = container.get_structure(parent_idx)
    atoms = ensure_uids(parent["structure"]).copy()
    existing_events = parent["events"].copy()
    pristine_idx = container._find_pristine_index(parent_idx)

    sublattice_arr = np.asarray(sublattice, float)
    if sublattice_arr.ndim != 2 or sublattice_arr.shape[1] != 3:
        raise ValueError(f"sublattice must have shape (N, 3), got {sublattice_arr.shape}.")
    validate_atoms_arrays(atoms)

    if site_ids is not None:
        if len(site_ids) == 0:
            raise ValueError("site_ids is empty -- provide at least one site index.")
        for sid in site_ids:
            if not (0 <= sid < len(sublattice_arr)):
                raise IndexError(f"site_id {sid} is out of range for sublattice with {len(sublattice_arr)} sites.")
        picked_ids = list(site_ids)
    else:
        if len(sublattice_arr) == 0:
            raise ValueError("sublattice is empty -- no candidate sites to sample from.")
        if n > len(sublattice_arr):
            raise ValueError(f"Requested n={n} interstitials but sublattice only has {len(sublattice_arr)} sites.")
        rng = np.random.default_rng(seed if seed is not None else 0)
        picked_ids = rng.choice(len(sublattice_arr), size=int(n), replace=False).tolist()

    new_events = []
    for sid in picked_ids:
        pos = sublattice_arr[int(sid)].tolist()
        new_uid = next_uid(atoms)
        atoms = append_atom_with_uid(atoms, element, pos)
        validate_atoms_arrays(atoms)
        new_events.append(
            {
                "type": "interstitial",
                "element": str(element),
                "atom_uid": int(new_uid),
                "pos0": pos,
                "site_label": f"site_{int(sid)}",
            }
        )

    n_inserted = len(picked_ids)
    operation_str = f"interstitial[{n_inserted}:{element}]" if n_inserted > 1 else f"interstitial[{element}]"

    metadata = {"parent_index": parent_idx, "pristine_index": pristine_idx}
    if site_ids is None:
        metadata["seed"] = seed

    container.add_defect(
        atoms=atoms,
        operation=operation_str,
        pristine_index=pristine_idx,
        parent_index=parent_idx,
        events=existing_events + new_events,
        metadata=metadata,
    )
    return container


def create_interstitial_batch(
    structure_container: StructureContainer,
    target_indices: List[int],
    sublattice: "np.ndarray",
    element: str,
    site_ids: Optional[List[int]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    separate_structures: bool = True,
    n_structures: int = 1,
) -> StructureContainer:
    """
    Apply `create_interstitial` across multiple existing parent structures.

    Mode is selected by `n`: if given, sites are randomly sampled per
    generated structure (as in `create_interstitial`'s random mode). If not,
    `site_ids` selects explicit sites -- defaulting to *every* site in
    `sublattice` when `site_ids` is also omitted.

    Parameters
    ----------
    structure_container : StructureContainer
        The container with structures to modify.
    target_indices : list of int
        Absolute indices of parent structures to modify.
    sublattice : (N, 3) array-like
        Cartesian coordinates (angstrom) of candidate interstitial sites.
    element : str
        Chemical symbol of the atom to insert.
    site_ids : list of int or None
        Explicit mode only. Indices into `sublattice` to use; `None`
        (default) uses every site.
    n : int or None
        Random mode: number of interstitial atoms per generated structure.
    seed : int or None
        Random mode: base random seed; each generated structure uses a
        distinct, incrementing seed (`seed + i`).
    separate_structures : bool
        Explicit mode only. True (default): one new structure per site per
        parent. False: one new structure per parent containing every
        selected site at once.
    n_structures : int
        Random mode only. Number of structures to generate from each parent.

    Returns
    -------
    StructureContainer with all new structures added.

    See Also
    --------
    create_interstitial : Single-parent version.
    """
    container = structure_container
    sublattice_arr = np.asarray(sublattice, float)

    if n is not None:
        if site_ids is not None:
            raise ValueError("Provide at most one of site_ids (explicit sites) or n (random selection).")
        structure_counter = 0
        for parent_idx in target_indices:
            pk = _parent_kwargs(container, parent_idx)
            for _ in range(n_structures):
                structure_seed = seed + structure_counter if seed is not None else None
                container = create_interstitial(
                    structure_container=container,
                    sublattice=sublattice_arr,
                    element=element,
                    n=n,
                    seed=structure_seed,
                    **pk,
                )
                structure_counter += 1
    else:
        effective_ids = list(range(len(sublattice_arr))) if site_ids is None else list(site_ids)
        for parent_idx in target_indices:
            pk = _parent_kwargs(container, parent_idx)
            id_groups = [[sid] for sid in effective_ids] if separate_structures else [effective_ids]
            for ids in id_groups:
                container = create_interstitial(
                    structure_container=container,
                    sublattice=sublattice_arr,
                    element=element,
                    site_ids=ids,
                    **pk,
                )

    return container
