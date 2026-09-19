"""
StructureContainer: storage and lineage tracking for pristine and point-defect
structures.

Ported from the notebook module used to build the single-defect phase
diagrams (Al-Mg FCC/HCP/Gamma/Epsilon/Beta), where a `StructureContainer`
accumulates a pristine reference structure plus every defect derived from it,
tracking parent/pristine lineage and the full chronological event history
needed to reconstruct how each structure was built.

Two parallel APIs are provided for everything the container can do:
`StructureContainer` methods, and module-level free functions that take the
container as their first argument (e.g. `add_pristine(container, atoms)`
instead of `container.add_pristine(atoms)`). The free functions are the ones
actually imported and used throughout the source notebooks -- they exist so
the container's behavior can be driven without holding a reference to the
instance being mutated in place, which matters for notebook cells and for
wrapping these functions as workflow nodes later.
"""

from __future__ import annotations

import fnmatch
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from ase import Atoms

UID_KEY = "uid"


# ============================================================================
# UID Helper Functions
# ============================================================================


def ensure_uids(atoms: Atoms, uid_key: str = UID_KEY) -> Atoms:
    """Attach stable per-atom uids if missing. Does not modify existing uids."""
    if uid_key in atoms.arrays:
        return atoms
    atoms = atoms.copy()
    atoms.arrays[uid_key] = np.arange(len(atoms), dtype=int)
    return atoms


def next_uid(atoms: Atoms, uid_key: str = UID_KEY) -> int:
    """Return a fresh uid greater than all existing ones."""
    if uid_key not in atoms.arrays or len(atoms) == 0:
        return 0
    return int(np.max(atoms.arrays[uid_key])) + 1


def uid_to_index(atoms: Atoms, uid: int, uid_key: str = UID_KEY) -> Optional[int]:
    """Return the current atom index for a given uid, or None if it no longer exists."""
    if uid_key not in atoms.arrays:
        return None
    hits = np.where(atoms.arrays[uid_key] == int(uid))[0]
    return int(hits[0]) if len(hits) else None


def element_uids(atoms: Atoms, element: str, uid_key: str = UID_KEY) -> List[int]:
    """Get all UIDs currently belonging to a given element."""
    atoms = ensure_uids(atoms, uid_key=uid_key)
    syms = np.array(atoms.get_chemical_symbols(), dtype=object)
    uids = atoms.arrays[uid_key].astype(int)
    return uids[syms == element].tolist()


def validate_atoms_arrays(atoms: Atoms, uid_key: str = UID_KEY) -> None:
    """Validate that all per-atom arrays (positions, uids, ...) have consistent length."""
    n = len(atoms)
    bad = {k: v.shape[0] for k, v in atoms.arrays.items() if v.shape[0] != n}
    if bad:
        raise ValueError(f"Inconsistent per-atom arrays: len(atoms)={n}, bad={bad}")


def append_atom_with_uid(atoms: Atoms, symbol: str, position, uid_key: str = UID_KEY) -> Atoms:
    """Append a new atom with a fresh UID and return the updated structure."""
    atoms = ensure_uids(atoms, uid_key=uid_key).copy()
    new_id = next_uid(atoms, uid_key=uid_key)
    atoms.append(Atoms(symbols=[symbol], positions=[position])[0])

    u = atoms.arrays[uid_key]
    if u.shape[0] == len(atoms) - 1:
        atoms.arrays[uid_key] = np.append(u, new_id).astype(int)
    elif u.shape[0] == len(atoms):
        atoms.arrays[uid_key][-1] = new_id
    else:
        raise ValueError(f"uid array has unexpected length {u.shape[0]} for len(atoms)={len(atoms)}")

    return atoms


def _protected_uids_from_events(events: Optional[List[dict]]) -> set:
    """UIDs of atoms explicitly created/modified by prior defect events (for `protect_history`)."""
    forbid = set()
    for ev in events or []:
        t = ev.get("type")
        if t == "substitution":
            if "atom_uid" in ev:
                forbid.add(int(ev["atom_uid"]))
            elif "site_uid" in ev:
                forbid.add(int(ev["site_uid"]))
        elif t == "interstitial":
            if "atom_uid" in ev:
                forbid.add(int(ev["atom_uid"]))
    return forbid


def validate_structure(atoms: Atoms, min_distance: float = 0.5) -> bool:
    """
    Validate a structure for common issues like atoms too close together.

    Parameters
    ----------
    atoms : Atoms
        The structure to validate.
    min_distance : float
        Minimum allowed interatomic distance in Angstroms.

    Returns
    -------
    bool
        True if the structure is valid.

    Raises
    ------
    ValueError
        If any two atoms are closer than `min_distance`.
    """
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            dist = atoms.get_distance(i, j, mic=True)
            if dist < min_distance:
                raise ValueError(
                    f"Structure validation failed: Atoms {i} and {j} are too close "
                    f"({dist:.3f} < {min_distance:.3f} Å). "
                    f"This may cause numerical issues in calculations."
                )
    return True


def get_stoichiometry(atoms: Atoms) -> str:
    """
    Chemical formula string from an Atoms object, e.g. "Al107Mg1".

    Examples
    --------
    >>> from ase.build import bulk
    >>> get_stoichiometry(bulk('Al', cubic=True))
    'Al4'
    """
    counts = Counter(atoms.get_chemical_symbols())
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def make_operations_short(events: List[dict]) -> str:
    """
    Pipe-separated short form of an event list, e.g.
    "vacancy[5]|substitution[Al->Mg]".
    """
    if not events:
        return "no_operations"

    short_ops = []
    for ev in events:
        t = ev.get("type")
        if t == "vacancy":
            uid = ev.get("site_uid", "?")
            short_ops.append(f"vacancy[{uid}]")
        elif t == "substitution":
            from_el = ev.get("from", "?")
            to_el = ev.get("to", "?")
            short_ops.append(f"substitution[{from_el}->{to_el}]")
        elif t == "interstitial":
            el = ev.get("element", "?")
            short_ops.append(f"interstitial[{el}]")

    return "|".join(short_ops) if short_ops else "no_operations"


# ============================================================================
# StructureContainer
# ============================================================================


@dataclass
class StructureContainer:
    """
    Container for pristine and defect structures with full lineage tracking.

    Each structure is stored as a dict in `_structures` with the following
    fields:

    Core data
        `structure` (Atoms), `unique_id` (str), `creation_timestamp` (datetime)
    Lineage (absolute row indices into `_structures`)
        `pristine_structure_index`, `parent_index`, `generation`
        (0 for pristine, +1 per defect step from its parent)
    Classification
        `is_pristine` (bool), `stoichiometry` (str)
    Operation description
        `operation` (full description), `operations_short`
        (pipe-separated, e.g. "vacancy[5]|substitution[10->Mg]")
    History
        `events`: complete chronological list of defect operations, each a
        dict with at least a `type` key (`"vacancy"`, `"substitution"`, or
        `"interstitial"`) -- this is what unambiguously distinguishes two
        structures with the same `operations_short` but different histories.
    `metadata`
        user-provided additional data.

    Structures are added via `add_pristine`/`add_defect`, which return the
    absolute row index of the new (or, for `add_pristine` with
    `check_duplicates=True`, pre-existing) entry. The `create_*` functions in
    `phase_diagram_workflows.structures.point_defects.defects` are the
    intended way to build defects on top of a container; they call
    `add_defect` internally after resolving the parent and applying the
    defect operation.
    """

    _structures: List[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the internal list of structures to a pandas DataFrame on demand."""
        df_data = []
        for s in self._structures:
            df_data.append(
                {
                    "structure": s["structure"],
                    "unique_id": s["unique_id"],
                    "is_pristine": s["is_pristine"],
                    "stoichiometry": s["stoichiometry"],
                    "generation": s["generation"],
                    "pristine_structure_index": s.get("pristine_structure_index", -1),
                    "parent_index": s.get("parent_index", -1),
                    "operation": s["operation"],
                    "operations_short": s.get("operations_short", ""),
                    "events": s["events"],
                    "metadata": s["metadata"],
                    "creation_timestamp": s["creation_timestamp"],
                }
            )
        return pd.DataFrame(df_data)

    def get_structure_table(self) -> pd.DataFrame:
        """Return the full structure table."""
        return self.to_dataframe()

    def get_defect_table(self) -> pd.DataFrame:
        """Return only defect rows (`is_pristine == False`)."""
        df = self.to_dataframe()
        return df[df["is_pristine"] == False].copy()  # noqa: E712

    def get_pristine_table(self) -> pd.DataFrame:
        """Return only pristine rows (`is_pristine == True`)."""
        df = self.to_dataframe()
        return df[df["is_pristine"] == True].copy()  # noqa: E712

    # ------------------------------------------------------------------
    # Add structures
    # ------------------------------------------------------------------

    def add_pristine(
        self,
        atoms: Atoms,
        unique_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        check_duplicates: bool = True,
        tolerance: float = 1e-6,
    ) -> int:
        """
        Add a pristine reference structure, with optional duplicate checking.

        Parameters
        ----------
        atoms : Atoms
            Structure to add.
        unique_id : str or None
            Custom unique identifier; defaults to "pristine_<row>".
        metadata : dict or None
            Additional metadata.
        check_duplicates : bool
            If True, return the row index of an already-stored pristine
            structure with matching stoichiometry, positions, cell, and
            chemical symbols (within `tolerance`) instead of adding a new one.
        tolerance : float
            Numerical tolerance for comparing positions and cell.

        Returns
        -------
        int
            Absolute row index of the (possibly pre-existing) pristine entry.
        """
        atoms = ensure_uids(atoms)
        uid = unique_id or f"pristine_{len(self._structures)}"

        if check_duplicates:
            candidate_stoich = get_stoichiometry(atoms)
            for idx, s in enumerate(self._structures):
                if not s["is_pristine"]:
                    continue
                s_structure = ensure_uids(s["structure"])
                if (
                    get_stoichiometry(s_structure) == candidate_stoich
                    and np.allclose(s_structure.positions, atoms.positions, atol=tolerance)
                    and np.allclose(s_structure.cell, atoms.cell, atol=tolerance)
                    and s_structure.get_chemical_symbols() == atoms.get_chemical_symbols()
                ):
                    return idx

        entry = {
            "structure": atoms.copy(),
            "unique_id": uid,
            "is_pristine": True,
            "stoichiometry": get_stoichiometry(atoms),
            "generation": 0,
            "pristine_structure_index": -1,
            "parent_index": -1,
            "operation": "pristine",
            "operations_short": "pristine",
            "events": [],
            "metadata": metadata or {},
            "creation_timestamp": datetime.now(),
        }
        self._structures.append(entry)
        return len(self._structures) - 1

    def add_defect(
        self,
        atoms: Atoms,
        operation: str,
        pristine_index: int,
        parent_index: int,
        events: List[dict],
        unique_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add a defect structure.

        Parameters
        ----------
        atoms : Atoms
            The defect structure.
        operation : str
            Full human-readable operation description.
        pristine_index : int
            Absolute index of the original pristine ancestor.
        parent_index : int
            Absolute index of the immediate parent.
        events : list of dict
            Complete chronological list of defect operations.
        unique_id : str or None
            Custom unique identifier; defaults to "defect_<row>".
        metadata : dict or None
            Additional metadata.

        Returns
        -------
        int
            Absolute row index of the newly added defect entry.
        """
        uid = unique_id or f"defect_{len(self._structures)}"
        entry = {
            "structure": atoms.copy(),
            "unique_id": uid,
            "is_pristine": False,
            "stoichiometry": get_stoichiometry(atoms),
            "generation": self._structures[parent_index]["generation"] + 1,
            "pristine_structure_index": pristine_index,
            "parent_index": parent_index,
            "operation": operation,
            "operations_short": make_operations_short(events),
            "events": events,
            "metadata": metadata or {},
            "creation_timestamp": datetime.now(),
        }
        self._structures.append(entry)
        return len(self._structures) - 1

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_structure_index(self, atoms: Atoms, tolerance: float = 1e-6) -> Optional[int]:
        """Return the absolute row index of a structure, or None if absent (identity, then value equality)."""
        atoms = ensure_uids(atoms)
        candidate_stoich = get_stoichiometry(atoms)

        for row_idx, stored in enumerate(self._structures):
            s_structure = ensure_uids(stored["structure"])
            if s_structure is atoms:
                return row_idx
            if (
                get_stoichiometry(s_structure) == candidate_stoich
                and np.allclose(s_structure.positions, atoms.positions, atol=tolerance)
                and np.allclose(s_structure.cell, atoms.cell, atol=tolerance)
                and s_structure.get_chemical_symbols() == atoms.get_chemical_symbols()
            ):
                return row_idx
        return None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_indices(self, indices: List[int]) -> List[dict]:
        """Get structures by absolute indices."""
        return [self._structures[i] for i in indices if i < len(self._structures)]

    def filter_by_generation(self, generation: int) -> List[dict]:
        """Get all structures at an exact distance from pristine."""
        return [s for s in self._structures if s["generation"] == generation]

    def filter_by_max_generation(self, max_generation: int) -> List[dict]:
        """Get all structures within `max_generation` steps of pristine (inclusive)."""
        return [s for s in self._structures if s["generation"] <= max_generation]

    def filter_by_operations_short(self, pattern: str) -> List[dict]:
        """
        Filter by `operations_short` (supports `fnmatch` wildcards `*`, `?`, `[...]`).

        Examples: "vacancy[5]" (exact), "*vacancy[5]*" (contains), "vacancy[*]|substitution[*]".
        """
        if "*" in pattern or "?" in pattern or "[" in pattern:
            return [s for s in self._structures if fnmatch.fnmatch(s.get("operations_short", ""), pattern)]
        return [s for s in self._structures if s.get("operations_short", "") == pattern]

    def filter_by_operations_contains(self, operation_type: str) -> List[dict]:
        """Filter structures whose `operations_short` contains a given substring, e.g. "vacancy"."""
        return [s for s in self._structures if operation_type in s.get("operations_short", "")]

    def filter_by_condition(self, condition: Callable[[dict], bool]) -> List[dict]:
        """Filter by an arbitrary predicate over each structure dict."""
        return [s for s in self._structures if condition(s)]

    def filter_by_unique_id(self, unique_id: str) -> Optional[dict]:
        """Get a structure by its `unique_id`."""
        for s in self._structures:
            if s["unique_id"] == unique_id:
                return s
        return None

    def filter_by_number_of_atoms(self, n_atoms: int) -> List[dict]:
        """Get structures with exactly `n_atoms` atoms."""
        return [s for s in self._structures if len(s["structure"]) == n_atoms]

    def filter_by_element_count(
        self,
        element: str,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
        exact_count: Optional[int] = None,
    ) -> List[dict]:
        """
        Get structures matching an element-count criterion.

        `exact_count`, if given, overrides `min_count`/`max_count` (both inclusive).
        """
        matching = []
        for s in self._structures:
            counts = Counter(s["structure"].get_chemical_symbols())
            element_count = counts.get(element, 0)
            if exact_count is not None:
                if element_count == exact_count:
                    matching.append(s)
            elif (min_count is None or element_count >= min_count) and (
                max_count is None or element_count <= max_count
            ):
                matching.append(s)
        return matching

    def filter_by_stoichiometry(self, formula_pattern: Optional[str] = None) -> List[dict]:
        """
        Get structures matching a stoichiometry pattern (`fnmatch` wildcards supported).

        `None` returns every structure. Examples: "Al107Mg1" (exact), "*Mg1*" (contains 1 Mg).
        """
        if formula_pattern is None:
            return self._structures.copy()
        return [s for s in self._structures if fnmatch.fnmatch(s["stoichiometry"], formula_pattern)]

    def filter_by_parent(self, parent_index: int) -> List[dict]:
        """Get the direct children of a given absolute parent index."""
        return [s for s in self._structures if s.get("parent_index") == parent_index]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def get_pristine_structures(self) -> List[dict]:
        """Get all pristine structures."""
        return [s for s in self._structures if s["is_pristine"]]

    def get_defect_structures(self) -> List[dict]:
        """Get all defect structures."""
        return [s for s in self._structures if not s["is_pristine"]]

    def get_structure(self, index: int) -> dict:
        """Get a structure by absolute index."""
        if 0 <= index < len(self._structures):
            return self._structures[index]
        raise IndexError(f"Index {index} out of range")

    def _find_pristine_index(self, structure_idx: int) -> int:
        """Find the pristine ancestor row for a given structure, walking `parent_index` if needed."""
        if self._structures[structure_idx]["is_pristine"]:
            return structure_idx
        pristine_idx = self._structures[structure_idx].get("pristine_structure_index", -1)
        if pristine_idx == -1:
            current = structure_idx
            visited: set = set()
            while current != -1 and current not in visited:
                visited.add(current)
                if self._structures[current]["is_pristine"]:
                    return current
                current = self._structures[current].get("parent_index", -1)
        return pristine_idx

    def latest_pristine_index(self) -> int:
        """Absolute row index of the most recently added pristine structure."""
        pristine_rows = [i for i, s in enumerate(self._structures) if s["is_pristine"]]
        if not pristine_rows:
            raise ValueError("No pristine structure found in the container.")
        return pristine_rows[-1]

    # ------------------------------------------------------------------
    # Relative index resolution
    # ------------------------------------------------------------------

    def resolve_defect_row(self, relative_index: int) -> int:
        """
        Convert a relative defect index (0 = first defect, -1 = most recent, ...)
        to an absolute row index.
        """
        defect_rows = [i for i, s in enumerate(self._structures) if not s["is_pristine"]]
        if not defect_rows:
            raise IndexError(
                "No defect structures found in the container. Add some defects first, "
                "e.g. with create_vacancy."
            )
        n_defects = len(defect_rows)

        if relative_index >= 0:
            if relative_index >= n_defects:
                raise IndexError(
                    f"Relative defect index {relative_index} out of range. Container has "
                    f"{n_defects} defect structures (indices 0 to {n_defects - 1}). "
                    f"Use a smaller index or negative indices (-{n_defects} to -1)."
                )
            return defect_rows[relative_index]

        abs_idx = n_defects + relative_index
        if abs_idx < 0:
            raise IndexError(
                f"Relative defect index {relative_index} out of range. Container has "
                f"{n_defects} defect structures. Valid negative indices: -{n_defects} to -1."
            )
        return defect_rows[abs_idx]

    def resolve_any_row(self, relative_index: int) -> int:
        """
        Convert a relative index (0 = first structure, -1 = most recent, ...) to an
        absolute row index. Works over both pristine and defect structures.
        """
        n_structures = len(self._structures)
        if n_structures == 0:
            raise IndexError("Container is empty. Add a structure first using add_pristine().")

        if relative_index >= 0:
            if relative_index >= n_structures:
                raise IndexError(
                    f"Relative index {relative_index} out of range. Container has "
                    f"{n_structures} structures (indices 0 to {n_structures - 1}). "
                    f"Use a smaller index or negative indices (-{n_structures} to -1)."
                )
            return relative_index

        abs_idx = n_structures + relative_index
        if abs_idx < 0:
            raise IndexError(
                f"Relative index {relative_index} out of range. Container has "
                f"{n_structures} structures. Valid negative indices: -{n_structures} to -1."
            )
        return abs_idx

    # ------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._structures)

    def __repr__(self) -> str:
        n_pristine = len(self.get_pristine_structures())
        n_defects = len(self.get_defect_structures())
        return f"StructureContainer({len(self)} structures: {n_pristine} pristine, {n_defects} defects)"


# ============================================================================
# Parent resolution (shared by every create_* function in defects.py)
# ============================================================================


def _resolve_parent(
    container: StructureContainer,
    parent_defect_index: Optional[int] = None,
    input_structure: Optional[Atoms] = None,
) -> int:
    """
    Resolve which structure a new defect should be built on top of.

    Resolution modes, in priority order:
    1. `parent_defect_index` given (explicit defect index; negative indices
       resolve via `resolve_defect_row`) -- takes precedence over
       `input_structure` if both are given, with a `UserWarning`.
    2. `input_structure` given -- looked up in the container by value, or
       added as a new pristine entry if not already present.
    3. Neither given -- falls back to the most recently added pristine
       structure.

    Returns
    -------
    int
        Absolute index of the parent structure.
    """
    import warnings

    if parent_defect_index is not None and input_structure is not None:
        warnings.warn(
            f"Both parent_defect_index ({parent_defect_index}) and input_structure provided. "
            f"parent_defect_index will be used and input_structure will be ignored. "
            f"To avoid this warning, provide only one of these parameters.",
            UserWarning,
            stacklevel=3,
        )

    if parent_defect_index is not None:
        if parent_defect_index < 0:
            parent_defect_index = container.resolve_defect_row(parent_defect_index)
        if not (0 <= parent_defect_index < len(container)):
            raise IndexError(
                f"Defect index {parent_defect_index} out of range. "
                f"Container has {len(container)} structures (indices 0 to {len(container) - 1})."
            )
        if container._structures[parent_defect_index]["is_pristine"]:
            raise ValueError(
                f"Index {parent_defect_index} is a pristine structure. "
                f"Use input_structure parameter instead if you want to work with a pristine structure."
            )
        return parent_defect_index

    if input_structure is not None:
        input_structure = ensure_uids(input_structure)
        for idx, s in enumerate(container._structures):
            s_structure = ensure_uids(s["structure"])
            if len(s_structure) != len(input_structure):
                continue
            if (
                np.allclose(s_structure.positions, input_structure.positions)
                and np.allclose(s_structure.cell, input_structure.cell)
                and s_structure.get_chemical_symbols() == input_structure.get_chemical_symbols()
            ):
                return idx
        return container.add_pristine(input_structure)

    pristine_indices = [i for i, s in enumerate(container._structures) if s["is_pristine"]]
    if not pristine_indices:
        raise ValueError("No pristine structure found in the container. Add one first using add_pristine(atoms).")
    return pristine_indices[-1]


# ============================================================================
# Free-function API (mirrors the StructureContainer methods above)
# ============================================================================
#
# These take the container as their first argument instead of being called as
# methods -- this is the form actually imported and used by the notebooks
# this module was ported from (e.g. `from ... import add_pristine,
# get_structure_table`), so it's kept alongside the class rather than folded
# away.


def get_structure_table(structure_container: StructureContainer) -> pd.DataFrame:
    """See `StructureContainer.get_structure_table`."""
    return structure_container.get_structure_table()


def get_defect_table(structure_container: StructureContainer) -> pd.DataFrame:
    """See `StructureContainer.get_defect_table`."""
    return structure_container.get_defect_table()


def get_pristine_table(structure_container: StructureContainer) -> pd.DataFrame:
    """See `StructureContainer.get_pristine_table`."""
    return structure_container.get_pristine_table()


def add_pristine(
    structure_container: Optional[StructureContainer] = None,
    atoms: Optional[Atoms] = None,
    unique_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    check_duplicates: bool = True,
    tolerance: float = 1e-6,
) -> StructureContainer:
    """
    Add a pristine reference structure, creating a new container if
    `structure_container` is None. See `StructureContainer.add_pristine`.

    Examples
    --------
    >>> from ase.build import bulk
    >>> container = add_pristine(atoms=bulk('Al', cubic=True), unique_id="Al_fcc")
    """
    if structure_container is None:
        structure_container = StructureContainer()
    structure_container.add_pristine(atoms, unique_id, metadata, check_duplicates, tolerance)
    return structure_container


def filter_by_indices(structure_container: StructureContainer, indices: List[int]) -> List[dict]:
    """See `StructureContainer.filter_by_indices`."""
    return structure_container.filter_by_indices(indices)


def filter_by_generation(structure_container: StructureContainer, generation: int) -> List[dict]:
    """See `StructureContainer.filter_by_generation`."""
    return structure_container.filter_by_generation(generation)


def filter_by_max_generation(structure_container: StructureContainer, max_generation: int) -> List[dict]:
    """See `StructureContainer.filter_by_max_generation`."""
    return structure_container.filter_by_max_generation(max_generation)


def filter_by_operations_short(structure_container: StructureContainer, pattern: str) -> List[dict]:
    """See `StructureContainer.filter_by_operations_short`."""
    return structure_container.filter_by_operations_short(pattern)


def filter_by_operations_contains(structure_container: StructureContainer, operation_type: str) -> List[dict]:
    """See `StructureContainer.filter_by_operations_contains`."""
    return structure_container.filter_by_operations_contains(operation_type)


def filter_by_condition(structure_container: StructureContainer, condition: Callable[[dict], bool]) -> List[dict]:
    """See `StructureContainer.filter_by_condition`."""
    return structure_container.filter_by_condition(condition)


def filter_by_unique_id(structure_container: StructureContainer, unique_id: str) -> Optional[dict]:
    """See `StructureContainer.filter_by_unique_id`."""
    return structure_container.filter_by_unique_id(unique_id)


def filter_by_number_of_atoms(structure_container: StructureContainer, n_atoms: int) -> List[dict]:
    """See `StructureContainer.filter_by_number_of_atoms`."""
    return structure_container.filter_by_number_of_atoms(n_atoms)


def filter_by_element_count(
    structure_container: StructureContainer,
    element: str,
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
    exact_count: Optional[int] = None,
) -> List[dict]:
    """See `StructureContainer.filter_by_element_count`."""
    return structure_container.filter_by_element_count(element, min_count, max_count, exact_count)


def filter_by_stoichiometry(structure_container: StructureContainer, formula_pattern: Optional[str] = None) -> List[dict]:
    """See `StructureContainer.filter_by_stoichiometry`."""
    return structure_container.filter_by_stoichiometry(formula_pattern)


def filter_by_parent(structure_container: StructureContainer, parent_index: int) -> List[dict]:
    """See `StructureContainer.filter_by_parent`."""
    return structure_container.filter_by_parent(parent_index)


def get_structure(structure_container: StructureContainer, index: int) -> dict:
    """See `StructureContainer.get_structure`."""
    return structure_container.get_structure(index)


def get_pristine_structures(structure_container: StructureContainer) -> List[dict]:
    """See `StructureContainer.get_pristine_structures`."""
    return structure_container.get_pristine_structures()


def get_defect_structures(structure_container: StructureContainer) -> List[dict]:
    """See `StructureContainer.get_defect_structures`."""
    return structure_container.get_defect_structures()


def latest_pristine_index(structure_container: StructureContainer) -> int:
    """See `StructureContainer.latest_pristine_index`."""
    return structure_container.latest_pristine_index()


def resolve_defect_row(structure_container: StructureContainer, relative_index: int) -> int:
    """See `StructureContainer.resolve_defect_row`."""
    return structure_container.resolve_defect_row(relative_index)


def resolve_any_row(structure_container: StructureContainer, relative_index: int) -> int:
    """See `StructureContainer.resolve_any_row`."""
    return structure_container.resolve_any_row(relative_index)
