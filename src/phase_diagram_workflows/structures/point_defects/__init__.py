"""
Point-defect structure creation on top of a StructureContainer.

- `container`: `StructureContainer` (pristine/defect storage with lineage
  tracking), UID helpers, and the free-function API mirroring its methods.
- `sites`: interstitial site discovery (Voronoi/Delaunay) and symmetry-unique
  sublattice discovery/tiling.
- `defects`: `create_vacancy`/`create_substitution`/`create_interstitial`
  (and their `_batch` variants) to actually build defect structures.
- `formation_energy`: `compute_formation_energy` and the `delta_n_for_*`
  helpers -- chemical-potential bookkeeping shared by every defect type.
- `random_antisites`: one-shot random antisite-substitution composition sweeps
  (`generate_random_binary_structures`), independent of `StructureContainer`.

The most commonly used names are re-exported here; submodules remain
importable directly for anything not listed (e.g.
`from phase_diagram_workflows.structures.point_defects.sites import get_delaunay_interstitial_sites`).
"""

from .container import (
    UID_KEY,
    StructureContainer,
    add_pristine,
    append_atom_with_uid,
    element_uids,
    ensure_uids,
    filter_by_condition,
    filter_by_element_count,
    filter_by_generation,
    filter_by_indices,
    filter_by_max_generation,
    filter_by_number_of_atoms,
    filter_by_operations_contains,
    filter_by_operations_short,
    filter_by_parent,
    filter_by_stoichiometry,
    filter_by_unique_id,
    get_defect_structures,
    get_defect_table,
    get_pristine_structures,
    get_pristine_table,
    get_stoichiometry,
    get_structure,
    get_structure_table,
    latest_pristine_index,
    make_operations_short,
    next_uid,
    resolve_any_row,
    resolve_defect_row,
    uid_to_index,
    validate_atoms_arrays,
    validate_structure,
)
from .defects import (
    create_interstitial,
    create_interstitial_batch,
    create_substitution,
    create_substitution_batch,
    create_vacancy,
    create_vacancy_batch,
)
from .formation_energy import (
    compute_formation_energy,
    delta_n_for_interstitial,
    delta_n_for_substitution,
    delta_n_for_vacancy,
)
from .random_antisites import generate_random_binary_structures, get_element_fractions
from .sites import (
    discover_atomic_sublattices,
    discover_interstitial_sublattices,
    get_delaunay_interstitial_sites,
    get_voronoi_interstitial_sites,
    tile_atomic_sublattices,
    tile_interstitial_sublattices,
    validate_sublattice_coverage,
)

__all__ = [
    "UID_KEY",
    "StructureContainer",
    "add_pristine",
    "append_atom_with_uid",
    "element_uids",
    "ensure_uids",
    "filter_by_condition",
    "filter_by_element_count",
    "filter_by_generation",
    "filter_by_indices",
    "filter_by_max_generation",
    "filter_by_number_of_atoms",
    "filter_by_operations_contains",
    "filter_by_operations_short",
    "filter_by_parent",
    "filter_by_stoichiometry",
    "filter_by_unique_id",
    "get_defect_structures",
    "get_defect_table",
    "get_pristine_structures",
    "get_pristine_table",
    "get_stoichiometry",
    "get_structure",
    "get_structure_table",
    "latest_pristine_index",
    "make_operations_short",
    "next_uid",
    "resolve_any_row",
    "resolve_defect_row",
    "uid_to_index",
    "validate_atoms_arrays",
    "validate_structure",
    "create_interstitial",
    "create_interstitial_batch",
    "create_substitution",
    "create_substitution_batch",
    "create_vacancy",
    "create_vacancy_batch",
    "compute_formation_energy",
    "delta_n_for_interstitial",
    "delta_n_for_substitution",
    "delta_n_for_vacancy",
    "discover_atomic_sublattices",
    "discover_interstitial_sublattices",
    "get_delaunay_interstitial_sites",
    "get_voronoi_interstitial_sites",
    "tile_atomic_sublattices",
    "tile_interstitial_sublattices",
    "validate_sublattice_coverage",
    "generate_random_binary_structures",
    "get_element_fractions",
]
