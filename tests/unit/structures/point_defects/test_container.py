"""
Unit tests for phase_diagram_workflows.structures.point_defects.container.

All tests build structures directly with ase (bulk/Atoms) -- no external
services to mock here, unlike test_materials_project.py.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from phase_diagram_workflows.structures.point_defects.container import (
    UID_KEY,
    StructureContainer,
    _protected_uids_from_events,
    _resolve_parent,
    add_pristine,
    append_atom_with_uid,
    element_uids,
    ensure_uids,
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


# ---------------------------------------------------------------------------
# UID helpers
# ---------------------------------------------------------------------------


class TestUidHelpers:
    def test_ensure_uids_assigns_sequential_ids(self):
        atoms = bulk("Al", cubic=True)
        atoms = ensure_uids(atoms)
        assert atoms.arrays[UID_KEY].tolist() == list(range(len(atoms)))

    def test_ensure_uids_does_not_overwrite_existing(self):
        atoms = ensure_uids(bulk("Al", cubic=True))
        atoms.arrays[UID_KEY][0] = 99
        atoms2 = ensure_uids(atoms)
        assert atoms2.arrays[UID_KEY][0] == 99

    def test_next_uid_increments_past_max(self):
        atoms = ensure_uids(bulk("Al", cubic=True))
        assert next_uid(atoms) == len(atoms)

    def test_next_uid_zero_for_uid_less_or_empty(self):
        assert next_uid(bulk("Al", cubic=True)) == 0
        assert next_uid(Atoms()) == 0

    def test_uid_to_index_roundtrip_and_missing(self):
        atoms = ensure_uids(bulk("Al", cubic=True))
        assert uid_to_index(atoms, 2) == 2
        assert uid_to_index(atoms, 999) is None

    def test_element_uids(self):
        atoms = ensure_uids(Atoms("AlMgAl"))
        assert element_uids(atoms, "Al") == [0, 2]
        assert element_uids(atoms, "Mg") == [1]

    def test_validate_atoms_arrays_raises_on_inconsistent_length(self):
        atoms = ensure_uids(bulk("Al", cubic=True))
        atoms.arrays[UID_KEY] = atoms.arrays[UID_KEY][:-1]
        with pytest.raises(ValueError, match="Inconsistent per-atom arrays"):
            validate_atoms_arrays(atoms)

    def test_append_atom_with_uid_gets_fresh_uid_and_position(self):
        atoms = ensure_uids(bulk("Al", cubic=True))
        n_before = len(atoms)
        atoms2 = append_atom_with_uid(atoms, "Mg", [1.0, 2.0, 3.0])
        assert len(atoms2) == n_before + 1
        assert atoms2.arrays[UID_KEY][-1] == n_before
        assert atoms2.get_chemical_symbols()[-1] == "Mg"
        np.testing.assert_allclose(atoms2.positions[-1], [1.0, 2.0, 3.0])
        # original left untouched
        assert len(atoms) == n_before

    def test_protected_uids_from_events(self):
        events = [
            {"type": "vacancy", "site_uid": 1},
            {"type": "substitution", "atom_uid": 2},
            {"type": "substitution", "site_uid": 3},  # no atom_uid key -> falls back
            {"type": "interstitial", "atom_uid": 4},
        ]
        assert _protected_uids_from_events(events) == {2, 3, 4}
        assert _protected_uids_from_events(None) == set()
        assert _protected_uids_from_events([]) == set()

    def test_validate_structure_passes_for_normal_structure(self):
        assert validate_structure(bulk("Al", cubic=True)) is True

    def test_validate_structure_raises_for_close_atoms(self):
        atoms = Atoms("Al2", positions=[[0, 0, 0], [0.1, 0, 0]], cell=[10, 10, 10], pbc=True)
        with pytest.raises(ValueError, match="too close"):
            validate_structure(atoms, min_distance=0.5)


class TestStoichiometryAndOperations:
    def test_get_stoichiometry(self):
        assert get_stoichiometry(bulk("Al", cubic=True)) == "Al4"
        assert get_stoichiometry(Atoms("Al3Mg1")) == "Al3Mg1"

    def test_make_operations_short_empty(self):
        assert make_operations_short([]) == "no_operations"

    def test_make_operations_short_single_and_multiple(self):
        events = [{"type": "vacancy", "site_uid": 5}]
        assert make_operations_short(events) == "vacancy[5]"

        events = [
            {"type": "vacancy", "site_uid": 5},
            {"type": "substitution", "from": "Al", "to": "Mg", "site_uid": 10},
            {"type": "interstitial", "element": "Mg", "atom_uid": 20},
        ]
        assert make_operations_short(events) == "vacancy[5]|substitution[Al->Mg]|interstitial[Mg]"


# ---------------------------------------------------------------------------
# StructureContainer
# ---------------------------------------------------------------------------


class TestAddPristine:
    def test_add_pristine_returns_row_index_and_fields(self):
        container = StructureContainer()
        idx = container.add_pristine(bulk("Al", cubic=True), unique_id="Al_fcc")
        assert idx == 0
        row = container.get_structure(0)
        assert row["is_pristine"] is True
        assert row["unique_id"] == "Al_fcc"
        assert row["generation"] == 0
        assert row["stoichiometry"] == "Al4"
        assert row["operations_short"] == "pristine"

    def test_default_unique_id(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        assert container.get_structure(0)["unique_id"] == "pristine_0"

    def test_check_duplicates_returns_existing_index(self):
        container = StructureContainer()
        atoms = bulk("Al", cubic=True)
        idx1 = container.add_pristine(atoms.copy())
        idx2 = container.add_pristine(atoms.copy(), check_duplicates=True)
        assert idx1 == idx2
        assert len(container) == 1

    def test_check_duplicates_false_always_adds(self):
        container = StructureContainer()
        atoms = bulk("Al", cubic=True)
        container.add_pristine(atoms.copy(), check_duplicates=False)
        container.add_pristine(atoms.copy(), check_duplicates=False)
        assert len(container) == 2

    def test_different_stoichiometry_is_not_a_duplicate(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        container.add_pristine(Atoms("Mg4", positions=bulk("Al", cubic=True).positions, cell=bulk("Al", cubic=True).cell, pbc=True))
        assert len(container) == 2


class TestAddDefect:
    def test_add_defect_sets_generation_and_lineage(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        defect_atoms = bulk("Al", cubic=True)[:-1]
        events = [{"type": "vacancy", "site_uid": 0}]
        d_idx = container.add_defect(
            atoms=defect_atoms, operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx, events=events
        )
        row = container.get_structure(d_idx)
        assert row["generation"] == 1
        assert row["is_pristine"] is False
        assert row["pristine_structure_index"] == p_idx
        assert row["parent_index"] == p_idx
        assert row["operations_short"] == "vacancy[0]"
        assert row["stoichiometry"] == "Al3"

    def test_second_generation_defect(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        d1 = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        d2 = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-2], operation="vacancy[1]", pristine_index=p_idx, parent_index=d1,
            events=[{"type": "vacancy", "site_uid": 0}, {"type": "vacancy", "site_uid": 1}],
        )
        assert container.get_structure(d2)["generation"] == 2


class TestTables:
    def _make_container(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True), unique_id="pristine")
        container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        return container

    def test_get_structure_table_has_all_rows(self):
        container = self._make_container()
        df = get_structure_table(container)
        assert len(df) == 2
        assert set(df["is_pristine"]) == {True, False}

    def test_get_defect_and_pristine_tables(self):
        container = self._make_container()
        assert len(get_defect_table(container)) == 1
        assert len(get_pristine_table(container)) == 1
        assert bool(get_pristine_table(container).iloc[0]["is_pristine"]) is True
        assert bool(get_defect_table(container).iloc[0]["is_pristine"]) is False


class TestFindStructureIndex:
    def test_finds_by_identity(self):
        container = StructureContainer()
        atoms = bulk("Al", cubic=True)
        container.add_pristine(atoms)
        assert container.find_structure_index(atoms) == 0

    def test_finds_by_value_equality(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        assert container.find_structure_index(bulk("Al", cubic=True)) == 0

    def test_returns_none_when_absent(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        assert container.find_structure_index(bulk("Cu", cubic=True)) is None


class TestFiltering:
    def _make_container(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        d0 = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        d1_atoms = bulk("Al", cubic=True).copy()
        d1_atoms[1].symbol = "Mg"
        container.add_defect(
            atoms=d1_atoms, operation="substitution[Al->Mg]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "substitution", "from": "Al", "to": "Mg", "site_uid": 1}],
        )
        return container, p_idx, d0

    def test_filter_by_indices(self):
        container, p_idx, d0 = self._make_container()
        result = container.filter_by_indices([0, 2, 99])
        assert len(result) == 2  # 99 silently dropped (out of range)

    def test_filter_by_generation(self):
        container, p_idx, d0 = self._make_container()
        assert len(container.filter_by_generation(0)) == 1
        assert len(container.filter_by_generation(1)) == 2

    def test_filter_by_max_generation(self):
        container, *_ = self._make_container()
        assert len(container.filter_by_max_generation(0)) == 1
        assert len(container.filter_by_max_generation(1)) == 3

    def test_filter_by_operations_short_exact_match(self):
        container, *_ = self._make_container()
        # "pristine" (no '[', '*', '?') takes the plain string-equality branch.
        assert len(container.filter_by_operations_short("pristine")) == 1

    def test_filter_by_operations_short_wildcard(self):
        container, *_ = self._make_container()
        assert len(container.filter_by_operations_short("*")) == 3
        assert len(container.filter_by_operations_short("vacancy*")) == 1

    def test_filter_by_operations_short_exact_match_with_brackets_is_a_known_quirk(self):
        # Any '[' routes the pattern through fnmatch (see filter_by_operations_short),
        # where "[0]" is a character class matching a literal '0' -- not the literal
        # substring "[0]". So a pattern that looks like an exact operations_short
        # value, e.g. "vacancy[0]", never actually matches that value: fnmatch reads
        # it as "vacancy" + one char equal to '0', i.e. "vacancy0". Inherited as-is
        # from the ported source; filter_by_operations_contains (substring, no
        # fnmatch) is the reliable way to match a specific op by its bracketed id.
        container, *_ = self._make_container()
        assert len(container.filter_by_operations_short("vacancy[0]")) == 0

    def test_filter_by_operations_contains(self):
        container, *_ = self._make_container()
        assert len(container.filter_by_operations_contains("vacancy")) == 1
        assert len(container.filter_by_operations_contains("substitution")) == 1

    def test_filter_by_condition(self):
        container, *_ = self._make_container()
        result = container.filter_by_condition(lambda s: s["is_pristine"])
        assert len(result) == 1

    def test_filter_by_unique_id(self):
        container, *_ = self._make_container()
        found = container.filter_by_unique_id("pristine_0")
        assert found is not None
        assert container.filter_by_unique_id("does-not-exist") is None

    def test_filter_by_number_of_atoms(self):
        container, *_ = self._make_container()
        assert len(container.filter_by_number_of_atoms(4)) == 2  # pristine + substitution
        assert len(container.filter_by_number_of_atoms(3)) == 1  # vacancy

    def test_filter_by_element_count(self):
        container, *_ = self._make_container()
        assert len(container.filter_by_element_count("Mg", exact_count=1)) == 1
        assert len(container.filter_by_element_count("Al", min_count=4)) == 1
        assert len(container.filter_by_element_count("Al", max_count=3)) == 2

    def test_filter_by_stoichiometry(self):
        container, *_ = self._make_container()
        assert len(container.filter_by_stoichiometry("Al4")) == 1
        assert len(container.filter_by_stoichiometry("*Mg1*")) == 1
        assert len(container.filter_by_stoichiometry(None)) == 3

    def test_filter_by_parent(self):
        container, p_idx, d0 = self._make_container()
        children = container.filter_by_parent(p_idx)
        assert len(children) == 2


class TestSelectionAndIndexing:
    def test_get_pristine_and_defect_structures(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        assert len(get_pristine_structures(container)) == 1
        assert len(get_defect_structures(container)) == 1

    def test_get_structure_raises_index_error(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        with pytest.raises(IndexError):
            get_structure(container, 5)

    def test_find_pristine_index_chain(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        d1 = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        d2 = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-2], operation="vacancy[1]", pristine_index=p_idx, parent_index=d1,
            events=[{"type": "vacancy", "site_uid": 0}, {"type": "vacancy", "site_uid": 1}],
        )
        assert container._find_pristine_index(d2) == p_idx

    def test_latest_pristine_index(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True), check_duplicates=False)
        idx2 = container.add_pristine(bulk("Cu", cubic=True), check_duplicates=False)
        assert latest_pristine_index(container) == idx2

    def test_latest_pristine_index_raises_when_empty(self):
        with pytest.raises(ValueError, match="No pristine"):
            latest_pristine_index(StructureContainer())

    def test_resolve_defect_row_positive_and_negative(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        d0 = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        d1 = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-2], operation="vacancy[1]", pristine_index=p_idx, parent_index=d0,
            events=[{"type": "vacancy", "site_uid": 0}, {"type": "vacancy", "site_uid": 1}],
        )
        assert resolve_defect_row(container, 0) == d0
        assert resolve_defect_row(container, 1) == d1
        assert resolve_defect_row(container, -1) == d1

    def test_resolve_defect_row_raises_when_no_defects(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        with pytest.raises(IndexError, match="No defect structures"):
            resolve_defect_row(container, 0)

    def test_resolve_defect_row_out_of_range(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        with pytest.raises(IndexError):
            resolve_defect_row(container, 5)
        with pytest.raises(IndexError):
            resolve_defect_row(container, -5)

    def test_resolve_any_row(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        assert resolve_any_row(container, 0) == 0
        assert resolve_any_row(container, -1) == 0

    def test_resolve_any_row_raises_when_empty(self):
        with pytest.raises(IndexError, match="empty"):
            resolve_any_row(StructureContainer(), 0)

    def test_len_and_repr(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        assert len(container) == 1
        assert "1 structures" in repr(container)
        assert "1 pristine" in repr(container)


class TestResolveParent:
    def test_defaults_to_latest_pristine(self):
        container = StructureContainer()
        idx = container.add_pristine(bulk("Al", cubic=True))
        assert _resolve_parent(container) == idx

    def test_raises_when_container_empty(self):
        with pytest.raises(ValueError, match="No pristine structure"):
            _resolve_parent(StructureContainer())

    def test_explicit_parent_defect_index(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        d_idx = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        assert _resolve_parent(container, parent_defect_index=d_idx) == d_idx

    def test_parent_defect_index_rejects_pristine_target(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        with pytest.raises(ValueError, match="pristine structure"):
            _resolve_parent(container, parent_defect_index=p_idx)

    def test_negative_parent_defect_index_resolves_relative(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        d_idx = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        assert _resolve_parent(container, parent_defect_index=-1) == d_idx

    def test_input_structure_reuses_existing_row(self):
        container = StructureContainer()
        atoms = bulk("Al", cubic=True)
        idx = container.add_pristine(atoms.copy())
        assert _resolve_parent(container, input_structure=atoms.copy()) == idx

    def test_input_structure_adds_new_pristine_if_absent(self):
        container = StructureContainer()
        container.add_pristine(bulk("Al", cubic=True))
        new_atoms = bulk("Cu", cubic=True)
        resolved = _resolve_parent(container, input_structure=new_atoms)
        assert len(container) == 2
        assert container.get_structure(resolved)["is_pristine"] is True

    def test_both_given_prefers_parent_defect_index_with_warning(self):
        container = StructureContainer()
        p_idx = container.add_pristine(bulk("Al", cubic=True))
        d_idx = container.add_defect(
            atoms=bulk("Al", cubic=True)[:-1], operation="vacancy[0]", pristine_index=p_idx, parent_index=p_idx,
            events=[{"type": "vacancy", "site_uid": 0}],
        )
        with pytest.warns(UserWarning, match="Both parent_defect_index"):
            resolved = _resolve_parent(container, parent_defect_index=d_idx, input_structure=bulk("Cu", cubic=True))
        assert resolved == d_idx
        assert len(container) == 2  # input_structure was ignored, nothing added


class TestAddPristineFreeFunction:
    def test_creates_new_container_when_none(self):
        container = add_pristine(atoms=bulk("Al", cubic=True))
        assert isinstance(container, StructureContainer)
        assert len(container) == 1

    def test_reuses_given_container(self):
        container = StructureContainer()
        result = add_pristine(container, atoms=bulk("Al", cubic=True))
        assert result is container
        assert len(container) == 1
