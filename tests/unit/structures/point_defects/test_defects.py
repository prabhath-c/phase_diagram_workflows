"""
Unit tests for phase_diagram_workflows.structures.point_defects.defects.

Uses ase.build.bulk('Al', cubic=True) (4-atom FCC cell) as the standard host
throughout, with StructureContainer built directly via add_pristine.
"""

import numpy as np
import pytest
from ase.build import bulk

from phase_diagram_workflows.structures.point_defects.container import StructureContainer
from phase_diagram_workflows.structures.point_defects.defects import (
    create_interstitial,
    create_interstitial_batch,
    create_substitution,
    create_substitution_batch,
    create_vacancy,
    create_vacancy_batch,
)


def _fresh_container():
    container = StructureContainer()
    container.add_pristine(bulk("Al", cubic=True), unique_id="host")
    return container


# ---------------------------------------------------------------------------
# Vacancies
# ---------------------------------------------------------------------------


class TestCreateVacancy:
    def test_explicit_removes_requested_atom(self):
        container = _fresh_container()
        container = create_vacancy(container, atom_ids=[0])
        defect = container.get_defect_structures()[0]
        assert len(defect["structure"]) == 3
        assert defect["events"][-1]["type"] == "vacancy"
        assert defect["events"][-1]["site_uid"] == 0
        assert defect["operations_short"] == "vacancy[0]"

    def test_explicit_multiple_atom_ids(self):
        container = _fresh_container()
        container = create_vacancy(container, atom_ids=[0, 1])
        defect = container.get_defect_structures()[0]
        assert len(defect["structure"]) == 2
        # `operation` is the single "vacancy[N]" summary computed by create_vacancy;
        # `operations_short` is a per-event join computed independently by the
        # container from the full events list, so it lists each removed uid.
        assert defect["operation"] == "vacancy[2]"
        assert defect["operations_short"] == "vacancy[0]|vacancy[1]"

    def test_explicit_out_of_range_raises_index_error(self):
        container = _fresh_container()
        with pytest.raises(IndexError, match="out of range"):
            create_vacancy(container, atom_ids=[99])

    def test_random_is_reproducible_with_seed(self):
        c1 = create_vacancy(_fresh_container(), n=1, seed=42)
        c2 = create_vacancy(_fresh_container(), n=1, seed=42)
        e1 = c1.get_defect_structures()[0]["events"][-1]
        e2 = c2.get_defect_structures()[0]["events"][-1]
        assert e1["site_uid"] == e2["site_uid"]

    def test_random_records_seed_in_metadata(self):
        container = create_vacancy(_fresh_container(), n=1, seed=7)
        assert container.get_defect_structures()[0]["metadata"]["seed"] == 7

    def test_explicit_mode_does_not_record_seed(self):
        container = create_vacancy(_fresh_container(), atom_ids=[0])
        assert "seed" not in container.get_defect_structures()[0]["metadata"]

    def test_random_element_filter(self):
        host = bulk("Al", cubic=True)
        host[1].symbol = "Mg"
        container = StructureContainer()
        container.add_pristine(host)
        container = create_vacancy(container, n=1, seed=0, vacancy_element="Mg")
        removed = container.get_defect_structures()[0]["events"][-1]
        assert removed["removed_element"] == "Mg"

    def test_random_element_list_requires_matching_n(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="must equal len"):
            create_vacancy(container, n=1, vacancy_element=["Al", "Mg"])

    def test_random_element_list_picks_one_per_element(self):
        host = bulk("Al", cubic=True)
        host[1].symbol = "Mg"
        container = StructureContainer()
        container.add_pristine(host)
        container = create_vacancy(container, n=2, seed=0, vacancy_element=["Al", "Mg"])
        removed_elements = {e["removed_element"] for e in container.get_defect_structures()[0]["events"][-2:]}
        assert removed_elements == {"Al", "Mg"}

    def test_forbid_uids_excludes_candidates(self):
        container = _fresh_container()
        container = create_vacancy(container, n=1, seed=0, forbid_uids=[0, 1, 2])
        removed_uid = container.get_defect_structures()[0]["events"][-1]["site_uid"]
        assert removed_uid == 3

    def test_protect_history_forbids_prior_defect_sites(self):
        container = _fresh_container()
        # substitute atom 0 (defect row 1), then random-vacate the other 3 sites --
        # protect_history should keep uid 0 out of the candidate pool even though
        # it's not in forbid_uids explicitly.
        container = create_substitution(container, to_element="Mg", atom_ids=[0])
        container = create_vacancy(container, n=3, seed=0, protect_history=True, parent_defect_index=1)
        removed_uids = {e["site_uid"] for e in container.get_defect_structures()[-1]["events"] if e["type"] == "vacancy"}
        assert 0 not in removed_uids
        assert removed_uids == {1, 2, 3}

    def test_not_enough_candidates_raises(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="Not enough candidates"):
            create_vacancy(container, n=10, seed=0)

    def test_requires_exactly_one_of_atom_ids_or_n(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="exactly one"):
            create_vacancy(container)
        with pytest.raises(ValueError, match="exactly one"):
            create_vacancy(container, atom_ids=[0], n=1)

    def test_no_valid_indices_after_forbid_raises(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="No valid indices"):
            create_vacancy(container, atom_ids=[0], forbid_uids=[0])


class TestCreateVacancyBatch:
    def test_explicit_separate_structures(self):
        container = _fresh_container()
        container = create_vacancy_batch(container, target_indices=[0], atom_ids=[0, 1], separate_structures=True)
        assert len(container.get_defect_structures()) == 2

    def test_explicit_combined_structure(self):
        container = _fresh_container()
        container = create_vacancy_batch(container, target_indices=[0], atom_ids=[0, 1], separate_structures=False)
        defects = container.get_defect_structures()
        assert len(defects) == 1
        assert len(defects[0]["structure"]) == 2

    def test_random_with_n_structures_uses_incrementing_seeds(self):
        container = _fresh_container()
        container = create_vacancy_batch(container, target_indices=[0], n=1, seed=10, n_structures=3)
        seeds = sorted(s["metadata"]["seed"] for s in container.get_defect_structures())
        assert seeds == [10, 11, 12]

    def test_requires_exactly_one_of_atom_ids_or_n(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="exactly one"):
            create_vacancy_batch(container, target_indices=[0])


# ---------------------------------------------------------------------------
# Substitutions
# ---------------------------------------------------------------------------


class TestCreateSubstitution:
    def test_explicit_substitutes_requested_atom(self):
        container = _fresh_container()
        container = create_substitution(container, to_element="Mg", atom_ids=[0])
        defect = container.get_defect_structures()[0]
        assert defect["structure"].get_chemical_symbols()[0] == "Mg"
        assert defect["events"][-1] == {
            "type": "substitution",
            "from": "Al",
            "to": "Mg",
            "atom_uid": 0,
            "site_uid": 0,
            "site_pos0": defect["events"][-1]["site_pos0"],
            "pos_at_creation": defect["events"][-1]["pos_at_creation"],
        }
        assert defect["operations_short"] == "substitution[Al->Mg]"

    def test_random_requires_from_element(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="from_element is required"):
            create_substitution(container, to_element="Mg", n=1, seed=0)

    def test_random_is_reproducible(self):
        c1 = create_substitution(_fresh_container(), to_element="Mg", n=1, seed=3, from_element="Al")
        c2 = create_substitution(_fresh_container(), to_element="Mg", n=1, seed=3, from_element="Al")
        assert c1.get_defect_structures()[0]["events"][-1]["site_uid"] == c2.get_defect_structures()[0]["events"][-1]["site_uid"]

    def test_not_enough_candidates_raises(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="Not enough candidates"):
            create_substitution(container, to_element="Mg", n=10, seed=0, from_element="Al")

    def test_requires_exactly_one_of_atom_ids_or_n(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="exactly one"):
            create_substitution(container, to_element="Mg")

    def test_explicit_out_of_range_raises(self):
        container = _fresh_container()
        with pytest.raises(IndexError):
            create_substitution(container, to_element="Mg", atom_ids=[99])


class TestCreateSubstitutionBatch:
    def test_explicit_separate_structures(self):
        container = _fresh_container()
        container = create_substitution_batch(container, target_indices=[0], to_element="Mg", atom_ids=[0, 1], separate_structures=True)
        assert len(container.get_defect_structures()) == 2

    def test_random_with_n_structures(self):
        container = _fresh_container()
        container = create_substitution_batch(
            container, target_indices=[0], to_element="Mg", n=1, seed=0, from_element="Al", n_structures=2
        )
        assert len(container.get_defect_structures()) == 2

    def test_requires_exactly_one_of_atom_ids_or_n(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="exactly one"):
            create_substitution_batch(container, target_indices=[0], to_element="Mg")


# ---------------------------------------------------------------------------
# Interstitials
# ---------------------------------------------------------------------------


class TestCreateInterstitial:
    def _sublattice(self):
        return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

    def test_explicit_inserts_atom_at_site(self):
        container = _fresh_container()
        sublattice = self._sublattice()
        container = create_interstitial(container, sublattice=sublattice, element="Mg", site_ids=[1])
        defect = container.get_defect_structures()[0]
        assert len(defect["structure"]) == 5
        assert defect["structure"].get_chemical_symbols()[-1] == "Mg"
        np.testing.assert_allclose(defect["structure"].positions[-1], sublattice[1])
        assert defect["operations_short"] == "interstitial[Mg]"

    def test_random_is_reproducible(self):
        sublattice = self._sublattice()
        c1 = create_interstitial(_fresh_container(), sublattice=sublattice, element="Mg", n=1, seed=1)
        c2 = create_interstitial(_fresh_container(), sublattice=sublattice, element="Mg", n=1, seed=1)
        p1 = c1.get_defect_structures()[0]["structure"].positions[-1]
        p2 = c2.get_defect_structures()[0]["structure"].positions[-1]
        np.testing.assert_allclose(p1, p2)

    def test_sublattice_wrong_shape_raises(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="shape"):
            create_interstitial(container, sublattice=np.array([1.0, 2.0, 3.0]), element="Mg", site_ids=[0])

    def test_empty_site_ids_raises(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="empty"):
            create_interstitial(container, sublattice=self._sublattice(), element="Mg", site_ids=[])

    def test_out_of_range_site_id_raises(self):
        container = _fresh_container()
        with pytest.raises(IndexError):
            create_interstitial(container, sublattice=self._sublattice(), element="Mg", site_ids=[99])

    def test_n_exceeding_sublattice_size_raises(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="only has"):
            create_interstitial(container, sublattice=self._sublattice(), element="Mg", n=10, seed=0)

    def test_requires_exactly_one_of_site_ids_or_n(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="exactly one"):
            create_interstitial(container, sublattice=self._sublattice(), element="Mg")


class TestCreateInterstitialBatch:
    def _sublattice(self):
        return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def test_default_uses_every_site_separately(self):
        container = _fresh_container()
        container = create_interstitial_batch(container, target_indices=[0], sublattice=self._sublattice(), element="Mg")
        assert len(container.get_defect_structures()) == 2

    def test_explicit_subset_combined(self):
        container = _fresh_container()
        container = create_interstitial_batch(
            container, target_indices=[0], sublattice=self._sublattice(), element="Mg",
            site_ids=[0, 1], separate_structures=False,
        )
        defects = container.get_defect_structures()
        assert len(defects) == 1
        assert len(defects[0]["structure"]) == 6

    def test_random_with_n_structures(self):
        container = _fresh_container()
        container = create_interstitial_batch(
            container, target_indices=[0], sublattice=self._sublattice(), element="Mg", n=1, seed=0, n_structures=2
        )
        assert len(container.get_defect_structures()) == 2

    def test_both_site_ids_and_n_raises(self):
        container = _fresh_container()
        with pytest.raises(ValueError, match="at most one"):
            create_interstitial_batch(
                container, target_indices=[0], sublattice=self._sublattice(), element="Mg", site_ids=[0], n=1
            )


class TestChaining:
    def test_defects_can_be_chained_on_top_of_each_other(self):
        container = _fresh_container()
        container = create_vacancy(container, atom_ids=[0])
        vacancy_idx = len(container) - 1
        container = create_substitution(container, to_element="Mg", atom_ids=[0], parent_defect_index=vacancy_idx)
        chained = container.get_structure(len(container) - 1)
        assert chained["generation"] == 2
        assert len(chained["events"]) == 2
        assert chained["events"][0]["type"] == "vacancy"
        assert chained["events"][1]["type"] == "substitution"
        assert len(chained["structure"]) == 3  # one removed, one substituted (still 3 atoms)
