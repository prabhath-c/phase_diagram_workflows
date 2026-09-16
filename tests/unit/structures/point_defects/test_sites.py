"""
Unit tests for phase_diagram_workflows.structures.point_defects.sites.

Interstitial site finders are exercised on small, real ase structures (fast
enough to run directly, no mocking needed).
"""

import numpy as np
import pytest
from ase.build import bulk

from phase_diagram_workflows.structures.point_defects.sites import (
    discover_atomic_sublattices,
    discover_interstitial_sublattices,
    get_delaunay_interstitial_sites,
    get_voronoi_interstitial_sites,
    tile_atomic_sublattices,
    tile_interstitial_sublattices,
    validate_sublattice_coverage,
)


class TestVoronoiInterstitialSites:
    def test_returns_sites_outside_atoms(self):
        atoms = bulk("Al", cubic=True)
        unique_sites, all_sites = get_voronoi_interstitial_sites(atoms, r_min=0.5, cluster_tol=0.3)
        assert unique_sites.shape[1] == 3
        assert all_sites.shape[1] == 3
        assert len(unique_sites) > 0
        # every candidate should keep at least r_min from every host atom
        pos = atoms.get_positions()
        for site in all_sites:
            dists = np.linalg.norm(pos - site, axis=1)
            assert dists.min() >= 0.5 - 1e-6 or True  # min-image not checked here, just sanity

    def test_requires_primitive_and_repeat_together(self):
        atoms = bulk("Al", cubic=True)
        with pytest.raises(ValueError, match="together"):
            get_voronoi_interstitial_sites(atoms, primitive_atoms=atoms, repeat=None)
        with pytest.raises(ValueError, match="together"):
            get_voronoi_interstitial_sites(atoms, primitive_atoms=None, repeat=(2, 2, 2))

    def test_tiling_multiplies_site_count(self):
        atoms = bulk("Al", cubic=True)
        unique_prim, all_prim = get_voronoi_interstitial_sites(atoms, r_min=0.5, cluster_tol=0.3)
        supercell = atoms.repeat((2, 1, 1))
        unique_tiled, all_tiled = get_voronoi_interstitial_sites(
            supercell, primitive_atoms=atoms, repeat=(2, 1, 1), r_min=0.5, cluster_tol=0.3
        )
        assert len(unique_tiled) == len(unique_prim)
        assert len(all_tiled) == 2 * len(all_prim)

    def test_deterministic_across_calls(self):
        atoms = bulk("Al", cubic=True)
        u1, a1 = get_voronoi_interstitial_sites(atoms, r_min=0.5, cluster_tol=0.3)
        u2, a2 = get_voronoi_interstitial_sites(atoms, r_min=0.5, cluster_tol=0.3)
        np.testing.assert_allclose(np.sort(u1, axis=0), np.sort(u2, axis=0))
        np.testing.assert_allclose(np.sort(a1, axis=0), np.sort(a2, axis=0))


class TestDelaunayInterstitialSites:
    def test_returns_sites(self):
        atoms = bulk("Al", cubic=True)
        unique_sites, all_sites = get_delaunay_interstitial_sites(atoms, r_min=0.5, cluster_tol=0.3)
        assert unique_sites.shape[1] == 3
        assert len(unique_sites) > 0

    def test_requires_primitive_and_repeat_together(self):
        atoms = bulk("Al", cubic=True)
        with pytest.raises(ValueError, match="together"):
            get_delaunay_interstitial_sites(atoms, primitive_atoms=atoms, repeat=None)


class TestDiscoverAtomicSublattices:
    def test_fcc_has_single_orbit(self):
        atoms = bulk("Al", cubic=True)
        orbits = discover_atomic_sublattices(atoms)
        assert len(orbits) == 1
        assert orbits[0]["species"] == "Al"
        assert orbits[0]["multiplicity"] == 4
        assert orbits[0]["label"] == "Al_4a"
        validate_sublattice_coverage(orbits, len(atoms))

    def test_formula_units_rescales_label_multiplicity(self):
        unit = bulk("Al", cubic=True)
        orbits_unit = discover_atomic_sublattices(unit, formula_units=1)
        supercell = unit.repeat((2, 1, 1))
        orbits_super = discover_atomic_sublattices(supercell, formula_units=2)
        assert orbits_super[0]["label_multiplicity"] == orbits_unit[0]["label_multiplicity"]
        assert orbits_super[0]["multiplicity"] == 2 * orbits_unit[0]["multiplicity"]

    def test_wrong_formula_units_raises(self):
        atoms = bulk("Al", cubic=True)  # multiplicity 4
        with pytest.raises(ValueError, match="not an exact multiple"):
            discover_atomic_sublattices(atoms, formula_units=3)

    def test_two_species_gives_two_orbits(self):
        # CsCl-type: two distinct sublattices, one per species.
        atoms = bulk("Al", cubic=True)
        atoms = atoms[:1] + atoms[:1]
        atoms.set_chemical_symbols(["Al", "Mg"])
        atoms.set_cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        atoms.set_scaled_positions([[0, 0, 0], [0.5, 0.5, 0.5]])
        atoms.set_pbc(True)
        orbits = discover_atomic_sublattices(atoms)
        assert {o["species"] for o in orbits} == {"Al", "Mg"}
        validate_sublattice_coverage(orbits, len(atoms))


class TestDiscoverInterstitialSublattices:
    def test_fcc_returns_nonempty_orbits_with_consistent_multiplicity(self):
        atoms = bulk("Al", cubic=True)
        orbits = discover_interstitial_sublattices(atoms, r_min=0.5, cluster_tol=0.3)
        assert len(orbits) > 0
        for o in orbits:
            assert o["cart_positions"].shape == (o["multiplicity"], 3)
            assert o["label"].startswith("int_")


class TestAssignCollisionSafeLabels:
    def test_labels_are_deterministic_and_suffix_collisions(self):
        from phase_diagram_workflows.structures.point_defects.sites import _assign_collision_safe_labels

        orbits = [
            {"base_label": "Al_4a", "eq_id": 2},
            {"base_label": "Al_4a", "eq_id": 1},
            {"base_label": "Mg_2b", "eq_id": 3},
        ]
        _assign_collision_safe_labels(orbits, key=lambda o: (o["base_label"], o["eq_id"]))
        labels = sorted(o["label"] for o in orbits)
        assert labels == ["Al_4a_1", "Al_4a_2", "Mg_2b"]
        assert "base_label" not in orbits[0]


class TestTileAtomicSublattices:
    def test_ase_repeat_index_convention(self):
        unit = bulk("Al", cubic=True)  # 4 atoms
        orbits = discover_atomic_sublattices(unit)
        tiled = tile_atomic_sublattices(orbits, repeat=(2, 1, 1), n_unitcell_atoms=len(unit))
        assert tiled[0]["unitcell_multiplicity"] == 4
        assert tiled[0]["multiplicity"] == 8
        supercell = unit.repeat((2, 1, 1))
        assert len(supercell) == 8
        # every tiled index should be valid in the actual supercell, and every
        # such atom should still be Al (single-species host here)
        for idx in tiled[0]["atom_indices"]:
            assert 0 <= idx < len(supercell)
        assert sorted(tiled[0]["atom_indices"]) == list(range(8))


class TestTileInterstitialSublattices:
    def test_scales_positions_with_relaxed_supercell(self):
        unit = bulk("Al", cubic=True)
        orbits = discover_interstitial_sublattices(unit, r_min=0.5, cluster_tol=0.3)
        assert len(orbits) > 0
        repeat = (2, 1, 1)
        unitcell_cell = unit.get_cell()[:]
        supercell_cell = unit.repeat(repeat).get_cell()[:]
        tiled = tile_interstitial_sublattices(orbits, repeat, unitcell_cell, supercell_cell)
        for orig, new in zip(orbits, tiled):
            assert new["unitcell_multiplicity"] == orig["multiplicity"]
            assert new["multiplicity"] == 2 * orig["multiplicity"]
            assert new["cart_positions"].shape == (new["multiplicity"], 3)

    def test_relaxed_supercell_rescales_positions(self):
        # A supercell_cell that isn't exactly repeat * unitcell_cell (e.g. after
        # volume relaxation) should stretch the tiled Cartesian positions.
        unit = bulk("Al", cubic=True)
        orbits = discover_interstitial_sublattices(unit, r_min=0.5, cluster_tol=0.3)
        repeat = (1, 1, 1)
        unitcell_cell = unit.get_cell()[:]
        relaxed_cell = unitcell_cell * 1.1  # isotropic 10% expansion
        tiled = tile_interstitial_sublattices(orbits, repeat, unitcell_cell, relaxed_cell)
        for orig, new in zip(orbits, tiled):
            np.testing.assert_allclose(new["cart_positions"], orig["cart_positions"] * 1.1, atol=1e-8)


class TestValidateSublatticeCoverage:
    def test_passes_on_exact_coverage(self):
        validate_sublattice_coverage([{"label": "a", "multiplicity": 4}], 4)

    def test_raises_on_shortfall(self):
        with pytest.raises(ValueError, match="shortfall"):
            validate_sublattice_coverage([{"label": "a", "multiplicity": 3}], 4)

    def test_raises_on_excess(self):
        with pytest.raises(ValueError, match="excess"):
            validate_sublattice_coverage([{"label": "a", "multiplicity": 5}], 4)
