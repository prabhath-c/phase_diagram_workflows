"""
Unit tests for phase_diagram_workflows.structures.point_defects.random_alloys.
"""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from phase_diagram_workflows.structures.point_defects.random_alloys import (
    generate_random_binary_structures,
    get_element_fractions,
)


class TestGetElementFractions:
    def test_all_elements(self):
        atoms = bulk("Al", cubic=True).repeat((1, 1, 1))
        atoms[0].symbol = "Mg"
        fractions = get_element_fractions(atoms)
        assert fractions == {"Al": 3 / 4, "Mg": 1 / 4}

    def test_single_element(self):
        atoms = bulk("Al", cubic=True)
        atoms[0].symbol = "Mg"
        assert get_element_fractions(atoms, element="Mg") == {"Mg": 1 / 4}

    def test_element_not_present_gives_zero(self):
        atoms = bulk("Al", cubic=True)
        assert get_element_fractions(atoms, element="Mg") == {"Mg": 0.0}


class TestGenerateRandomBinaryStructures:
    def test_default_concentrations(self):
        atoms = bulk("Al", cubic=True)  # 4 atoms
        df = generate_random_binary_structures(atoms, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert list(df["c_in"]) == [0, 0.5, 1]
        # c (achieved) matches c_in exactly for a 4-atom cell (all target
        # fractions land on an integer atom count).
        assert list(df["c"]) == [0.0, 0.5, 1.0]

    def test_stoichiometric_when_already_at_target(self):
        atoms = bulk("Al", cubic=True)
        df = generate_random_binary_structures(atoms, concentrations=[0.0], seed=0)
        assert df.iloc[0]["approximations"] == ["stoichiometric"]

    def test_antisites_reaches_exact_target_concentration(self):
        atoms = bulk("Al", cubic=True)  # 4 atoms, all Al
        df = generate_random_binary_structures(atoms, concentrations=[0.25], seed=0)
        row = df.iloc[0]
        assert row["c"] == pytest.approx(0.25)
        assert row["approximations"] == ["antisites"]
        assert row["atoms"].get_chemical_symbols().count("Mg") == 1

    def test_antisites_can_decrease_concentration(self):
        # Start fully Mg, target a lower concentration -- exercises the
        # delta_n < 0 (Mg -> Al) branch.
        atoms = bulk("Al", cubic=True)
        atoms.set_chemical_symbols(["Mg"] * 4)
        df = generate_random_binary_structures(atoms, concentrations=[0.25], seed=0)
        row = df.iloc[0]
        assert row["c"] == pytest.approx(0.25)

    def test_raises_when_not_enough_atoms_to_reach_concentration(self):
        atoms = bulk("Al", cubic=True)  # 4 atoms, all Al, no Mg to remove
        with pytest.raises(ValueError, match="Cannot reach concentration"):
            generate_random_binary_structures(atoms, concentrations=[-0.25], seed=0)

    def test_seed_reproducibility(self):
        atoms = bulk("Al", cubic=True).repeat((2, 2, 2))  # 32 atoms
        df1 = generate_random_binary_structures(atoms, concentrations=[0.5], seed=42)
        df2 = generate_random_binary_structures(atoms, concentrations=[0.5], seed=42)
        symbols1 = df1.iloc[0]["atoms"].get_chemical_symbols()
        symbols2 = df2.iloc[0]["atoms"].get_chemical_symbols()
        assert symbols1 == symbols2

    def test_different_seeds_can_differ(self):
        atoms = bulk("Al", cubic=True).repeat((2, 2, 2))  # 32 atoms
        df1 = generate_random_binary_structures(atoms, concentrations=[0.5], seed=1)
        df2 = generate_random_binary_structures(atoms, concentrations=[0.5], seed=2)
        symbols1 = df1.iloc[0]["atoms"].get_chemical_symbols()
        symbols2 = df2.iloc[0]["atoms"].get_chemical_symbols()
        assert symbols1 != symbols2

    def test_reshuffle_randomizes_assignment_for_later_entries(self):
        atoms = bulk("Al", cubic=True).repeat((2, 2, 2))  # 32 atoms
        df = generate_random_binary_structures(
            atoms, concentrations=[0.5, 0.5], approximations=["antisites", "reshuffle"], seed=0
        )
        # Both rows hit the same target concentration...
        assert df.iloc[0]["c"] == pytest.approx(0.5)
        assert df.iloc[1]["c"] == pytest.approx(0.5)
        # ...but the second (i > 0) is reshuffled with its own seed, so the
        # actual site assignment differs from the first.
        symbols0 = df.iloc[0]["atoms"].get_chemical_symbols()
        symbols1 = df.iloc[1]["atoms"].get_chemical_symbols()
        assert symbols0 != symbols1
        assert df.iloc[1]["approximations"] == ["reshuffle", "antisites"]

    def test_reshuffle_implies_antisites_when_not_given(self):
        atoms = bulk("Al", cubic=True).repeat((2, 2, 2))
        # approximations=["reshuffle"] only -- generate_random_binary_structures
        # should still add "antisites" internally so the first (i==0) row is
        # reachable via substitution.
        df = generate_random_binary_structures(atoms, concentrations=[0.5], approximations=["reshuffle"], seed=0)
        assert df.iloc[0]["c"] == pytest.approx(0.5)

    def test_does_not_mutate_caller_approximations_list(self):
        atoms = bulk("Al", cubic=True)
        approximations = ["antisites"]
        generate_random_binary_structures(atoms, concentrations=[0.25], approximations=approximations, seed=0)
        assert approximations == ["antisites"]

    def test_row_schema(self):
        atoms = bulk("Al", cubic=True)
        df = generate_random_binary_structures(
            atoms, main_element="Al", mixing_element="Mg", phase_type="fcc", reference_phase="solid",
            concentrations=[0.25], seed=0,
        )
        row = df.iloc[0]
        for key in (
            "symbol", "main_element", "mixing_element", "fractions", "c", "c_in",
            "atoms", "phase_type", "reference_phase", "approximations", "seed",
        ):
            assert key in row
        assert row["main_element"] == "Al"
        assert row["mixing_element"] == "Mg"
        assert row["phase_type"] == "fcc"
        assert row["reference_phase"] == "solid"
