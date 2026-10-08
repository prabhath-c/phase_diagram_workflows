"""
Unit tests for phase_diagram_workflows.structures.point_defects.random_antisites.
"""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from phase_diagram_workflows.structures.point_defects.random_antisites import (
    generate_antisite_structures,
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


def mixed_base(n_mg, repeat=(2, 2, 2)):
    """A 32-atom fcc cell with `n_mg` Mg atoms (the first ones) and the rest Al: stands in for an intermetallic."""
    atoms = bulk("Al", cubic=True).repeat(repeat)
    symbols = ["Al"] * len(atoms)
    symbols[:n_mg] = ["Mg"] * n_mg
    atoms.set_chemical_symbols(symbols)
    return atoms


class TestGenerateAntisiteStructures:
    def test_default_concentrations(self):
        df = generate_antisite_structures(bulk("Al", cubic=True), seed=0)  # 4 atoms
        assert list(df["c_in"]) == [0, 0.5, 1]
        assert list(df["c"]) == [0.0, 0.5, 1.0]

    def test_stoichiometric_when_already_at_target(self):
        atoms = bulk("Al", cubic=True)
        df = generate_antisite_structures(atoms, concentrations=[0.0], seed=0)
        row = df.iloc[0]
        assert row["approximations"] == ["stoichiometric"]
        assert row["atoms"] == atoms
        assert row["substituted_indices"] == []

    def test_reaches_exact_target_concentration(self):
        df = generate_antisite_structures(bulk("Al", cubic=True), concentrations=[0.25], seed=0)
        row = df.iloc[0]
        assert row["c"] == pytest.approx(0.25)
        assert row["approximations"] == ["antisites"]
        assert row["atoms"].get_chemical_symbols().count("Mg") == 1

    def test_can_decrease_concentration(self):
        atoms = bulk("Al", cubic=True)
        atoms.set_chemical_symbols(["Mg"] * 4)
        row = generate_antisite_structures(atoms, concentrations=[0.25], seed=0).iloc[0]
        assert row["c"] == pytest.approx(0.25)

    def test_raises_when_not_enough_atoms_to_reach_concentration(self):
        atoms = bulk("Al", cubic=True)  # no Mg to remove
        with pytest.raises(ValueError, match="Cannot reach concentration"):
            generate_antisite_structures(atoms, concentrations=[-0.25], seed=0)

    def test_raises_above_one(self):
        with pytest.raises(ValueError, match="Cannot reach concentration"):
            generate_antisite_structures(bulk("Al", cubic=True), concentrations=[1.5], seed=0)

    def test_seed_reproducibility(self):
        atoms = bulk("Al", cubic=True).repeat((2, 2, 2))
        first = generate_antisite_structures(atoms, concentrations=[0.5], seed=42).iloc[0]["atoms"]
        second = generate_antisite_structures(atoms, concentrations=[0.5], seed=42).iloc[0]["atoms"]
        assert first.get_chemical_symbols() == second.get_chemical_symbols()

    def test_different_seeds_can_differ(self):
        atoms = bulk("Al", cubic=True).repeat((2, 2, 2))
        first = generate_antisite_structures(atoms, concentrations=[0.5], seed=1).iloc[0]["atoms"]
        second = generate_antisite_structures(atoms, concentrations=[0.5], seed=2).iloc[0]["atoms"]
        assert first.get_chemical_symbols() != second.get_chemical_symbols()

    def test_does_not_mutate_the_base_structure(self):
        atoms = bulk("Al", cubic=True).repeat((2, 2, 2))
        before = atoms.copy()
        generate_antisite_structures(atoms, concentrations=[0.25, 0.5], seed=0)
        assert atoms == before

    def test_row_schema(self):
        df = generate_antisite_structures(
            bulk("Al", cubic=True), main_element="Al", mixing_element="Mg", phase_type="fcc",
            reference_phase="solid", concentrations=[0.25], seed=0,
        )
        row = df.iloc[0]
        for key in (
            "symbol", "main_element", "mixing_element", "fractions", "c", "c_in",
            "atoms", "phase_type", "reference_phase", "approximations", "seed",
        ):
            assert key in row
        assert (row["main_element"], row["mixing_element"]) == ("Al", "Mg")
        assert (row["phase_type"], row["reference_phase"]) == ("fcc", "solid")
        assert row["seed"] == 0

    def test_output_atoms_carry_no_container_bookkeeping(self):
        atoms = generate_antisite_structures(bulk("Al", cubic=True), concentrations=[0.25], seed=0).iloc[0]["atoms"]
        assert "uid" not in atoms.arrays

    # --- both sides of a base structure that already contains both elements ---

    def test_both_sides_of_the_base_composition(self):
        base = mixed_base(16)  # 16 of 32 atoms are Mg
        df = generate_antisite_structures(base, concentrations=[0.25, 0.5, 0.75], seed=0)
        assert [row.atoms.get_chemical_symbols().count("Mg") for row in df.itertuples()] == [8, 16, 24]
        assert list(df["approximations"].map(tuple)) == [("antisites",), ("stoichiometric",), ("antisites",)]

    def test_lower_side_only_turns_mixing_atoms_into_main_atoms(self):
        base = mixed_base(16)
        row = generate_antisite_structures(base, concentrations=[0.25], seed=0).iloc[0]
        base_symbols = base.get_chemical_symbols()
        assert len(row["substituted_indices"]) == 8
        assert all(base_symbols[i] == "Mg" and row["atoms"].symbols[i] == "Al" for i in row["substituted_indices"])

    def test_upper_side_only_turns_main_atoms_into_mixing_atoms(self):
        base = mixed_base(16)
        row = generate_antisite_structures(base, concentrations=[0.75], seed=0).iloc[0]
        base_symbols = base.get_chemical_symbols()
        assert len(row["substituted_indices"]) == 8
        assert all(base_symbols[i] == "Al" and row["atoms"].symbols[i] == "Mg" for i in row["substituted_indices"])

    def test_only_the_listed_atoms_change(self):
        base = mixed_base(10)
        row = generate_antisite_structures(base, concentrations=[0.6], seed=3).iloc[0]
        changed = [i for i, (a, b) in enumerate(zip(base.get_chemical_symbols(), row["atoms"].get_chemical_symbols())) if a != b]
        assert changed == row["substituted_indices"]
        assert (row["atoms"].positions == base.positions).all()

    def test_a_third_element_limits_what_can_be_substituted(self):
        # within a binary base every target in 0-1 is reachable; a third element occupies sites that
        # can be neither replaced nor filled, so full Mg would need more Al atoms than there are
        base = mixed_base(12)
        symbols = base.get_chemical_symbols()
        symbols[-8:] = ["Cu"] * 8   # 12 Mg, 12 Al, 8 Cu
        base.set_chemical_symbols(symbols)
        with pytest.raises(ValueError, match="not enough Al"):
            generate_antisite_structures(base, concentrations=[1.0], seed=0)

    # --- every structure is built on top of the previous one ---

    def test_upper_side_is_a_chain_each_structure_contains_the_previous_antisites(self):
        base = bulk("Al", cubic=True).repeat((3, 3, 3))  # 108 atoms
        df = generate_antisite_structures(base, concentrations=[0.05, 0.10, 0.20, 0.40], seed=11)
        substituted = [set(indices) for indices in df["substituted_indices"]]
        assert [len(indices) for indices in substituted] == [5, 11, 22, 43]
        for nearer, further in zip(substituted, substituted[1:]):
            assert nearer < further

    def test_lower_side_is_a_chain_each_structure_contains_the_previous_antisites(self):
        base = mixed_base(24)   # 24 of 32 atoms are Mg
        df = generate_antisite_structures(base, concentrations=[0.6, 0.4, 0.2], seed=5)
        substituted = [set(indices) for indices in df["substituted_indices"]]
        assert [len(indices) for indices in substituted] == [5, 11, 18]
        for nearer, further in zip(substituted, substituted[1:]):
            assert nearer < further

    def test_the_two_sides_of_the_base_are_separate_chains(self):
        base = mixed_base(16)
        df = generate_antisite_structures(base, concentrations=[0.25, 0.375, 0.5, 0.625, 0.75], seed=2)
        lower, upper = [set(df.iloc[i]["substituted_indices"]) for i in (0, 4)]
        assert set(df.iloc[1]["substituted_indices"]) < lower
        assert set(df.iloc[3]["substituted_indices"]) < upper
        assert df.iloc[2]["substituted_indices"] == []

    def test_the_chain_does_not_depend_on_the_order_of_the_concentrations(self):
        base = mixed_base(16)
        ascending = generate_antisite_structures(base, concentrations=[0.25, 0.5, 0.75, 0.9], seed=7)
        shuffled = generate_antisite_structures(base, concentrations=[0.9, 0.25, 0.75, 0.5], seed=7)
        for c in (0.25, 0.5, 0.75, 0.9):
            first = ascending[ascending["c_in"] == c].iloc[0]["atoms"]
            second = shuffled[shuffled["c_in"] == c].iloc[0]["atoms"]
            assert first.get_chemical_symbols() == second.get_chemical_symbols()

    def test_targets_beyond_a_structure_do_not_change_it(self):
        base = mixed_base(16)
        short = generate_antisite_structures(base, concentrations=[0.6, 0.7], seed=7)
        longer = generate_antisite_structures(base, concentrations=[0.6, 0.7, 0.9], seed=7)
        for i in (0, 1):
            assert short.iloc[i]["atoms"].get_chemical_symbols() == longer.iloc[i]["atoms"].get_chemical_symbols()

    def test_an_inserted_target_changes_the_structures_beyond_it(self):
        base = bulk("Al", cubic=True).repeat((3, 3, 3))
        without = generate_antisite_structures(base, concentrations=[0.1, 0.3], seed=1).iloc[1]
        with_middle = generate_antisite_structures(base, concentrations=[0.1, 0.2, 0.3], seed=1).iloc[2]
        assert without["c"] == with_middle["c"]
        assert without["atoms"].get_chemical_symbols() != with_middle["atoms"].get_chemical_symbols()

    def test_recorded_seed_reproduces_an_unseeded_run(self):
        base = mixed_base(16)
        first = generate_antisite_structures(base, concentrations=[0.75], seed=None).iloc[0]
        again = generate_antisite_structures(base, concentrations=[0.75], seed=int(first["seed"])).iloc[0]
        assert first["atoms"].get_chemical_symbols() == again["atoms"].get_chemical_symbols()

    def test_substitutions_are_spread_over_all_candidate_sites(self):
        # 200 different seeds: every one of the 32 Al sites must get substituted at least once,
        # and none far more often than the 8/32 chance would give (a stuck or biased draw would show here)
        base = bulk("Al", cubic=True).repeat((2, 2, 2))
        counts = np.zeros(len(base), dtype=int)
        for seed in range(200):
            counts[generate_antisite_structures(base, concentrations=[0.25], seed=seed).iloc[0]["substituted_indices"]] += 1
        assert counts.min() > 0
        assert counts.max() < 2 * 200 * 8 / 32

    # --- input validation ---

    def test_same_main_and_mixing_element_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            generate_antisite_structures(bulk("Al", cubic=True), main_element="Al", mixing_element="Al", concentrations=[0.5], seed=0)

    def test_negative_seed_raises_whatever_the_composition(self):
        base = bulk("Al", cubic=True).repeat((2, 2, 2))
        for concentration in (0.0, 0.5, 1.0):   # composition-independent: would otherwise depend on seed + n
            with pytest.raises(ValueError, match="non-negative"):
                generate_antisite_structures(base, concentrations=[concentration], seed=-5)

    def test_float_seed_raises_type_error(self):
        with pytest.raises(TypeError):
            generate_antisite_structures(bulk("Al", cubic=True), concentrations=[0.5], seed=6919.0)

    def test_numpy_integer_seed_is_accepted(self):
        base = bulk("Al", cubic=True).repeat((2, 2, 2))
        as_int = generate_antisite_structures(base, concentrations=[0.5], seed=5).iloc[0]["atoms"]
        as_numpy = generate_antisite_structures(base, concentrations=[0.5], seed=np.int64(5)).iloc[0]["atoms"]
        assert as_int.get_chemical_symbols() == as_numpy.get_chemical_symbols()

    def test_repeated_concentration_gives_the_same_structure_each_time(self):
        base = bulk("Al", cubic=True).repeat((2, 2, 2))
        df = generate_antisite_structures(base, concentrations=[0.5, 0.25, 0.5], seed=3)
        assert df.iloc[0]["atoms"].get_chemical_symbols() == df.iloc[2]["atoms"].get_chemical_symbols()
        assert df.iloc[1]["atoms"].symbols.count("Mg") == 8
