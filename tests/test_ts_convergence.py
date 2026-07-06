"""
Unit tests for the ts_convergence package: the purpose-agnostic core in
base.py, and the binary-alloy layer built on it in ti_binary.py.

Tests that touch calphy run a real, small, fast calculation (Al, EAM
potential via lammpsparser's iprpy database, ~100 equilibration/switching
steps) instead of fabricating result files -- so they exercise the real
forward/backward TI output format, not a guessed approximation of it.
"""

import matplotlib
matplotlib.use("Agg")

import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor

import pandas as pd
import pytest
from ase.build import bulk
from lammpsparser import get_potential_by_name

from phase_diagram_workflows.free_energies.ts_convergence.base import (
    check_ts_overlap,
    resolve_current_bracket,
    step_bracket,
    ts_overlap_criterion,
)
from phase_diagram_workflows.free_energies.ts_convergence.ti_binary import (
    _bracket_prefix,
    _bracket_working_directory,
    _find_tried_brackets,
    build_criterion_table,
    get_unique_criterion_table,
    plot_criterion_by_bracket,
    plot_criterion_vs_concentration,
    refine_temperature_bracket,
)


class TestCheckTsOverlap:
    def test_overlapping_within_tolerance(self):
        assert check_ts_overlap([1.0, 2.0, 3.0], [3.0, 2.0, 1.05], tolerance=0.1) is True

    def test_not_overlapping_outside_tolerance(self):
        assert check_ts_overlap([1.0, 2.0, 3.0], [3.0, 2.0, 1.5], tolerance=0.1) is False

    def test_boundary_is_converged(self):
        assert check_ts_overlap([1.0], [1.1], tolerance=0.11) is True


class TestStepBracket:
    def test_step_upper_only(self):
        assert step_bracket(300, 1000, step_upper=100) == (300, 900)

    def test_step_lower_only(self):
        assert step_bracket(300, 1000, step_lower=50) == (350, 1000)

    def test_step_both_independently(self):
        assert step_bracket(300, 1000, step_lower=50, step_upper=100) == (350, 900)

    def test_no_step_given_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            step_bracket(300, 1000)

    def test_collapsed_bracket_raises(self):
        with pytest.raises(ValueError, match="collapsed"):
            step_bracket(300, 350, step_upper=100)


class TestResolveCurrentBracket:
    def test_no_prior_attempts_returns_initial(self):
        assert resolve_current_bracket((300, 1000), []) == (300, 1000)

    def test_upper_bound_narrows_to_most_restrictive(self):
        assert resolve_current_bracket((300, 1000), [(300, 900), (300, 800)]) == (300, 800)

    def test_lower_bound_narrows_to_most_restrictive(self):
        assert resolve_current_bracket((300, 1000), [(350, 1000), (400, 1000)]) == (400, 1000)

    def test_both_bounds_narrow_independently(self):
        assert resolve_current_bracket((300, 1000), [(350, 900)]) == (350, 900)

    def test_bounds_narrowed_past_each_other_raises(self):
        # Neither (600, 1000) nor (300, 500) was ever run as a single
        # attempt; combining their most-restrictive bounds independently
        # gives (600, 500), which is inverted.
        with pytest.raises(ValueError, match="narrowed past each other"):
            resolve_current_bracket((300, 1000), [(600, 1000), (300, 500)])


class TestBracketPrefixAndWorkingDirectory:
    def test_prefix_uses_six_decimals_by_default(self):
        row = pd.Series({
            "main_element": "Al", "mixing_element": "Mg",
            "phase_type": "fcc", "reference_phase": "solid", "c_in": 0.125,
        })
        assert _bracket_prefix(row) == "AlMg_fcc_solid_0.125000"

    def test_prefix_respects_conc_decimals(self):
        row = pd.Series({
            "main_element": "Al", "mixing_element": "Mg",
            "phase_type": "liquid", "reference_phase": "liquid", "c_in": 0.5,
        })
        assert _bracket_prefix(row, conc_decimals=2) == "AlMg_liquid_liquid_0.50"

    def test_working_directory_encodes_bracket(self):
        wd = _bracket_working_directory("/root", "AlMg_fcc_solid_0.125000", 300.0, 1000.0)
        assert wd == "/root/AlMg_fcc_solid_0.125000_T_300.00_1000.00"


class TestFindTriedBrackets:
    def test_empty_root_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _find_tried_brackets(os.path.join(tmpdir, "missing"), "prefix") == []

    def test_finds_matching_subfolders_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "AlMg_fcc_solid_0.125000_T_300.00_900.00"))
            os.makedirs(os.path.join(tmpdir, "AlMg_fcc_solid_0.125000_T_350.00_900.00"))
            os.makedirs(os.path.join(tmpdir, "AlMg_hcp_solid_0.125000_T_300.00_900.00"))  # different prefix

            tried = _find_tried_brackets(tmpdir, "AlMg_fcc_solid_0.125000")
            assert sorted(tried) == [(300.0, 900.0), (350.0, 900.0)]

    def test_skips_unparsable_folder_names_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "AlMg_fcc_solid_0.125000_T_300.00_900.00"))
            # A manual backup copy with an extra underscore in the suffix.
            os.makedirs(os.path.join(tmpdir, "AlMg_fcc_solid_0.125000_T_300.00_900.00_backup"))
            # Non-numeric suffix.
            os.makedirs(os.path.join(tmpdir, "AlMg_fcc_solid_0.125000_T_abc_def"))

            tried = _find_tried_brackets(tmpdir, "AlMg_fcc_solid_0.125000")
            assert tried == [(300.0, 900.0)]


class EagerExecutor:
    """Executes the submitted function immediately, in-process.

    Not a mock: `submit` genuinely calls the real function (e.g.
    calc_free_energy_with_calphy) right away and wraps its real result in a
    real, already-completed concurrent.futures.Future. Only the scheduling is
    fake; the calphy execution behind it is real.
    """

    def __init__(self):
        self.submit_calls = []

    def submit(self, fn, **kwargs):
        self.submit_calls.append(kwargs)
        future = Future()
        try:
            future.set_result(fn(**kwargs))
        except Exception as exc:  # pragma: no cover - surfaced via future.result()
            future.set_exception(exc)
        return future


def _make_row(atoms, c_in=0.1, phase_type="fcc", reference_phase="solid"):
    return pd.Series({
        "main_element": "Al", "mixing_element": "Mg",
        "phase_type": phase_type, "reference_phase": reference_phase,
        "c_in": c_in, "atoms": atoms,
    })


@pytest.fixture(scope="module")
def potential_df():
    pot = get_potential_by_name("1999--Mishin-Y--Al--LAMMPS--ipr1")
    return pot.to_frame().transpose()


@pytest.fixture(scope="module")
def small_structure():
    return bulk("Al", cubic=True).repeat(2)  # 32 atoms: small and fast


@pytest.fixture
def base_ts_params():
    return {
        "mode": "ts",
        "pressure": 0,
        "n_equilibration_steps": 100,
        "n_switching_steps": 100,
        "n_print_steps": 25,
        "equilibration_control": "berendsen",
        "md": {"thermostat_damping": 0.5},
        "tolerance": {"spring_constant": 0.01, "pressure": 0.5},
        "queue": {"cores": 1, "scheduler": "local"},
        "reference_phase": "solid",
        "file_format": "lammps-data",
    }


class TestRefineTemperatureBracketReal:
    def test_converges_with_generous_tolerance(self, small_structure, potential_df, base_ts_params, tmp_path):
        row = _make_row(small_structure, c_in=0.1)
        executor = EagerExecutor()

        result = refine_temperature_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=executor,
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,
            step_upper=10.0,
        )

        assert result["status"] == "converged"
        assert result["bracket"] == (700.0, 720.0)
        assert len(executor.submit_calls) == 1

    def test_resubmits_then_resolves_narrowed_bracket_on_next_call(
        self, small_structure, potential_df, base_ts_params, tmp_path
    ):
        row = _make_row(small_structure, c_in=0.2)

        first = refine_temperature_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1e-8,  # unreachable: forces resubmission
            step_upper=10.0,
        )
        assert first["status"] == "resubmitted"
        assert first["bracket"] == (700.0, 710.0)

        # Fresh call (simulating a new session): should resolve straight to
        # the narrowed bracket already on disk, not restart from (700, 720).
        second = refine_temperature_bracket(
            row=row,
            calphy_parameters=base_ts_params,
            potential_df=potential_df,
            executor=EagerExecutor(),
            working_directory_root=str(tmp_path),
            initial_bracket=(700.0, 720.0),
            tolerance=1.0,
            step_upper=10.0,
        )
        assert second["bracket"] == (700.0, 710.0)
        assert second["status"] == "converged"

    def test_pending_with_real_background_executor(self, small_structure, potential_df, base_ts_params, tmp_path):
        row = _make_row(small_structure, c_in=0.3)
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            result = refine_temperature_bracket(
                row=row,
                calphy_parameters=base_ts_params,
                potential_df=potential_df,
                executor=executor,
                working_directory_root=str(tmp_path),
                initial_bracket=(700.0, 720.0),
                tolerance=1.0,
                step_upper=10.0,
            )
            assert result["status"] == "pending"
        finally:
            executor.shutdown(wait=True)


@pytest.fixture(scope="module")
def criterion_sweep(small_structure, potential_df, tmp_path_factory):
    """Runs one real, small sweep: one concentration that converges
    immediately, one that needs a single resubmission. Shared across the
    criterion-table/plotting tests so the real calphy calls happen once.
    """
    working_directory_root = str(tmp_path_factory.mktemp("criterion_sweep"))
    base_params = {
        "mode": "ts",
        "pressure": 0,
        "n_equilibration_steps": 100,
        "n_switching_steps": 100,
        "n_print_steps": 25,
        "equilibration_control": "berendsen",
        "md": {"thermostat_damping": 0.5},
        "tolerance": {"spring_constant": 0.01, "pressure": 0.5},
        "queue": {"cores": 1, "scheduler": "local"},
        "reference_phase": "solid",
        "file_format": "lammps-data",
    }

    structures_df = pd.DataFrame([
        _make_row(small_structure, c_in=0.1),
        _make_row(small_structure, c_in=0.2),
    ])

    refine_temperature_bracket(
        row=structures_df.iloc[0],
        calphy_parameters=base_params,
        potential_df=potential_df,
        executor=EagerExecutor(),
        working_directory_root=working_directory_root,
        initial_bracket=(700.0, 720.0),
        tolerance=1.0,  # converges immediately
        step_upper=10.0,
    )
    refine_temperature_bracket(
        row=structures_df.iloc[1],
        calphy_parameters=base_params,
        potential_df=potential_df,
        executor=EagerExecutor(),
        working_directory_root=working_directory_root,
        initial_bracket=(700.0, 720.0),
        tolerance=1e-8,  # forces exactly one resubmission
        step_upper=10.0,
    )

    return structures_df, working_directory_root


class TestCriterionTableAndPlots:
    def test_build_criterion_table(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)

        expected_columns = {
            "main_element", "mixing_element", "phase_type", "reference_phase",
            "c_in", "t_low", "t_high", "criterion", "working_directory",
        }
        assert expected_columns.issubset(table.columns)

        converged_rows = table[table["c_in"] == 0.1]
        resubmitted_rows = table[table["c_in"] == 0.2]
        assert len(converged_rows) == 1
        assert len(resubmitted_rows) == 2
        assert converged_rows["criterion"].notna().all()
        assert resubmitted_rows["criterion"].notna().all()
        assert set(zip(resubmitted_rows["t_low"], resubmitted_rows["t_high"])) == {
            (700.0, 720.0), (700.0, 710.0),
        }

    def test_get_unique_criterion_table_picks_current_bracket(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)
        unique = get_unique_criterion_table(table, initial_bracket=(700.0, 720.0))

        assert len(unique) == 2
        narrowed_row = unique[unique["c_in"] == 0.2].iloc[0]
        assert (narrowed_row["t_low"], narrowed_row["t_high"]) == (700.0, 710.0)

    def test_plot_criterion_vs_concentration(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)
        unique = get_unique_criterion_table(table, initial_bracket=(700.0, 720.0))

        fig, ax = plot_criterion_vs_concentration(unique, tolerance=0.5)
        assert len(ax.collections) == 1  # the scatter
        assert len(ax.lines) == 1  # the tolerance reference line

    def test_plot_criterion_by_bracket(self, criterion_sweep):
        structures_df, working_directory_root = criterion_sweep
        table = build_criterion_table(structures_df, working_directory_root)

        fig, ax = plot_criterion_by_bracket(table, group_by="t_high", tolerance=0.5)
        # distinct t_high values across the whole table: 720 (both rows) and 710 (row2 only)
        assert len(ax.lines) == 2 + 1  # one line per t_high group, plus the tolerance line
