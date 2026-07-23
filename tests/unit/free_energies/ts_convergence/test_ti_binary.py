"""
Unit tests for the pure, disk-scanning/naming logic in
phase_diagram_workflows.free_energies.ts_convergence.ti_binary.

No calphy, no executor: just directory-name encoding/decoding.
"""

import os
import tempfile

import pandas as pd

from phase_diagram_workflows.free_energies.ts_convergence.ti_binary import (
    _bracket_prefix,
    _bracket_working_directory,
    _find_tried_brackets,
)


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

