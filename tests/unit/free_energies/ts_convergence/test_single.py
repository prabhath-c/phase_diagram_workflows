"""
Unit tests for the pure, disk-scanning/naming logic in
phase_diagram_workflows.free_energies.ts_convergence.single.

No calphy, no executor: just directory-name encoding/decoding for one
structure's bracket search.
"""

import os
import tempfile

from phase_diagram_workflows.free_energies.ts_convergence.single import (
    _bracket_log_path,
    _bracket_working_directory,
    _find_tried_brackets,
    _precompute_bracket_chain,
    _read_bracket_log,
    _record_bracket,
    load_bracket_history,
)


class TestBracketWorkingDirectory:
    def test_working_directory_encodes_bracket(self):
        wd = _bracket_working_directory("/root", 300.0, 1000.0)
        assert wd == "/root/T_300.00_1000.00"


class TestFindTriedBrackets:
    def test_empty_root_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _find_tried_brackets(os.path.join(tmpdir, "missing")) == []

    def test_finds_matching_subfolders_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "T_300.00_900.00"))
            os.makedirs(os.path.join(tmpdir, "T_350.00_900.00"))
            os.makedirs(os.path.join(tmpdir, "not_a_bracket"))  # doesn't match the marker

            tried = _find_tried_brackets(tmpdir)
            assert sorted(tried) == [(300.0, 900.0), (350.0, 900.0)]

    def test_skips_unparsable_folder_names_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "T_300.00_900.00"))
            # A manual backup copy with an extra underscore in the suffix.
            os.makedirs(os.path.join(tmpdir, "T_300.00_900.00_backup"))
            # Non-numeric suffix.
            os.makedirs(os.path.join(tmpdir, "T_abc_def"))

            tried = _find_tried_brackets(tmpdir)
            assert tried == [(300.0, 900.0)]


class TestBracketLog:
    """No calphy here either: `_record_bracket`/`load_bracket_history` only
    ever touch `bracket_log.csv`, never calphy output, so a bare tmpdir with
    no calphy directories at all is a valid target."""

    def test_load_bracket_history_with_no_log_and_no_disk_state_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tried, criteria = load_bracket_history(tmpdir)
            assert tried == []
            assert criteria == {}

    def test_record_bracket_creates_log_with_pending_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _record_bracket(tmpdir, (700.0, 720.0))

            assert os.path.isfile(_bracket_log_path(tmpdir))
            tried, criteria = load_bracket_history(tmpdir)
            assert tried == [(700.0, 720.0)]
            assert criteria == {}  # pending: no criterion recorded yet

    def test_record_bracket_then_resolving_criterion_updates_the_same_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _record_bracket(tmpdir, (700.0, 720.0))  # submitted, pending
            _record_bracket(tmpdir, (700.0, 720.0), criterion=0.5)  # result now known

            tried, criteria = load_bracket_history(tmpdir)
            assert tried == [(700.0, 720.0)]  # still one row, not two
            assert criteria == {(700.0, 720.0): 0.5}

    def test_load_bracket_history_falls_back_to_disk_scan_when_log_missing(self):
        # A working_directory_root with real calphy-shaped directories but
        # no bracket_log.csv (e.g. produced before this log existed) --
        # load_bracket_history must still recover the tried brackets by
        # rescanning, exactly like _find_tried_brackets does on its own.
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "T_300.00_900.00"))
            assert _read_bracket_log(tmpdir) is None

            tried, criteria = load_bracket_history(tmpdir)
            assert tried == [(300.0, 900.0)]
            assert criteria == {}  # no parseable calphy output in this bare directory


class TestPrecomputeBracketChain:
    def test_no_step_given_returns_only_current_bracket(self):
        assert _precompute_bracket_chain((700.0, 720.0), None, None, max_iterations=3) == [(700.0, 720.0)]

    def test_narrows_up_to_max_iterations(self):
        chain = _precompute_bracket_chain((700.0, 720.0), None, 5.0, max_iterations=2)
        assert chain == [(700.0, 720.0), (700.0, 715.0), (700.0, 710.0)]

    def test_collapse_truncates_the_chain_instead_of_raising(self):
        # (700, 710) narrowed by step_upper=10 would collapse to a
        # zero-width (700, 700) bracket -- base.step_bracket rejects that,
        # and this must stop the sequence there rather than propagate the
        # exception, however high max_iterations allows it to go.
        chain = _precompute_bracket_chain((700.0, 720.0), None, 10.0, max_iterations=5)
        assert chain == [(700.0, 720.0), (700.0, 710.0)]
