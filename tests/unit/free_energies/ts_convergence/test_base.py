"""
Unit tests for the purpose-agnostic core in
phase_diagram_workflows.free_energies.ts_convergence.base.

Pure decision logic only: no executor, no disk I/O, no calphy anywhere in
this file.
"""

import pytest

from phase_diagram_workflows.free_energies.ts_convergence.base import (
    check_ts_overlap,
    decide_next_bracket,
    pick_best_converged_bracket,
    resolve_current_bracket,
    step_bracket,
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


class TestDecideNextBracket:
    """Pure decision logic: no executor involved anywhere in this class."""

    def test_unknown_criterion_returns_current_bracket_unchanged(self):
        assert decide_next_bracket((300, 1000), None, tolerance=0.005) == (300, 1000)

    def test_within_tolerance_returns_none(self):
        assert decide_next_bracket((300, 1000), criterion=0.003, tolerance=0.005) is None

    def test_exceeds_tolerance_returns_narrowed_bracket(self):
        assert decide_next_bracket(
            (300, 1000), criterion=0.02, tolerance=0.005, step_upper=100
        ) == (300, 900)

    def test_same_criterion_different_tolerances_only_changes_the_decision(self):
        # The exact scenario from the notebook: nothing about the bracket or
        # its criterion changes -- only the tolerance a human is trying out.
        current_bracket, criterion = (700.0, 900.0), 0.0212
        assert decide_next_bracket(current_bracket, criterion, tolerance=0.005, step_upper=100.0) == (700.0, 800.0)
        assert decide_next_bracket(current_bracket, criterion, tolerance=0.03, step_upper=100.0) is None



class TestPickBestConvergedBracket:
    """Pure decision logic: given precomputed criteria, no disk/executor involved."""

    def test_no_candidates_satisfy_returns_none(self):
        criteria = {(700.0, 900.0): 0.02, (700.0, 800.0): 0.015}
        assert pick_best_converged_bracket(criteria, tolerance=0.005) is None

    def test_single_candidate_satisfies(self):
        criteria = {(700.0, 900.0): 0.02, (700.0, 800.0): 0.003}
        assert pick_best_converged_bracket(criteria, tolerance=0.005) == (700.0, 800.0)

    def test_prefers_widest_among_multiple_satisfying_candidates(self):
        # A wider bracket that already satisfies tolerance should win over a
        # narrower one that also satisfies it -- there's no reason to trust
        # the extra narrowing once a cheaper, wider bracket already sufficed.
        criteria = {(700.0, 900.0): 0.003, (700.0, 800.0): 0.001, (700.0, 750.0): 0.02}
        assert pick_best_converged_bracket(criteria, tolerance=0.005) == (700.0, 900.0)

    def test_empty_criteria_returns_none(self):
        assert pick_best_converged_bracket({}, tolerance=0.005) is None
