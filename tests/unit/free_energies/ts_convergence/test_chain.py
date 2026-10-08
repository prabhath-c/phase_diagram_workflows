"""
Unit tests for `refine_temperature_bracket_with_chain` (precomputed dependency chain).

calphy is stubbed at the two calls `single` makes to it; the executor stand-in resolves Future
arguments the way executorlib's cluster executors do (each step receives the previous step's
result). Everything else -- the precomputed sequence, pass-through after convergence, the
`max_iterations` cap, the bracket log -- is real code on a real temporary directory.
"""

import collections
import inspect
import os
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

import phase_diagram_workflows.free_energies.ts_convergence.single as single_module
from phase_diagram_workflows.free_energies.ts_convergence.single import (
    _precompute_bracket_chain,
    load_bracket_history,
    refine_temperature_bracket_with_chain,
)


class DependencyResolvingExecutor:
    """Runs each submitted function immediately, first replacing Future arguments by their results."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, **kwargs):
        self.submitted += 1
        resolved = {key: (value.result() if isinstance(value, Future) else value) for key, value in kwargs.items()}
        future = Future()
        future.set_result(fn(**resolved))
        return future


class Calphy:
    """Stand-in for calphy: which brackets converge is decided by `criteria[t_high]`."""

    def __init__(self, criteria):
        self.criteria = criteria
        self.runs = []  # (t_low, t_high)
        self.finished = {}

    def _result(self, working_directory):
        criterion = self.finished[working_directory]
        return pd.DataFrame(
            [{
                "status": True,
                "forward_energy_diff": [np.array([0.0, 0.0])],
                "backward_energy_diff": [np.array([0.0, criterion])],
            }]
        )

    def calc(self, input_structure, potential_df, calphy_parameters, working_directory):
        t_low, t_high = calphy_parameters["temperature"]
        self.runs.append((t_low, t_high))
        os.makedirs(working_directory, exist_ok=True)
        self.finished[working_directory] = self.criteria[t_high]
        return None, self._result(working_directory)

    def gather(self, working_directory):
        if working_directory not in self.finished:
            raise FileNotFoundError(working_directory)
        return self._result(working_directory)


@pytest.fixture
def calphy(monkeypatch):
    def install(criteria):
        fake = Calphy(criteria)
        monkeypatch.setattr(single_module, "calc_free_energy_with_calphy", fake.calc)
        monkeypatch.setattr(single_module, "gather_calphy_results_detailed", fake.gather)
        return fake

    return install


@pytest.fixture
def setup(tmp_path):
    return dict(
        input_structure=bulk("Al", cubic=True),
        calphy_parameters={"mode": "ts", "reference_phase": "solid", "temperature": [0, 0], "n_switching_steps": 80000},
        potential_df=pd.DataFrame({"Config": [["pair_style eam\n", "pair_coeff * *\n"]], "Species": [["Al"]]}),
        executor=DependencyResolvingExecutor(),
        working_directory_root=str(tmp_path / "structure"),
        initial_bracket=(100.0, 1000.0),
        tolerance=0.0075,
        step_upper=10.0,
    )


class TestChainWithThreeIterations:
    def test_never_converging_structure_runs_first_bracket_plus_three_narrowing_steps(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        futures, _ = refine_temperature_bracket_with_chain(**setup, max_iterations=3)

        assert len(futures) == 4
        assert [run[1] for run in fake.runs] == [1000.0, 990.0, 980.0, 970.0]
        assert futures[-1].result()["status"] == "narrowed"  # the chain ended without meeting the tolerance
        assert len(load_bracket_history(setup["working_directory_root"])[1]) == 4

    def test_steps_after_convergence_run_nothing(self, calphy, setup):
        fake = calphy({1000.0: 0.03, 990.0: 0.001})
        futures, _ = refine_temperature_bracket_with_chain(**setup, max_iterations=3)

        assert len(futures) == 4  # the whole chain is submitted up front ...
        assert [run[1] for run in fake.runs] == [1000.0, 990.0]  # ... but only two brackets actually ran
        assert futures[-1].result()["status"] == "converged"
        assert futures[-1].result()["bracket"] == (100.0, 990.0)

    def test_collapse_shortens_the_chain_instead_of_raising(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        setup.update(initial_bracket=(700.0, 720.0))
        futures, _ = refine_temperature_bracket_with_chain(**setup, max_iterations=3)
        assert [run[1] for run in fake.runs] == [720.0, 710.0]  # a third step would collapse to (700, 700)
        assert len(futures) == 2


class TestIterationCap:
    def test_default_cap_is_five(self):
        assert inspect.signature(refine_temperature_bracket_with_chain).parameters["max_iterations"].default == 5

    def test_default_cap_applies_when_not_given(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        futures, _ = refine_temperature_bracket_with_chain(**setup)
        assert len(futures) == 6  # first bracket + 5 narrowing steps
        assert len(fake.runs) == 6

    @pytest.mark.parametrize("bad", [-1, 2.5, True, "3"])
    def test_anything_but_a_non_negative_integer_or_none_is_refused(self, setup, bad):
        with pytest.raises(ValueError, match="max_iterations"):
            refine_temperature_bracket_with_chain(**setup, max_iterations=bad)

    def test_none_runs_until_the_bracket_collapses_and_warns(self, calphy, setup):
        fake = calphy(collections.defaultdict(lambda: 1.0))
        setup.update(initial_bracket=(700.0, 740.0))
        with pytest.warns(UserWarning, match="max_iterations=None"):
            futures, _ = refine_temperature_bracket_with_chain(**setup, max_iterations=None)
        assert [run[1] for run in fake.runs] == [740.0, 730.0, 720.0, 710.0]

    def test_none_with_a_step_that_would_never_end_is_refused_before_anything_is_submitted(self, calphy, setup):
        calphy(collections.defaultdict(lambda: 1.0))
        setup.update(initial_bracket=(0.0, 10.0), step_upper=0.001)
        with pytest.warns(UserWarning, match="max_iterations=None"):
            with pytest.raises(ValueError, match="more than 1000 brackets"):
                refine_temperature_bracket_with_chain(**setup, max_iterations=None)
        assert setup["executor"].submitted == 0

    def test_precompute_with_none_stops_at_the_collapse(self):
        assert _precompute_bracket_chain((700.0, 720.0), None, 10.0, None) == [(700.0, 720.0), (700.0, 710.0)]
