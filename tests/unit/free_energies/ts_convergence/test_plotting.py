"""
Unit tests for `plot_forward_backward` in phase_diagram_workflows.free_energies.ts_convergence.plotting.

`gather_calphy_results_detailed` is stubbed with synthetic forward/backward sweeps, so no calphy output
is needed; the bracket log is the real one, written by `_record_bracket` into a temporary directory.
"""

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import phase_diagram_workflows.free_energies.ts_convergence.plotting as plotting_module
import phase_diagram_workflows.free_energies.ts_convergence.single as single_module
from phase_diagram_workflows.free_energies.ts_convergence.plotting import plot_forward_backward


def sweep(t_low, t_high, gap):
    """A sweep whose backward branch ends `gap` eV/atom away from where the forward one starts (= the criterion)."""
    lam = np.linspace(1.0, t_low / t_high, 50)
    forward = np.linspace(0.0, 0.1, 50)
    backward = np.linspace(0.1, 0.0, 50) + gap
    return pd.DataFrame([{
        "forward_energy_diff": [forward],
        "backward_energy_diff": [backward],
        "forward_lambda": [lam],
        "backward_lambda": [lam[::-1]],
    }])


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = str(tmp_path / "structure")
    for bracket, criterion in [((300.0, 1000.0), 0.03), ((300.0, 900.0), 0.001), ((300.0, 800.0), None)]:
        single_module._record_bracket(root, bracket, criterion)

    def gather(working_directory):
        if "800.00" in working_directory:
            raise FileNotFoundError(working_directory)  # the last bracket has not produced output yet
        t_high = 1000.0 if "1000.00" in working_directory else 900.0
        return sweep(300.0, t_high, 0.03 if t_high == 1000.0 else 0.0)

    monkeypatch.setattr(plotting_module, "gather_calphy_results_detailed", gather)
    return root


def test_one_panel_per_bracket_widest_first(root):
    fig, axes = plot_forward_backward(root)
    assert axes.shape == (1, 3)
    titles = [ax.get_title() for ax in axes[0]]
    assert titles[0].startswith("300-1000 K")
    assert titles[1].startswith("300-900 K")
    assert titles[2] == "300-800 K (no output yet)"


def test_title_carries_the_criterion(root):
    _, axes = plot_forward_backward(root)
    assert "criterion=0.0300" in axes[0][0].get_title()
    assert "criterion=0.0000" in axes[0][1].get_title()


def test_both_sweeps_are_drawn_against_temperature(root):
    _, axes = plot_forward_backward(root)
    lines = axes[0][0].get_lines()
    assert [line.get_label() for line in lines] == ["forward", "backward"]
    forward_t = lines[0].get_xdata()
    assert forward_t[0] == pytest.approx(300.0) and forward_t[-1] == pytest.approx(1000.0)


def test_chosen_brackets_only(root):
    _, axes = plot_forward_backward(root, brackets=[(300.0, 900.0)])
    assert axes.shape == (1, 1)


def test_no_brackets_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No brackets"):
        plot_forward_backward(str(tmp_path / "empty"))
