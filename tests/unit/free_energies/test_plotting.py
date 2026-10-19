"""
Unit tests for `plot_forward_backward_composition_scaling` in phase_diagram_workflows.free_energies.plotting.

The job folder is a small synthetic one: input_file.yaml, report.yaml and the forward/backward sweeps, so no calphy
run is needed.
"""

import numpy as np
import pytest
import yaml

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_diagram_workflows.free_energies.plotting import plot_forward_backward_composition_scaling

REPORT = {
    "input": {"element": "Al Mg", "concentration": "0.95 0.05"},
    "results": {"free_energy": 0.1145, "work": 0.1141, "mass_correction": -0.0003, "dissipation": 2.54e-05, "entropy_contribution": -1.7e-05, "error": 0.0},
}
N_STEPS = 400


def write_job(folder, mode="composition_scaling", with_sweeps=True, with_report=True):
    folder.mkdir(parents=True, exist_ok=True)
    calculation = {"mode": mode, "reference_phase": "solid", "pressure": 0.0, "temperature": 500, "reference_composition": 0.0}
    (folder / "input_file.yaml").write_text(yaml.safe_dump({"calculations": [calculation]}))
    if with_report:
        (folder / "report.yaml").write_text(yaml.safe_dump(REPORT))
    if with_sweeps:
        lam = np.linspace(1.0, 0.0, N_STEPS)
        header = "dU_1[eV/atom] dU_2[eV/atom] lambda"
        # U(target) - U(reference) = 0.2 - 0.1 * lambda going forward; the backward sweep lies 1 meV/atom above it
        np.savetxt(folder / "forward_1.dat", np.column_stack([np.full(N_STEPS, -3.5), -3.5 + 0.2 - 0.1 * lam, lam]), header=header)
        np.savetxt(folder / "backward_1.dat", np.column_stack([np.full(N_STEPS, -3.5), -3.5 + 0.201 - 0.1 * lam[::-1], lam[::-1]]), header=header)
    return str(folder)


class TestPlotForwardBackwardCompositionScaling:
    def test_draws_forward_and_backward_with_the_ts_look(self, tmp_path):
        fig, ax = plot_forward_backward_composition_scaling(write_job(tmp_path / "job"), smoothing=0)

        assert [line.get_label() for line in ax.get_lines()] == ["forward", "backward"]
        assert ax.get_xlabel() == "lambda"
        assert ax.get_ylabel() == "Energy difference [eV/atom]"
        assert ax.get_title() == "500 K, Al_0.950_Mg_0.050, dissipation=2.540e-05"
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_energy_difference_is_target_minus_reference(self, tmp_path):
        fig, ax = plot_forward_backward_composition_scaling(write_job(tmp_path / "job"), smoothing=0)

        forward = ax.get_lines()[0]
        assert forward.get_xdata()[0] == pytest.approx(1.0)       # the forward sweep starts at the reference
        assert forward.get_ydata()[0] == pytest.approx(0.1)       # 0.2 - 0.1 * 1
        assert forward.get_ydata()[-1] == pytest.approx(0.2)      # at the target
        plt.close(fig)

    def test_smoothing_shortens_the_curves_by_the_window(self, tmp_path):
        fig, ax = plot_forward_backward_composition_scaling(write_job(tmp_path / "job"), smoothing=100)

        assert len(ax.get_lines()[0].get_xdata()) == N_STEPS - 99
        plt.close(fig)

    def test_smoothing_longer_than_the_data_is_clamped(self, tmp_path):
        fig, ax = plot_forward_backward_composition_scaling(write_job(tmp_path / "job"), smoothing=10 * N_STEPS)

        forward = ax.get_lines()[0]
        assert len(forward.get_xdata()) == len(forward.get_ydata()) >= 1     # np.convolve would have swapped its arguments
        plt.close(fig)

    def test_job_that_is_still_running_with_a_single_row_is_not_an_error(self, tmp_path):
        folder = write_job(tmp_path / "job", with_sweeps=False, with_report=False)
        (tmp_path / "job" / "forward_1.dat").write_text("# dU_1 dU_2 lambda\n-3.5 -3.3 1.0\n")
        (tmp_path / "job" / "backward_1.dat").write_text("# dU_1 dU_2 lambda\n-3.5 -3.3 0.0\n")

        fig, ax = plot_forward_backward_composition_scaling(folder)

        assert ax.get_title() == "500 K (running)"
        plt.close(fig)

    def test_draws_into_the_axes_it_is_given(self, tmp_path):
        fig, axes = plt.subplots(1, 2)
        write_job(tmp_path / "a")
        write_job(tmp_path / "b")

        returned_fig, returned_ax = plot_forward_backward_composition_scaling(str(tmp_path / "b"), ax=axes[1])

        assert returned_ax is axes[1] and returned_fig is fig
        assert len(axes[1].get_lines()) == 2 and len(axes[0].get_lines()) == 0
        plt.close(fig)

    def test_job_without_sweeps_is_running(self, tmp_path):
        fig, ax = plot_forward_backward_composition_scaling(write_job(tmp_path / "job", with_sweeps=False, with_report=False))

        assert ax.get_title() == "500 K (running)"
        assert ax.get_lines() == []
        plt.close(fig)

    def test_folder_without_output_says_so(self, tmp_path):
        fig, ax = plot_forward_backward_composition_scaling(str(tmp_path / "nothing"))

        assert ax.get_title() == "(no output yet)"
        plt.close(fig)

    def test_other_modes_raise(self, tmp_path):
        with pytest.raises(ValueError, match="not composition_scaling"):
            plot_forward_backward_composition_scaling(write_job(tmp_path / "job", mode="fe"))
