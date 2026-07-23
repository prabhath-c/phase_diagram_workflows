"""
Unit tests for phase_diagram_workflows.phase_diagram.convex_hull.

atomistics (real LAMMPS), executorlib (via run_nested_batch), and Materials
Project (via structures.materials_project) are all mocked at their call
boundary -- these tests never run real LAMMPS, submit a real job, or touch
the network/a real API key. The hull geometry (compute_mixing_energy,
compute_convex_hull) and the plotting functions are exercised for real:
they're cheap, deterministic, and plotly/matplotlib/seaborn are lightweight
enough to just run.
"""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless -- no display needed/available in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from phase_diagram_workflows.phase_diagram.convex_hull import (
    _default_axis_labels,
    _default_yrange,
    _structure_color,
    analyze_convex_hull,
    compute_convex_hull,
    compute_energies_with_nested_executor,
    compute_mixing_energy,
    fetch_structures_and_energies,
    optimize_and_compute_energy_per_atom,
    optimize_structure,
    compute_energy_per_atom,
    plot_convex_hull,
    plot_convex_hull_matplotlib,
)

# All positive except the endpoints -- the lower hull is just the (0,0)-(1,0)
# edge, so exactly Al and Mg end up on it.
HULL_DF = pd.DataFrame({
    "x": [0.0, 0.25, 0.5, 0.5, 0.75, 1.0],
    "mixing_energy": [0.0, 0.105, 0.2, 0.35, 0.375, 0.0],
    "formula_pretty": ["Al", "Al3Mg", "AlMg_a", "AlMg_b", "AlMg3", "Mg"],
})

# Has a clearly negative interior point, which is always a hull vertex --
# used for the yrange-clipping tests.
HULL_DF_WITH_DIP = pd.DataFrame({
    "x": [0.0, 0.2, 0.333, 0.6, 0.8, 1.0],
    "mixing_energy": [0.0, 0.05, -0.04, 0.02, 0.05, 0.0],
    "formula_pretty": ["Al", "S1", "MgAl2", "S2", "S3", "Mg"],
})


class TestOptimizeAndComputeEnergyPerAtom:
    @patch("atomistics.calculators.lammps.libcalculator.optimize_positions_and_volume_with_lammpslib")
    def test_optimize_structure_forwards_args(self, mock_optimize):
        atoms = bulk("Al")
        potential_df = pd.DataFrame({"Species": [["Al"]]})
        mock_optimize.return_value = "relaxed"

        result = optimize_structure(atoms, potential_df, min_style="fire")

        assert result == "relaxed"
        kwargs = mock_optimize.call_args.kwargs
        assert kwargs["structure"] is atoms
        assert kwargs["potential_dataframe"] is potential_df
        assert kwargs["min_style"] == "fire"

    @patch("atomistics.calculators.lammps.libcalculator.calc_static_with_lammpslib")
    def test_compute_energy_per_atom_divides_by_natoms(self, mock_calc_static):
        atoms = bulk("Al", cubic=True)  # 4 atoms
        potential_df = pd.DataFrame({"Species": [["Al"]]})
        mock_calc_static.return_value = {"energy": -12.0}

        epa = compute_energy_per_atom(atoms, potential_df)

        assert epa == pytest.approx(-3.0)
        assert mock_calc_static.call_args.kwargs["output_keys"] == ("energy",)

    @patch("phase_diagram_workflows.phase_diagram.convex_hull.compute_energy_per_atom")
    @patch("phase_diagram_workflows.phase_diagram.convex_hull.optimize_structure")
    def test_chains_optimize_then_static(self, mock_optimize, mock_compute_epa):
        atoms = bulk("Al")
        potential_df = pd.DataFrame()
        mock_optimize.return_value = "relaxed_atoms"
        mock_compute_epa.return_value = -3.5

        relaxed, epa = optimize_and_compute_energy_per_atom(atoms, potential_df)

        assert relaxed == "relaxed_atoms"
        assert epa == -3.5
        mock_optimize.assert_called_once_with(atoms, potential_df)
        mock_compute_epa.assert_called_once_with("relaxed_atoms", potential_df)


class TestComputeEnergiesWithNestedExecutor:
    @patch("phase_diagram_workflows.phase_diagram.convex_hull.run_nested_batch")
    def test_attaches_results_to_dataframe(self, mock_run_nested_batch):
        atoms_al, atoms_mg = bulk("Al"), bulk("Mg")
        df = pd.DataFrame({"structure_ase": [atoms_al, atoms_mg], "x": [0.0, 1.0]})
        potential_df = pd.DataFrame({"Species": [["Al", "Mg"]]})
        mock_run_nested_batch.return_value = [("relaxed_al", -3.5), ("relaxed_mg", -1.5)]

        result = compute_energies_with_nested_executor(
            df=df, atoms_col="structure_ase", potential_df=potential_df, outer_executor_cls=object,
        )

        assert list(result["atoms_relaxed"]) == ["relaxed_al", "relaxed_mg"]
        assert list(result["energy_per_atom_calc"]) == [-3.5, -1.5]
        kwargs = mock_run_nested_batch.call_args.kwargs
        assert len(kwargs["items"]) == 2
        assert kwargs["items"][0] is atoms_al
        assert kwargs["items"][1] is atoms_mg
        assert kwargs["outer_executor_cls"] is object
        assert kwargs["task_args"][0] is potential_df

    @patch("phase_diagram_workflows.phase_diagram.convex_hull.run_nested_batch")
    def test_custom_result_column_names(self, mock_run_nested_batch):
        df = pd.DataFrame({"structure_ase": [bulk("Al")]})
        mock_run_nested_batch.return_value = [("relaxed", -3.5)]

        result = compute_energies_with_nested_executor(
            df=df, atoms_col="structure_ase", potential_df=pd.DataFrame(), outer_executor_cls=object,
            result_atoms_col="my_atoms", result_value_col="my_energy",
        )

        assert "my_atoms" in result.columns
        assert "my_energy" in result.columns

    @patch("phase_diagram_workflows.phase_diagram.convex_hull.run_nested_batch")
    def test_wait_false_returns_raw_results(self, mock_run_nested_batch):
        df = pd.DataFrame({"structure_ase": [bulk("Al")]})
        mock_run_nested_batch.return_value = "a-future"

        result = compute_energies_with_nested_executor(
            df=df, atoms_col="structure_ase", potential_df=pd.DataFrame(), outer_executor_cls=object,
            wait=False,
        )

        assert result == "a-future"

    @patch("phase_diagram_workflows.phase_diagram.convex_hull.run_nested_batch")
    def test_does_not_mutate_input_dataframe(self, mock_run_nested_batch):
        df = pd.DataFrame({"structure_ase": [bulk("Al")]})
        mock_run_nested_batch.return_value = [("relaxed", -3.5)]

        compute_energies_with_nested_executor(
            df=df, atoms_col="structure_ase", potential_df=pd.DataFrame(), outer_executor_cls=object,
        )

        assert "energy_per_atom_calc" not in df.columns


class TestComputeMixingEnergy:
    def test_basic_formula(self):
        df = pd.DataFrame({"x": [0.0, 0.5, 1.0], "energy_per_atom": [-3.0, -2.0, -1.0]})
        out = compute_mixing_energy(df)
        assert out["mixing_energy"].tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_uses_minimum_energy_at_each_endpoint(self):
        df = pd.DataFrame({
            "x": [0.0, 0.0, 1.0, 1.0],
            "energy_per_atom": [-3.0, -3.5, -1.0, -0.5],
        })
        out = compute_mixing_energy(df)
        assert out.loc[out["x"] == 0.0, "mixing_energy"].tolist() == pytest.approx([0.5, 0.0])
        assert out.loc[out["x"] == 1.0, "mixing_energy"].tolist() == pytest.approx([0.0, 0.5])

    def test_raises_without_x_equal_zero(self):
        df = pd.DataFrame({"x": [0.5, 1.0], "energy_per_atom": [-2.0, -1.0]})
        with pytest.raises(ValueError, match="x == 0"):
            compute_mixing_energy(df)

    def test_raises_without_x_equal_one(self):
        df = pd.DataFrame({"x": [0.0, 0.5], "energy_per_atom": [-3.0, -2.0]})
        with pytest.raises(ValueError, match="x == 1"):
            compute_mixing_energy(df)

    def test_custom_column_names(self):
        df = pd.DataFrame({"c": [0.0, 1.0], "epa": [-3.0, -1.0]})
        out = compute_mixing_energy(df, x_col="c", energy_col="epa", mixing_energy_col="e_mix")
        assert "e_mix" in out.columns


class TestComputeConvexHull:
    def test_v_shape_returns_lower_hull_only(self):
        df = pd.DataFrame({
            "x": [0.0, 0.25, 0.5, 0.5, 0.75, 1.0],
            "y": [0.0, -0.05, -0.1, 0.2, -0.03, 0.0],
        })
        hull = compute_convex_hull(df, x_col="x", energy_col="y")
        assert hull["x"].tolist() == pytest.approx([0.0, 0.5, 1.0])
        assert hull["y"].tolist() == pytest.approx([0.0, -0.1, 0.0])

    def test_sorted_by_x(self):
        df = pd.DataFrame({"x": [1.0, 0.0, 0.5], "y": [0.0, 0.0, -0.1]})
        hull = compute_convex_hull(df, x_col="x", energy_col="y")
        assert hull["x"].is_monotonic_increasing

    def test_raises_with_fewer_than_three_points(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]})
        with pytest.raises(ValueError, match="at least 3"):
            compute_convex_hull(df, x_col="x", energy_col="y")

    def test_drops_nan_rows_before_counting(self):
        df = pd.DataFrame({"x": [0.0, 0.5, 1.0, 0.7], "y": [0.0, -0.1, 0.0, np.nan]})
        hull = compute_convex_hull(df, x_col="x", energy_col="y")
        assert len(hull) == 3


class TestDefaultYrange:
    def test_none_when_cap_is_none(self):
        df_hull = pd.DataFrame({"e": [0.0, -0.1, 0.0]})
        assert _default_yrange(df_hull, "e", None) is None

    def test_pads_below_and_caps_above_at_hull_max_plus_margin(self):
        df_hull = pd.DataFrame({"e": [0.0, -0.1, 0.0]})
        lo, hi = _default_yrange(df_hull, "e", max_energy_above_hull=0.2)
        assert lo == pytest.approx(-0.11)  # hull_lo - 10% of hull span
        assert hi == pytest.approx(0.2)  # hull_hi (0.0) + margin


class TestDefaultAxisLabels:
    def test_matplotlib_latex_with_element(self):
        xlabel, ylabel = _default_axis_labels("x", "mixing_energy", element="Mg", latex=True)
        assert xlabel == r"$x_\mathrm{Mg}$"
        assert ylabel == "Mixing energy [eV/atom]"

    def test_plotly_html_with_element(self):
        xlabel, _ = _default_axis_labels("x", "mixing_energy", element="Mg", latex=False)
        assert xlabel == "x<sub>Mg</sub>"

    def test_generic_without_element(self):
        xlabel_latex, _ = _default_axis_labels("x", "energy_per_atom_calc", latex=True)
        xlabel_html, _ = _default_axis_labels("x", "energy_per_atom_calc", latex=False)
        assert xlabel_latex == "Composition, $x$"
        assert xlabel_html == "Composition, x"

    def test_ylabel_falls_back_for_non_mixing_energy_columns(self):
        _, ylabel = _default_axis_labels("x", "energy_per_atom_calc")
        assert ylabel == "Energy per atom [eV/atom]"


class TestStructureColor:
    def test_matches_seaborn_muted_slot_zero(self):
        import seaborn as sns

        assert _structure_color() == sns.color_palette("muted").as_hex()[0]


class TestPlotConvexHull:
    def test_returns_figure_and_hull(self):
        fig, hull = plot_convex_hull(HULL_DF, x_col="x", energy_col="mixing_energy")
        assert len(fig.data) == 2  # structures trace + hull trace
        assert sorted(hull["formula_pretty"]) == ["Al", "Mg"]

    def test_color_col_creates_one_trace_per_group_plus_hull(self):
        df = HULL_DF.assign(space_group=["A", "A", "B", "B", "A", "B"])
        fig, _ = plot_convex_hull(df, x_col="x", energy_col="mixing_energy", color_col="space_group")
        assert len(fig.data) == 3  # 2 groups + hull

    def test_warns_when_yrange_clips_a_hull_point(self):
        with pytest.warns(UserWarning, match="clips"):
            plot_convex_hull(HULL_DF_WITH_DIP, x_col="x", energy_col="mixing_energy", yrange=(-0.02, 0.1))

    def test_no_warning_when_yrange_contains_the_hull(self):
        with warnings_as_errors():
            plot_convex_hull(HULL_DF_WITH_DIP, x_col="x", energy_col="mixing_energy", yrange=(-0.1, 0.1))

    def test_element_produces_html_subscript_label(self):
        fig, _ = plot_convex_hull(HULL_DF, x_col="x", energy_col="mixing_energy", element="Mg")
        assert fig.layout.xaxis.title.text == "x<sub>Mg</sub>"

    def test_explicit_labels_override_defaults(self):
        fig, _ = plot_convex_hull(HULL_DF, x_col="x", energy_col="mixing_energy", xlabel="custom x", ylabel="custom y")
        assert fig.layout.xaxis.title.text == "custom x"
        assert fig.layout.yaxis.title.text == "custom y"


class TestPlotConvexHullMatplotlib:
    def teardown_method(self):
        plt.close("all")

    def test_returns_fig_ax_hull(self):
        fig, ax, hull = plot_convex_hull_matplotlib(HULL_DF, x_col="x", energy_col="mixing_energy")
        assert ax.get_xlabel() == "Composition, $x$"
        assert sorted(hull["formula_pretty"]) == ["Al", "Mg"]

    def test_element_produces_latex_label(self):
        _, ax, _ = plot_convex_hull_matplotlib(HULL_DF, x_col="x", energy_col="mixing_energy", element="Mg")
        assert ax.get_xlabel() == r"$x_\mathrm{Mg}$"

    def test_warns_when_yrange_clips_a_hull_point(self):
        with pytest.warns(UserWarning, match="clips"):
            plot_convex_hull_matplotlib(HULL_DF_WITH_DIP, x_col="x", energy_col="mixing_energy", yrange=(-0.02, 0.1))

    def test_reuses_provided_axes(self):
        fig, ax = plt.subplots()
        fig2, ax2, _ = plot_convex_hull_matplotlib(HULL_DF, x_col="x", energy_col="mixing_energy", ax=ax)
        assert ax2 is ax
        assert fig2 is fig

    def test_no_labels_when_label_col_is_none(self):
        fig, ax, _ = plot_convex_hull_matplotlib(HULL_DF, x_col="x", energy_col="mixing_energy", label_col=None)
        assert len(ax.texts) == 0


class TestFetchStructuresAndEnergies:
    @patch("phase_diagram_workflows.phase_diagram.convex_hull.compute_energies_with_nested_executor")
    @patch("phase_diagram_workflows.structures.materials_project.append_structure")
    @patch("phase_diagram_workflows.structures.materials_project.build_structures_dataframe")
    @patch("phase_diagram_workflows.structures.materials_project.get_materials_project_df")
    def test_derives_chemsys_from_elements(self, mock_get_mp, mock_build, mock_append, mock_compute):
        mock_get_mp.return_value = pd.DataFrame({"material_id": ["mp-1"]})
        mock_build.return_value = pd.DataFrame({"structure_ase": [bulk("Al")]})
        mock_compute.return_value = "final_df"

        result = fetch_structures_and_energies(
            elements=["Al", "Mg"], api_key="dummy", potential_df=pd.DataFrame(), outer_executor_cls=object,
        )

        mock_get_mp.assert_called_once_with("Al-Mg", "dummy", include_pure=True, fields=None)
        mock_append.assert_not_called()
        assert result == "final_df"

    @patch("phase_diagram_workflows.phase_diagram.convex_hull.compute_energies_with_nested_executor")
    @patch("phase_diagram_workflows.structures.materials_project.append_structure")
    @patch("phase_diagram_workflows.structures.materials_project.build_structures_dataframe")
    @patch("phase_diagram_workflows.structures.materials_project.get_materials_project_df")
    def test_folds_in_extra_structures(self, mock_get_mp, mock_build, mock_append, mock_compute):
        base_df = pd.DataFrame({"structure_ase": [bulk("Al")]})
        appended_df = pd.DataFrame({"structure_ase": [bulk("Al"), bulk("Mg")]})
        mock_get_mp.return_value = pd.DataFrame()
        mock_build.return_value = base_df
        mock_append.return_value = appended_df
        mock_compute.return_value = appended_df
        extra_atoms = bulk("Mg")

        fetch_structures_and_energies(
            elements=["Al", "Mg"], api_key="dummy", potential_df=pd.DataFrame(), outer_executor_cls=object,
            extra_structures=[{"atoms": extra_atoms, "material_id": "external", "formula_pretty": "Mg"}],
        )

        mock_append.assert_called_once_with(
            base_df, elements=["Al", "Mg"], atoms=extra_atoms, material_id="external", formula_pretty="Mg",
        )

    @patch("phase_diagram_workflows.phase_diagram.convex_hull.compute_energies_with_nested_executor")
    @patch("phase_diagram_workflows.structures.materials_project.build_structures_dataframe")
    @patch("phase_diagram_workflows.structures.materials_project.get_materials_project_df")
    def test_forwards_potential_and_executor_kwargs(self, mock_get_mp, mock_build, mock_compute):
        mock_get_mp.return_value = pd.DataFrame()
        mock_build.return_value = pd.DataFrame({"structure_ase": [bulk("Al")]})
        mock_compute.return_value = "final_df"
        potential_df = pd.DataFrame({"Species": [["Al"]]})

        fetch_structures_and_energies(
            elements=["Al", "Mg"], api_key="dummy", potential_df=potential_df,
            outer_executor_cls="SomeExecutor", inner_max_workers=8,
        )

        kwargs = mock_compute.call_args.kwargs
        assert kwargs["potential_df"] is potential_df
        assert kwargs["outer_executor_cls"] == "SomeExecutor"
        assert kwargs["inner_max_workers"] == 8
        assert kwargs["atoms_col"] == "structure_ase"


class TestAnalyzeConvexHull:
    def teardown_method(self):
        plt.close("all")

    @staticmethod
    def _energy_df():
        return pd.DataFrame({
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "energy_per_atom_calc": [-3.0, -2.9, -2.7, -1.8, -1.5],
            "formula_pretty": ["Al", "S1", "S2", "S3", "Mg"],
        })

    def test_plotly_backend(self):
        fig, hull = analyze_convex_hull(self._energy_df(), backend="plotly")
        assert len(fig.data) == 2
        assert not hull.empty

    def test_matplotlib_backend(self):
        fig, ax, hull = analyze_convex_hull(self._energy_df(), backend="matplotlib")
        assert not hull.empty

    def test_does_not_mutate_input(self):
        df = self._energy_df()
        analyze_convex_hull(df, backend="plotly")
        assert "mixing_energy" not in df.columns

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            analyze_convex_hull(self._energy_df(), backend="bokeh")


class warnings_as_errors:
    """Context manager: fail the test if any warning is raised inside it."""

    def __enter__(self):
        import warnings

        self._catcher = warnings.catch_warnings()
        self._catcher.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._catcher.__exit__(*exc)
