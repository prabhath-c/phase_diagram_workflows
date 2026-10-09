"""Unit tests for phase_diagram.defect_diagram: the table of lines, the summary tables, and that every figure is drawn."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase.build import bulk

from phase_diagram_workflows.phase_diagram.defect_diagram import (
    BOLTZMANN_EV,
    concentration_table,
    defect_color,
    defect_lines,
    mole_fraction,
    plot_concentration_vs_composition,
    plot_concentration_vs_composition_pdm,
    plot_concentration_vs_dmu,
    plot_defect_phase_diagram,
    plot_T_vs_dmu,
    stable_defect_table,
)


def _record(label, kind, species, dn_al, dn_mg, intercept, site=None):
    return {
        "defect_label": label, "defect_latex": f"${species}_{{{kind[0]}}}^{{{site or 'x'}}}$", "defect_type": kind, "species": species,
        "sublattice_int": site, "delta_n_Al": dn_al, "delta_n_Mg": dn_mg, "delta_n_total": dn_al + dn_mg, "intercept": intercept,
    }


RECORDS = [
    _record("Al_vac_4a", "vacancy", "Al", -1, 0, 0.65),
    _record("Mg_on_Al_4a", "substitutional", "Mg", -1, 1, 0.38),
    _record("Al_i_int_4b", "interstitial", "Al", 1, 0, 2.5, "int_4b"),
    _record("Al_i_int_8c", "interstitial", "Al", 1, 0, 2.5, "int_8c"),     # same line as 4b
    _record("Mg_i_int_8c", "interstitial", "Mg", 0, 1, 3.5, "int_8c"),
    _record("Mg_vac_2d", "vacancy", "Mg", 0, -1, 0.9),
]


class TestDefectLines:
    def test_slope_is_minus_delta_n_of_the_swept_element(self):
        mg = defect_lines(RECORDS, "Mg").set_index("defect_label")
        al = defect_lines(RECORDS, "Al").set_index("defect_label")
        assert mg.loc["Mg_on_Al_4a", "slope"] == -1 and al.loc["Mg_on_Al_4a", "slope"] == 1
        assert mg.loc["Al_vac_4a", "slope"] == 0 and al.loc["Al_vac_4a", "slope"] == 1
        assert mg.loc["Mg_vac_2d", "slope"] == 1 and al.loc["Mg_vac_2d", "slope"] == 0

    def test_elements_come_from_the_records_and_unknown_ones_are_refused(self):
        table = defect_lines(RECORDS, "Mg")
        assert table.attrs["elements"] == ["Al", "Mg"] and table.attrs["swept_element"] == "Mg"
        with pytest.raises(ValueError, match="delta_n_Ti"):
            defect_lines(RECORDS, "Ti")
        with pytest.raises(ValueError, match="no records"):
            defect_lines([], "Mg")

    def test_unique_merges_defects_with_the_same_line_keeping_the_first(self):
        assert len(defect_lines(RECORDS, "Mg")) == 6
        unique = defect_lines(RECORDS, "Mg", unique=True)
        assert len(unique) == 5 and "Al_i_int_8c" not in set(unique["defect_label"])

    def test_colours_by_type_and_species_with_a_fallback_for_other_species(self):
        assert defect_color("vacancy", "Al") == "#c0392b" and defect_color("interstitial", "Mg") == "#9b59b6"
        assert defect_color("vacancy", "Ti") == "#c0392b"          # an element without its own colour
        assert defect_color("interstitial", "Al", "int_18f_3", shade_long_interstitials=True) != defect_color("interstitial", "Al")
        assert defect_color("interstitial", "Al", "int_4b", shade_long_interstitials=True) == (
            pytest.approx(0x27 / 255), pytest.approx(0xAE / 255), pytest.approx(0x60 / 255))


class TestTables:
    def test_stable_defect_is_the_lowest_line(self):
        table = defect_lines(RECORDS, "Mg")
        result = stable_defect_table(table, [0.0, -1.0])
        assert list(result["stable_defect"]) == ["Mg_on_Al_4a", "Mg_vac_2d"]      # 0.38 < 0.65 at 0;  at -1: 0.9 - 1.0 = -0.1 is lowest

    def test_concentration_is_boltzmann(self):
        table = defect_lines(RECORDS, "Mg")
        row = concentration_table(table, [0.0], [300]).iloc[0]
        assert row["dominant defect"] == "Mg_on_Al_4a"
        assert row["log10(c)"] == pytest.approx(-0.38 / (BOLTZMANN_EV * 300 * np.log(10)), abs=0.01)

    def test_boltzmann_constant(self):
        assert BOLTZMANN_EV == pytest.approx(8.617333262e-5, rel=1e-9)

    def test_mole_fraction(self):
        atoms = bulk("Al", cubic=True)
        atoms[0].symbol = "Mg"
        assert mole_fraction(atoms, "Mg") == 0.25 and mole_fraction(atoms, "Al") == 0.75


class TestFigures:
    unique = defect_lines(RECORDS, "Mg", unique=True)

    def test_phase_diagram_merges_labels_of_defects_with_the_same_energy_at_the_edge(self):
        table = defect_lines(RECORDS, "Mg", shade_long_interstitials=True)
        fig = plot_defect_phase_diagram(table, (-2.0, 0.0), "FCC")
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("Al_i^{4b,8c}" in t for t in texts)
        plt.close("all")

    def test_concentration_vs_dmu_gives_a_grid_and_an_envelope_figure_and_saves_them(self, tmp_path):
        grid, envelope = plot_concentration_vs_dmu(
            self.unique, (-1.5, 0.0), "FCC", save_grid=str(tmp_path / "g.png"), save_envelope=str(tmp_path / "e.png"), dpi=40)
        assert len(grid.axes) == 9 + 0 and (tmp_path / "g.png").is_file() and (tmp_path / "e.png").is_file()
        plt.close("all")

    def test_T_vs_dmu_gives_a_grid_and_an_iso_figure(self):
        grid, iso = plot_T_vs_dmu(self.unique, (-1.5, 0.0), "FCC")
        assert len(grid.axes) == 9 and len(iso.axes) == 1
        plt.close("all")

    def test_composition_figures(self):
        fig = plot_concentration_vs_composition(self.unique, (-1.5, 0.0), "FCC", 0.0)
        assert len(fig.axes) == 9
        log, linear = plot_concentration_vs_composition_pdm(self.unique, (-1.5, 0.5), "FCC", 0.0, dpi=40)
        assert len(log.axes) == 9 and len(linear.axes) == 9
        plt.close("all")

    def test_the_composition_axis_needs_a_binary_system(self):
        ternary = [{**r, "delta_n_Ti": 0} for r in RECORDS]
        with pytest.raises(ValueError, match="binary"):
            plot_concentration_vs_composition(defect_lines(ternary, "Mg", unique=True), (-1.5, 0.0), "FCC", 0.0)
        plt.close("all")
