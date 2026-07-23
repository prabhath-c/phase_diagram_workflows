"""
Unit tests for phase_diagram_workflows.structures.materials_project.

MPRester is mocked throughout -- these tests never touch the network and
never need a real Materials Project API key; mp-api only needs to be
importable so there is something to patch. build_structures_dataframe and
append_structure are exercised against real (small, in-memory) pymatgen/ASE
structures rather than mocks, since structure conversion is the actual thing
being tested there.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk

from phase_diagram_workflows.structures.materials_project import (
    _get_element_fractions,
    append_structure,
    build_structures_dataframe,
    get_materials_project_df,
)


def _make_structure_dict(species, coords, lattice=((4, 0, 0), (0, 4, 0), (0, 0, 4))):
    from pymatgen.core import Lattice, Structure

    return Structure(Lattice(lattice), species, coords).as_dict()


def _mock_doc(**fields):
    doc = MagicMock()
    doc.model_dump.return_value = fields
    return doc


class TestGetElementFractions:
    def test_basic_fractions(self):
        atoms = Atoms("Al2Mg2")
        assert _get_element_fractions(atoms, ["Al", "Mg"]) == {"Al": 0.5, "Mg": 0.5}

    def test_denominator_is_only_the_requested_elements(self):
        # Total is count(Al) + count(Mg), ignoring the 4 Cu atoms entirely --
        # fractions describe the binary system of interest, not the whole cell.
        atoms = Atoms("Al2Mg2Cu4")
        assert _get_element_fractions(atoms, ["Al", "Mg"]) == {"Al": 0.5, "Mg": 0.5}

    def test_raises_when_no_matching_elements(self):
        atoms = bulk("Cu")
        with pytest.raises(ValueError, match="No atoms"):
            _get_element_fractions(atoms, ["Al", "Mg"])


class TestGetMaterialsProjectDf:
    @patch("mp_api.client.MPRester")
    def test_include_pure_queries_chemsys_and_both_pure_elements(self, mock_mprester_cls):
        mock_mprester = MagicMock()
        mock_mprester_cls.return_value.__enter__.return_value = mock_mprester
        # systems = sorted({"Al-Mg", "Al", "Mg"}) == ["Al", "Al-Mg", "Mg"]
        mock_mprester.materials.summary.search.side_effect = [
            [_mock_doc(material_id="mp-2", formula_pretty="Al")],
            [_mock_doc(material_id="mp-1", formula_pretty="AlMg")],
            [_mock_doc(material_id="mp-3", formula_pretty="Mg")],
        ]

        df = get_materials_project_df("Al-Mg", "dummy-key", include_pure=True, fields=None)

        queried = [call.kwargs["chemsys"] for call in mock_mprester.materials.summary.search.call_args_list]
        assert queried == ["Al", "Al-Mg", "Mg"]
        assert sorted(df["material_id"]) == ["mp-1", "mp-2", "mp-3"]

    @patch("mp_api.client.MPRester")
    def test_include_pure_false_only_queries_the_chemsys(self, mock_mprester_cls):
        mock_mprester = MagicMock()
        mock_mprester_cls.return_value.__enter__.return_value = mock_mprester
        mock_mprester.materials.summary.search.return_value = [_mock_doc(material_id="mp-1")]

        get_materials_project_df("Al-Mg", "dummy-key", include_pure=False)

        assert mock_mprester.materials.summary.search.call_count == 1
        assert mock_mprester.materials.summary.search.call_args.kwargs["chemsys"] == "Al-Mg"

    @patch("mp_api.client.MPRester")
    def test_deduplicates_by_material_id(self, mock_mprester_cls):
        mock_mprester = MagicMock()
        mock_mprester_cls.return_value.__enter__.return_value = mock_mprester
        # The same entry can legitimately show up from both the chemsys
        # search and one of the pure-element searches.
        mock_mprester.materials.summary.search.side_effect = [
            [_mock_doc(material_id="mp-1")],
            [_mock_doc(material_id="mp-1")],
            [],
        ]

        df = get_materials_project_df("Al-Mg", "dummy-key", include_pure=True)

        assert len(df) == 1

    @patch("mp_api.client.MPRester")
    def test_forwards_fields_and_api_key(self, mock_mprester_cls):
        mock_mprester = MagicMock()
        mock_mprester_cls.return_value.__enter__.return_value = mock_mprester
        mock_mprester.materials.summary.search.return_value = []

        get_materials_project_df("Al", "my-key", include_pure=False, fields=["material_id", "formula_pretty"])

        mock_mprester_cls.assert_called_once_with("my-key")
        assert mock_mprester.materials.summary.search.call_args.kwargs["fields"] == ["material_id", "formula_pretty"]


class TestBuildStructuresDataframe:
    def test_adds_expected_columns_and_computes_x(self):
        structure_dict = _make_structure_dict(["Al", "Mg"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        mp_df = pd.DataFrame([{
            "material_id": "mp-1",
            "structure": structure_dict,
            "symmetry": {"crystal_system": "Cubic", "symbol": "Pm-3m"},
        }])

        df = build_structures_dataframe(mp_df, elements=["Al", "Mg"])

        assert df.loc[0, "material_id_underscore"] == "mp_1"
        assert df.loc[0, "crystal_system"] == "Cubic"
        assert df.loc[0, "space_group"] == "Pm-3m"
        assert df.loc[0, "x"] == pytest.approx(0.5)
        assert df.loc[0, "fractions"] == {"Al": 0.5, "Mg": 0.5}
        assert len(df.loc[0, "structure_ase"]) == 2

    def test_x_is_the_fraction_of_the_second_element(self):
        structure_dict = _make_structure_dict(["Al", "Al", "Al", "Mg"], [
            [0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0],
        ])
        mp_df = pd.DataFrame([{
            "material_id": "mp-2",
            "structure": structure_dict,
            "symmetry": {"crystal_system": "Cubic", "symbol": "Fm-3m"},
        }])

        df = build_structures_dataframe(mp_df, elements=["Al", "Mg"])

        assert df.loc[0, "x"] == pytest.approx(0.25)

    def test_raises_for_wrong_number_of_elements(self):
        mp_df = pd.DataFrame([{"material_id": "mp-1", "structure": {}, "symmetry": {}}])
        with pytest.raises(ValueError, match="exactly 2 elements"):
            build_structures_dataframe(mp_df, elements=["Al", "Mg", "Cu"])


class TestAppendStructure:
    def test_appends_row_with_correct_fractions_and_x(self):
        df = pd.DataFrame({"material_id": ["mp-1"], "x": [0.0]})
        atoms = Atoms("Al1Mg3")

        out = append_structure(df, atoms, elements=["Al", "Mg"], material_id="external-beta")

        assert len(out) == 2
        new_row = out.iloc[1]
        assert new_row["material_id"] == "external-beta"
        assert new_row["material_id_underscore"] == "external_beta"
        assert new_row["formula_pretty"] == atoms.get_chemical_formula()
        assert new_row["x"] == pytest.approx(0.75)
        assert new_row["structure_ase"] is atoms

    def test_formula_pretty_override_and_extra_columns(self):
        df = pd.DataFrame({"material_id": []})
        atoms = Atoms("Al1Mg3")

        out = append_structure(
            df, atoms, elements=["Al", "Mg"], material_id="external-beta",
            formula_pretty="Al534Mg345", energy_per_atom_calc=-2.5,
        )

        row = out.iloc[0]
        assert row["formula_pretty"] == "Al534Mg345"
        assert row["energy_per_atom_calc"] == -2.5

    def test_original_dataframe_is_not_mutated(self):
        df = pd.DataFrame({"material_id": ["mp-1"]})
        append_structure(df, Atoms("Al1Mg1"), elements=["Al", "Mg"], material_id="external")
        assert len(df) == 1
