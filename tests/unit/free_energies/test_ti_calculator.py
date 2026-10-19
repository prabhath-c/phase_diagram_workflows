"""
Unit tests for phase_diagram_workflows.free_energies.ti_calculator.

All external/heavy dependencies (_run_calphy, gather_calphy_results_detailed,
_build_calphy_config) are mocked here -- these tests check orchestration and
input-validation behavior, not real calphy execution. See
tests/integration/free_energies/test_ti_calculator.py for real-execution
coverage of gather_calphy_results_detailed.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from ase.atoms import Atoms
from ase.build import bulk

class TestCalcFreeEnergyWithCalphydIntegration:
    """Integration tests for main workflow with mocking"""
    
    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_main_function_calls_correct_sequence(
        self, mock_build_config, mock_gather, mock_run_calphy
    ):
        """Test that main function calls helper functions in correct sequence"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy
        
        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})
        
        # Setup inputs
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call main function
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                user_dict={}
            )
            
            # Verify correct calls were made
            mock_build_config.assert_called_once()
            mock_run_calphy.assert_called_once()
            mock_gather.assert_called_once()
            
            # Verify return values
            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)


class TestInputValidationErrors:
    """Test main workflow input validation"""

    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_invalid_structure_raises_error(self, mock_build_config):
        """Test that invalid structure raises ValueError"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        structure = Atoms()  # Empty - invalid
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Input validation failed"):
                calc_free_energy_with_calphy(
                    input_structure=structure,
                    potential_df=potential_df,
                    calphy_parameters=params,
                    working_directory=tmpdir,
                    user_dict={}
                )

    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_invalid_potential_raises_error(self, mock_build_config):
        """Test that invalid potential DataFrame raises ValueError"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({'NotSpecies': [['Al']]})  # Wrong column
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Input validation failed"):
                calc_free_energy_with_calphy(
                    input_structure=structure,
                    potential_df=potential_df,
                    calphy_parameters=params,
                    working_directory=tmpdir,
                    user_dict={}
                )

    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_invalid_parameters_raises_error(self, mock_build_config):
        """Test that invalid parameters raises ValueError"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {'mode': 'invalid'}  # Missing temperature, reference_phase

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Input validation failed"):
                calc_free_energy_with_calphy(
                    input_structure=structure,
                    potential_df=potential_df,
                    calphy_parameters=params,
                    working_directory=tmpdir,
                    user_dict={}
                )

class TestExecutorIntegration:
    """Tests for executor-based workflow with metadata and LAMMPS library"""

    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_executor_workflow_with_metadata_and_lammps(
        self, mock_build_config, mock_gather, mock_run_calphy
    ):
        """Test executor-based workflow with metadata and LAMMPS library"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy
        import executorlib
        import pylammpsmpi
        SingleNodeExecutor = executorlib.SingleNodeExecutor
        LammpsLibrary = pylammpsmpi.LammpsLibrary

        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})

        # Setup executor and LAMMPS library (mocked)
        mock_executor = Mock(spec=SingleNodeExecutor)
        mock_lmp = Mock(spec=LammpsLibrary)

        # Setup inputs with metadata (same as notebook executor example)
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        metadata_dict = {
            'project': 'test',
            'material': 'Al',
            'temperature': 300,
            'method': 'executor'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call with executor, LAMMPS library, and metadata
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                lmp=mock_lmp,
                metadata_dict=metadata_dict,
                user_dict={}
            )

            # Verify results
            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty

    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_metadata_validation(self, mock_build_config, mock_gather, mock_run_calphy):
        """Test that metadata dictionary is handled correctly"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy

        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})

        # Test with empty metadata (should work)
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with empty metadata
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                metadata_dict={},  # Empty metadata should be allowed
                user_dict={}
            )

            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)

    @patch('phase_diagram_workflows.free_energies.ti_calculator._run_calphy')
    @patch('phase_diagram_workflows.free_energies.ti_calculator.gather_calphy_results_detailed')
    @patch('phase_diagram_workflows.free_energies.ti_calculator._build_calphy_config')
    def test_lammps_library_integration(self, mock_build_config, mock_gather, mock_run_calphy):
        """Test that LAMMPS library parameter is handled correctly"""
        from phase_diagram_workflows.free_energies.ti_calculator import calc_free_energy_with_calphy
        import pylammpsmpi
        LammpsLibrary = pylammpsmpi.LammpsLibrary

        # Setup mocks
        mock_calculation = Mock()
        mock_build_config.return_value = mock_calculation
        mock_gather.return_value = pd.DataFrame({'energy': [1.0]})

        # Setup LAMMPS library mock
        mock_lmp = Mock(spec=LammpsLibrary)

        # Test with LAMMPS library
        structure = bulk('Al', cubic=True)
        potential_df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with LAMMPS library
            result_calc, result_df = calc_free_energy_with_calphy(
                input_structure=structure,
                potential_df=potential_df,
                calphy_parameters=params,
                working_directory=tmpdir,
                lmp=mock_lmp,
                user_dict={}
            )

            assert result_calc == mock_calculation
            assert isinstance(result_df, pd.DataFrame)

    def test_metadata_content_validation(self):
        """Test metadata content validation"""
        from phase_diagram_workflows.free_energies.ti_helpers import _validate_metadata

        # Test valid metadata
        valid_metadata = {
            'project': 'test',
            'material': 'Al',
            'temperature': 300,
            'method': 'executor'
        }
        _validate_metadata(valid_metadata)  # Should not raise

        # Test empty metadata (should be allowed)
        _validate_metadata({})  # Should not raise

        # Test metadata with extra fields (should be allowed)
        extended_metadata = {
            'project': 'test',
            'material': 'Al',
            'temperature': 300,
            'method': 'executor',
            'additional_field': 'extra_value'
        }
        _validate_metadata(extended_metadata)  # Should not raise


class TestRunCalphyDispatch:
    """Unit tests for _run_calphy's job-class dispatch logic.

    mode="composition_scaling" must always route to calphy's Alchemy class (it owns the
    switching + MC-swap + mass_integration machinery, per calphy/queuekernel.py's own
    dispatch), regardless of reference_phase -- not to Solid/Liquid, which never had that
    machinery and silently produced physically-wrong results for this mode.
    """

    @staticmethod
    def _make_input_class(mode, reference_phase, lattice_path):
        mock_input = Mock()
        mock_input.mode = mode
        mock_input.reference_phase = reference_phase
        mock_input.lattice = lattice_path
        return mock_input

    @patch('calphy.routines.routine_composition_scaling')
    @patch('calphy.Alchemy')
    @patch('calphy.Solid')
    @patch('calphy.Liquid')
    def test_composition_scaling_dispatches_to_alchemy_not_solid(
        self, mock_liquid, mock_solid, mock_alchemy, mock_routine_cs
    ):
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('composition_scaling', 'solid', lattice_path)
            mock_job = Mock()
            mock_alchemy.return_value = mock_job

            _run_calphy(input_class=input_class)

            mock_alchemy.assert_called_once_with(calculation=input_class, simfolder=tmpdir)
            mock_solid.assert_not_called()
            mock_liquid.assert_not_called()
            mock_routine_cs.assert_called_once_with(mock_job)

    @patch('calphy.routines.routine_composition_scaling')
    @patch('calphy.Alchemy')
    @patch('calphy.Liquid')
    def test_composition_scaling_dispatches_to_alchemy_not_liquid(
        self, mock_liquid, mock_alchemy, mock_routine_cs
    ):
        """reference_phase='liquid' should not matter either -- mode alone decides."""
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('composition_scaling', 'liquid', lattice_path)
            mock_alchemy.return_value = Mock()

            _run_calphy(input_class=input_class)

            mock_alchemy.assert_called_once_with(calculation=input_class, simfolder=tmpdir)
            mock_liquid.assert_not_called()

    @patch('calphy.routines.routine_composition_scaling')
    @patch('calphy.Alchemy')
    def test_composition_scaling_passes_lmp_through_to_alchemy(
        self, mock_alchemy, mock_routine_cs
    ):
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('composition_scaling', 'solid', lattice_path)
            sentinel_lmp = Mock(name='lmp')
            mock_alchemy.return_value = Mock()

            _run_calphy(input_class=input_class, lmp=sentinel_lmp)

            mock_alchemy.assert_called_once_with(
                calculation=input_class, simfolder=tmpdir, lmp=sentinel_lmp
            )

    @patch('calphy.Alchemy')
    def test_composition_scaling_rejects_an_invalid_reference_phase(self, mock_alchemy):
        """The reference_phase is checked for every mode, not only for fe and ts."""
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('composition_scaling', 'gas', lattice_path)

            with pytest.raises(ValueError, match="Invalid reference_phase"):
                _run_calphy(input_class=input_class)

            mock_alchemy.assert_not_called()

    @patch('calphy.routines.routine_fe')
    @patch('calphy.Solid')
    def test_fe_mode_still_dispatches_to_solid(self, mock_solid, mock_routine_fe):
        """Regression check: composition_scaling's new branch must not affect fe/ts."""
        from phase_diagram_workflows.free_energies.ti_calculator import _run_calphy

        with tempfile.TemporaryDirectory() as tmpdir:
            lattice_path = os.path.join(tmpdir, 'input_structure.data')
            input_class = self._make_input_class('fe', 'solid', lattice_path)
            mock_job = Mock()
            mock_solid.return_value = mock_job

            _run_calphy(input_class=input_class)

            mock_solid.assert_called_once_with(calculation=input_class, simfolder=tmpdir)
            mock_routine_fe.assert_called_once_with(mock_job)


def _write_calphy_folder(folder, input_overrides, report):
    """A minimal calphy output folder: input_file.yaml and (if report is not None) report.yaml."""
    import yaml

    folder.mkdir(parents=True, exist_ok=True)
    calculation = {"reference_phase": "solid", "pressure": 0.0, "temperature": 500}
    calculation.update(input_overrides)
    (folder / "input_file.yaml").write_text(yaml.safe_dump({"calculations": [calculation]}))
    if report is not None:
        (folder / "report.yaml").write_text(yaml.safe_dump(report))
        if input_overrides.get("mode") == "composition_scaling":    # a finished job has its forward sweep
            (folder / "forward_1.dat").write_text("# dU_1 dU_2 lambda\n-3.5 -3.3 1.0\n-3.5 -3.3 0.0\n")
    return str(folder)


# the values of the ACE test job: 500 atoms of Al, 25 turned into Mg, 500 K
COMPOSITION_SCALING_REPORT = {
    "input": {"element": "Al Mg", "concentration": "0.95 0.05"},
    "results": {
        "free_energy": 0.11447121891913319,
        "work": 0.11413362158866863,
        "mass_correction": -0.00033759733046455234,
        "dissipation": 2.539714182888536e-05,
        "entropy_contribution": -1.7106720095272323e-05,
        "error": 0.0,
    },
}
FE_REPORT = {
    "input": {"element": "Al", "concentration": "1.0"},
    "results": {"free_energy": -3.5, "dissipation": 1e-5},
}


class TestGatherCompositionScalingResults:
    """gather_calphy_results_detailed() for mode='composition_scaling'"""

    def test_delta_and_the_pieces_of_it_are_separate_columns(self, tmp_path):
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        folder = _write_calphy_folder(
            tmp_path / "cs", {"mode": "composition_scaling", "reference_composition": 0.0}, COMPOSITION_SCALING_REPORT
        )
        row = gather_calphy_results_detailed(folder).iloc[0]

        assert row["calculation_mode"] == "composition_scaling"
        assert row["status"]
        assert row["composition"] == pytest.approx({"Al": 0.95, "Mg": 0.05})
        assert row["reference_composition"] == 0.0
        assert row["delta_free_energy"] == pytest.approx(0.11447121891913319)
        assert row["work"] == pytest.approx(0.11413362158866863)
        assert row["mass_correction"] == pytest.approx(-0.00033759733046455234)
        # calphy's convention: ideal_entropy = -entropy_contribution of the report (positive, eV/atom/K),
        # and fix_composition_scaling subtracts T * ideal_entropy
        assert row["ideal_entropy"] == pytest.approx(1.7106720095272323e-05)
        assert row["delta_free_energy_with_ideal_entropy"] == pytest.approx(
            0.11447121891913319 - 500 * 1.7106720095272323e-05
        )
        assert row["dissipation"] == pytest.approx(2.539714182888536e-05)

    def test_results_have_only_the_difference_not_an_absolute_free_energy(self, tmp_path):
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        folder = _write_calphy_folder(
            tmp_path / "cs", {"mode": "composition_scaling", "reference_composition": 0.0}, COMPOSITION_SCALING_REPORT
        )
        row = gather_calphy_results_detailed(folder).iloc[0]

        # no absolute free energy, and no placeholder for one
        for column in ["free_energy", "reference_free_energy", "free_energy_with_ideal_entropy"]:
            assert column not in row.index

    def test_running_job_has_no_energy_of_the_target_and_no_warning(self, tmp_path):
        """No report.yaml yet: forward_1.dat is still being written, even with a single row, and that is normal."""
        import warnings
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        folder = _write_calphy_folder(tmp_path / "cs", {"mode": "composition_scaling", "reference_composition": 0.0}, None)
        (tmp_path / "cs" / "forward_1.dat").write_text("# dU_1 dU_2 lambda\n-3.5 -3.3 1.0\n")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            row = gather_calphy_results_detailed(folder).iloc[0]

        assert pd.isna(row["energy_target"])

    def test_finished_job_with_an_empty_forward_file_warns_and_gives_nan(self, tmp_path):
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        folder = _write_calphy_folder(
            tmp_path / "cs", {"mode": "composition_scaling", "reference_composition": 0.0}, COMPOSITION_SCALING_REPORT
        )
        (tmp_path / "cs" / "forward_1.dat").write_text("# dU_1 dU_2 lambda\n")

        with pytest.warns(UserWarning, match="has no data"):
            row = gather_calphy_results_detailed(folder).iloc[0]

        assert pd.isna(row["energy_target"])
        assert row["delta_free_energy"] == pytest.approx(0.11447121891913319)   # the rest of the row is not affected

    def test_finished_job_without_a_forward_file_warns_and_gives_nan(self, tmp_path):
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        folder = _write_calphy_folder(
            tmp_path / "cs", {"mode": "composition_scaling", "reference_composition": 0.0}, COMPOSITION_SCALING_REPORT
        )
        (tmp_path / "cs" / "forward_1.dat").unlink()

        with pytest.warns(UserWarning, match="no forward_1.dat"):
            row = gather_calphy_results_detailed(folder).iloc[0]

        assert pd.isna(row["energy_target"])

    def test_job_without_report_gives_nan_not_an_error(self, tmp_path):
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        folder = _write_calphy_folder(
            tmp_path / "cs", {"mode": "composition_scaling", "reference_composition": 0.0}, None
        )
        row = gather_calphy_results_detailed(folder).iloc[0]

        assert not row["status"]
        assert pd.isna(row["delta_free_energy"])
        assert pd.isna(row["dissipation"]) and pd.isna(row["energy_target"])

    def test_energy_of_the_target_is_read_from_the_end_of_the_forward_switching(self, tmp_path):
        import numpy as np
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        folder = _write_calphy_folder(
            tmp_path / "cs", {"mode": "composition_scaling", "reference_composition": 0.0}, COMPOSITION_SCALING_REPORT
        )
        lam = np.linspace(1.0, 0.0, 1000)
        for iteration, target_energy in [(1, -3.35), (2, -3.37)]:   # dU_1 is the reference potential: it must not be used
            rows = np.column_stack([np.full_like(lam, -3.58), np.full_like(lam, target_energy), lam])
            np.savetxt(tmp_path / "cs" / f"forward_{iteration}.dat", rows, header="dU_1[eV/atom] dU_2[eV/atom] lambda")

        row = gather_calphy_results_detailed(folder).iloc[0]

        assert row["energy_target"] == pytest.approx(-3.36)   # mean over the two iterations

    def test_reference_fe_job_gives_dissipation(self, tmp_path):
        from phase_diagram_workflows.free_energies.ti_calculator import gather_calphy_results_detailed

        row = gather_calphy_results_detailed(
            _write_calphy_folder(tmp_path / "fe", {"mode": "fe"}, FE_REPORT)
        ).iloc[0]

        assert row["dissipation"] == pytest.approx(1e-5)
