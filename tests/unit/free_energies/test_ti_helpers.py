"""
Unit tests for phase_diagram_workflows.free_energies.ti_helpers.

Pure validation/utility functions: no calphy, no executor.
"""

import os
import tempfile

import pandas as pd
import pytest
from ase.atoms import Atoms
from ase.build import bulk

from phase_diagram_workflows.free_energies.ti_helpers import (
    _validate_calphy_parameters,
    _validate_input_structure,
    _validate_potential_df,
    _working_directory_context,
)

class TestValidateInputStructure:
    """Tests for _validate_input_structure()"""
    
    def test_valid_structure(self):
        """Test that valid structure passes without error"""
        structure = bulk('Al', cubic=True)
        _validate_input_structure(structure)  # Should not raise
    
    def test_empty_structure_raises_error(self):
        """Test that empty structure raises ValueError"""
        structure = Atoms()
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_input_structure(structure)


class TestValidatePotentialDf:
    """Tests for _validate_potential_df()"""
    
    def test_valid_potential_df(self):
        """Test that valid DataFrame passes without error"""
        df = pd.DataFrame({
            'Species': [['Al']],
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        _validate_potential_df(df)  # Should not raise
    
    def test_missing_Species_column_raises_error(self):
        """Test that missing Species column raises ValueError"""
        df = pd.DataFrame({
            'Config': [['pair_style eam', 'pair_coeff * * Al.eam']]
        })
        with pytest.raises(ValueError, match="Species"):
            _validate_potential_df(df)
    
    def test_missing_Config_column_raises_error(self):
        """Test that missing Config column raises ValueError"""
        df = pd.DataFrame({
            'Species': [['Al']]
        })
        with pytest.raises(ValueError, match="Config"):
            _validate_potential_df(df)


class TestValidateCalphyParameters:
    """Tests for _validate_calphy_parameters()"""
    
    def test_valid_parameters(self):
        """Test that valid parameters pass without error"""
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        _validate_calphy_parameters(params)  # Should not raise
    
    def test_missing_mode_raises_error(self):
        """Test that missing mode raises ValueError"""
        params = {
            'temperature': 300,
            'reference_phase': 'solid'
        }
        with pytest.raises(ValueError, match="mode"):
            _validate_calphy_parameters(params)
    
    def test_missing_temperature_raises_error(self):
        """Test that missing temperature raises ValueError"""
        params = {
            'mode': 'fe',
            'reference_phase': 'solid'
        }
        with pytest.raises(ValueError, match="temperature"):
            _validate_calphy_parameters(params)
    
    def test_missing_reference_phase_raises_error(self):
        """Test that missing reference_phase raises ValueError"""
        params = {
            'mode': 'fe',
            'temperature': 300
        }
        with pytest.raises(ValueError, match="reference_phase"):
            _validate_calphy_parameters(params)
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        params = {
            'mode': 'invalid',
            'temperature': 300,
            'reference_phase': 'solid'
        }
        with pytest.raises(ValueError, match="mode must be"):
            _validate_calphy_parameters(params)
    
    def test_invalid_reference_phase_raises_error(self):
        """Test that invalid reference_phase raises ValueError"""
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'gas'
        }
        with pytest.raises(ValueError, match="reference_phase must be"):
            _validate_calphy_parameters(params)
    
    def test_valid_ts_mode(self):
        """Test that 'ts' mode is accepted"""
        params = {
            'mode': 'ts',
            'temperature': [300, 600],
            'reference_phase': 'solid'
        }
        _validate_calphy_parameters(params)  # Should not raise
    
    def test_valid_liquid_phase(self):
        """Test that 'liquid' phase is accepted"""
        params = {
            'mode': 'fe',
            'temperature': 300,
            'reference_phase': 'liquid'
        }
        _validate_calphy_parameters(params)  # Should not raise


class TestWorkingDirectoryContext:
    """Tests for _working_directory_context()"""
    
    def test_context_manager_changes_directory(self):
        """Test that context manager changes to specified directory"""
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            with _working_directory_context(tmpdir):
                assert os.getcwd() == tmpdir
            # Should be back to original directory
            assert os.getcwd() == original_dir
    
    def test_context_manager_creates_directory(self):
        """Test that context manager creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'subdir', 'nested')
            assert not os.path.exists(new_dir)
            
            with _working_directory_context(new_dir):
                assert os.path.exists(new_dir)
                assert os.getcwd() == new_dir
    
    def test_context_manager_restores_directory_on_error(self):
        """Test that context manager restores directory even on error"""
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with _working_directory_context(tmpdir):
                    raise RuntimeError("Test error")
            except RuntimeError:
                pass
            
            # Should be back to original directory despite error
            assert os.getcwd() == original_dir

