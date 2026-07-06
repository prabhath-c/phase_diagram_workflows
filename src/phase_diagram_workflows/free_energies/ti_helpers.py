from __future__ import annotations

import os
from typing import Dict, Any, Tuple, Optional

from ase.data import chemical_symbols, atomic_masses
from ase.atoms import Atoms
from contextlib import contextmanager
import pandas as pd
from pydantic import ValidationError
from lammpsparser import write_lammps_structure

def _validate_input_structure(structure: Atoms) -> None:
    """Validate that input structure is not empty.
    
    Parameters
    ----------
    structure : Atoms
        Structure to validate
        
    Raises
    ------
    ValueError
        If structure is empty
    """
    if len(structure) == 0:
        raise ValueError("input_structure cannot be empty")

def _validate_potential_df(potential_df: pd.DataFrame) -> None:
    """Validate that potential_df has required columns from pyiron.
    
    Parameters
    ----------
    potential_df : pd.DataFrame
        Potential DataFrame to validate
        
    Raises
    ------
    ValueError
        If required columns are missing
    """
    required_cols = {'Species', 'Config'}
    missing_cols = required_cols - set(potential_df.columns)
    if missing_cols:
        raise ValueError(f"potential_df missing required columns: {missing_cols}")

def _validate_calphy_parameters(calphy_parameters: Dict[str, Any]) -> None:
    """Validate that calphy_parameters has required keys and valid values.
    
    Parameters
    ----------
    calphy_parameters : Dict[str, Any]
        Parameters to validate
        
    Raises
    ------
    ValueError
        If required keys are missing or have invalid values
    """
    required_keys = {'mode', 'temperature', 'reference_phase'}
    missing_keys = required_keys - set(calphy_parameters.keys())
    if missing_keys:
        raise ValueError(f"calphy_parameters missing required keys: {missing_keys}")
    
    # Validate mode and reference_phase values
    valid_modes = {'fe', 'ts'}
    if calphy_parameters['mode'] not in valid_modes:
        raise ValueError(f"mode must be 'fe' or 'ts', got '{calphy_parameters['mode']}'")
    
    valid_phases = {'solid', 'liquid'}
    if calphy_parameters['reference_phase'] not in valid_phases:
        raise ValueError(f"reference_phase must be 'solid' or 'liquid', got '{calphy_parameters['reference_phase']}'")

@contextmanager
def _working_directory_context(path: str):
    """Context manager to temporarily change working directory.
    
    Changes to the specified directory, creates it if it doesn't exist,
    and ensures the original directory is restored on exit (even if errors occur).
    
    Parameters
    ----------
    path : str
        Target directory path
        
    Yields
    ------
    None
    
    Examples
    --------
    >>> with _working_directory_context('/tmp/workdir'):
    ...     # Code here runs in /tmp/workdir
    ...     os.getcwd()  # returns /tmp/workdir
    >>> os.getcwd()  # back to original directory
    """
    prev_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

def _write_structure(
    structure: Atoms,
    potential_df: pd.DataFrame,
    file_name: str,
    working_directory: str
) -> None:
    """Write ASE structure to LAMMPS data file format.
    
    Uses lammpsparser to convert ASE Atoms object to LAMMPS format and
    writes it to the specified working directory. Validates that all elements
    in the structure are supported by the selected potential. LAMMPS units are
    fixed to ``metal`` in this helper.
    
    Parameters
    ----------
    structure : Atoms
        ASE Atoms object to write
    potential_df : pd.DataFrame
        Potential DataFrame in pyiron-compatible format
    file_name : str
        Output file name (will be written to working_directory)
    working_directory : str
        Output working directory
        
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If structure contains elements not supported by the potential
    RuntimeError
        If file writing fails
    """
    try:
        potential_elements = potential_df["Species"].to_list()[0]
        structure_elements = set(structure.get_chemical_symbols())
        potential_elements_set = set(potential_elements)

        if not structure_elements.issubset(potential_elements_set):
            unsupported = structure_elements - potential_elements_set
            raise ValueError(
                f"The selected potential does not support element(s): {unsupported}. "
                f"Potential supports: {potential_elements_set}"
            )

        write_lammps_structure(
            structure=structure,
            potential_elements=potential_elements,
            units="metal",
            file_name=file_name,
            working_directory=working_directory,
        )
    
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to write structure to file '{file_name}' in '{working_directory}': "
            f"{type(e).__name__}: {str(e)}"
        ) from e

def _ensure_potential(calphy_parameters: Dict[str, Any], potential_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract and ensure pair_style and pair_coeff parameters from potential.
    
    If pair_style or pair_coeff are missing from calphy_parameters, extracts them
    from the potential DataFrame (pyiron-compatible format).
    
    Parameters
    ----------
    calphy_parameters : Dict[str, Any]
        Calphy parameters dictionary to update
    potential_df : pd.DataFrame
        Potential DataFrame in pyiron-compatible format with 'Config' column
        
    Returns
    -------
    Dict[str, Any]
        Updated calphy_parameters with pair_style and pair_coeff set
        
    Raises
    ------
    ValueError
        If Config column is missing or cannot be parsed
    RuntimeError
        If extraction fails
    """
    try:
        if "pair_style" not in calphy_parameters or "pair_coeff" not in calphy_parameters:
            if 'Config' not in potential_df.columns:
                raise ValueError("'Config' column not found in potential_df")
            
            config_list = potential_df['Config'].tolist()
            if not config_list or not config_list[0]:
                raise ValueError("Config data is empty")

            pair_style = None
            pair_coeff = None
            for line in config_list[0]:
                stripped_line = str(line).strip()
                if stripped_line.startswith("pair_style"):
                    pair_style = stripped_line.replace("pair_style", "", 1).strip()
                elif stripped_line.startswith("pair_coeff"):
                    pair_coeff = stripped_line.replace("pair_coeff", "", 1).strip()

            missing_items = [
                item
                for item, value in (("pair_style", pair_style), ("pair_coeff", pair_coeff))
                if not value
            ]
            if missing_items:
                raise ValueError(
                    f"Could not extract {', '.join(missing_items)} from potential_df['Config']"
                )
            
            # update dict if missing
            if "pair_style" not in calphy_parameters:
                calphy_parameters["pair_style"] = pair_style
            if "pair_coeff" not in calphy_parameters:
                calphy_parameters["pair_coeff"] = pair_coeff
        
        return calphy_parameters
    
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract potential parameters: {type(e).__name__}: {str(e)}"
        ) from e

def _ensure_elements_and_masses(
    input_structure: Atoms,
    potential_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Ensure 'element' and 'mass' keys exist in calphy parameters.
    
    If missing, computes them from the potential dataframe and input structure.
    
    Parameters
    ----------
    input_structure : Atoms
        ASE Atoms object with the structure
    potential_df : pd.DataFrame
        Potential DataFrame in pyiron-compatible format
    calphy_parameters : Dict[str, Any]
        Calphy parameters dictionary to update
        
    Returns
    -------
    Dict[str, Any]
        Updated calphy_parameters with element and mass keys set
        
    Raises
    ------
    ValueError
        If Species column is missing from potential_df
    RuntimeError
        If mass calculation fails
    """
    try:
        if "element" not in calphy_parameters or "mass" not in calphy_parameters:
            if 'Species' not in potential_df.columns:
                raise ValueError("'Species' column not found in potential_df")

            structure_symbols = list(set(input_structure.get_chemical_symbols()))

            element_symbols = potential_df["Species"].tolist()[0]
            # masses = list([atomic_masses[chemical_symbols.index(el)] for el in element_symbols])    
            
            masses = [
                atomic_masses[chemical_symbols.index(el)] if el in structure_symbols else 1.0
                for el in element_symbols
            ]

            if "element" not in calphy_parameters:
                calphy_parameters["element"] = element_symbols
            if "mass" not in calphy_parameters:
                calphy_parameters["mass"] = masses

        return calphy_parameters
    
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to set elements and masses: {type(e).__name__}: {str(e)}"
        ) from e

def _validate_metadata(metadata: Dict[str, Any]) -> None:
    """Validate the metadata dictionary.

    Any dict (including empty) is accepted. Raises TypeError if the argument
    is not a dict.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Metadata dictionary to validate

    Raises
    ------
    TypeError
        If metadata is not a dictionary
    """
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a dict, got {type(metadata).__name__}")


def _create_input_class(input_parameters: Dict[str, Any]) -> Calculation:
    """Create and validate a calphy Calculation object from input parameters.
    
    Parameters
    ----------
    input_parameters : Dict[str, Any]
        Dictionary of parameters for calphy Calculation
        
    Returns
    -------
    Calculation
        Validated calphy Calculation object
        
    Raises
    ------
    ValueError
        If parameters fail validation against calphy's Calculation model
    """
    try:
        from calphy import Calculation
        return Calculation.model_validate(input_parameters)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters: {e}") from e

def _build_calphy_config(
    input_structure: Atoms,
    potential_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any],
    working_directory: Optional[str]
) -> Calculation:
    """Build complete calphy configuration from structure, potential, and parameters.
    
    Orchestrates the configuration building by:
    1. Writing the input structure to LAMMPS format (if not provided)
    2. Extracting potential parameters (pair_style, pair_coeff)
    3. Setting element types and atomic masses
    4. Creating and returning a validated Calculation object
    
    Parameters
    ----------
    input_structure : Atoms
        ASE Atoms object with the structure
    potential_df : pd.DataFrame
        Potential DataFrame in pyiron-compatible format
    calphy_parameters : Dict[str, Any]
        Parameters for calphy calculation
    working_directory : Optional[str]
        Directory for writing files. If None, uses current directory
        
    Returns
    -------
    Calculation
        Configured and validated calphy Calculation object
        
    Raises
    ------
    ValueError
        If configuration is invalid
    RuntimeError
        If configuration building fails
    """
    try:
        if working_directory is None:
            working_directory = os.getcwd()
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
        
        if 'lattice' not in calphy_parameters:
            _write_structure(
                structure=input_structure, 
                potential_df=potential_df, 
                file_name='input_structure.data', 
                working_directory=working_directory
            )
            
            lattice_file = os.path.join(working_directory, 'input_structure.data')
            calphy_parameters["lattice"] = lattice_file

        ## FIXME: Check calphy pyiron job for handling elements, masses etc
        input_parameters = _ensure_potential(calphy_parameters, potential_df)
        input_parameters = _ensure_elements_and_masses(input_structure, potential_df, input_parameters)
        
        input_class = _create_input_class(input_parameters=input_parameters)

        return input_class
    
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to build calphy configuration: {type(e).__name__}: {str(e)}"
        ) from e