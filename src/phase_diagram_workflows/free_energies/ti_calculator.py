from __future__ import annotations

import os
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

import numpy as np
import yaml
from ase.atoms import Atoms
from calphy import Calculation
import pandas as pd

from phase_diagram_workflows.free_energies.ti_helpers import (
    _working_directory_context,
    _build_calphy_config,
    _validate_input_structure,
    _validate_potential_df,
    _validate_calphy_parameters,
)

def _run_calphy(input_class: Calculation, lmp: Optional[Any] = None) -> None:
    """Execute calphy calculation based on the input configuration.

    Parameters
    ----------
    input_class : Calculation
        Calphy Calculation object with all parameters configured
    lmp : Optional[Any], optional
        Optional LAMMPS library object from pylammpsmpi with embedded executor.
        If provided, the calculation will use this lmp object instead of creating
        its own, enabling executor-based parallel execution.

    Raises
    ------
    ValueError
        If reference_phase is not 'solid' or 'liquid'
        If mode is not 'fe' (free energy), 'ts' (temperature scaling), or 'composition_scaling'
    RuntimeError
        If calphy execution fails
    """
    # Use the working directory from the lattice path (this is where files are written)
    lattice_path = Path(input_class.lattice)
    working_directory = str(lattice_path.parent)

    with _working_directory_context(working_directory):
        try:
            from calphy import Solid, Liquid
            from calphy.routines import routine_fe, routine_ts, routine_composition_scaling
            if input_class.reference_phase == "solid":
                if lmp is not None:
                    job = Solid(calculation=input_class, simfolder=working_directory, lmp=lmp)
                else:
                    job = Solid(calculation=input_class, simfolder=working_directory)
            elif input_class.reference_phase == "liquid":
                if lmp is not None:
                    job = Liquid(calculation=input_class, simfolder=working_directory, lmp=lmp)
                else:
                    job = Liquid(calculation=input_class, simfolder=working_directory)
            else:
                raise ValueError(
                    f"Invalid reference_phase: {input_class.reference_phase}. "
                    "Must be 'solid' or 'liquid'"
                )

            if input_class.mode == "fe":
                routine_fe(job)
            elif input_class.mode == "ts":
                routine_ts(job)
            elif input_class.mode == "composition_scaling":
                routine_composition_scaling(job)
            else:
                raise ValueError(
                    f"Invalid mode: {input_class.mode}. Must be 'fe', 'ts', or 'composition_scaling'"
                )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Calphy execution failed with {type(e).__name__}: {str(e)}"
            ) from e

def gather_calphy_results(parent_directory: str) -> pd.DataFrame:
    """Gather and return results from calphy calculations.

    Parameters
    ----------
    parent_directory : str
        Path to parent directory containing calphy calculation folders

    Returns
    -------
    pd.DataFrame
        DataFrame containing aggregated results from all calculations
    """
    with _working_directory_context(parent_directory):
        from calphy.postprocessing import gather_results
        df = gather_results('.')
    return df


def gather_calphy_results_detailed(working_directory: str) -> pd.DataFrame:
    """Read calphy results from a single calculation directory.

    Independent of calphy's gather_results. Reads temperature sweep data
    and forward/backward energy differences for phase-transition checking.

    Parameters
    ----------
    working_directory : str
        Path to a single calphy calculation directory (contains input_file.yaml)

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        - calculation: folder name
        - calculation_mode: 'ts' or 'fe'
        - reference_phase: 'solid' or 'liquid'
        - pressure: float
        - status: bool
        - composition: dict mapping element to concentration
        - temperature: array (ts) or scalar (fe)
        - free_energy: array (ts) or scalar (fe)
        - free_energy_error: array (ts) or 0.0 (fe)
        - forward_energy_diff: list of arrays, one per iteration (ts only)
        - backward_energy_diff: list of arrays, one per iteration (ts only)
        - forward_lambda: list of arrays, one per iteration (ts only)
        - backward_lambda: list of arrays, one per iteration (ts only)

    Raises
    ------
    FileNotFoundError
        If input_file.yaml is not found in working_directory
    """
    wd = os.path.abspath(working_directory)
    folder_name = os.path.basename(wd)

    inpfile = os.path.join(wd, "input_file.yaml")
    if not os.path.exists(inpfile):
        raise FileNotFoundError(f"No input_file.yaml found in {wd}")

    with open(inpfile) as f:
        inp_yaml = yaml.safe_load(f)
    inp = inp_yaml["calculations"][0]
    mode = inp["mode"]

    data: Dict[str, Any] = {
        "calculation": folder_name,
        "calculation_mode": mode,
        "reference_phase": inp["reference_phase"],
        "pressure": inp["pressure"],
    }

    repfile = os.path.join(wd, "report.yaml")
    if os.path.exists(repfile):
        with open(repfile) as f:
            rep = yaml.safe_load(f)
        data["status"] = True
        el_arr = np.array(rep["input"]["element"].split()).astype(str)
        comp_arr = np.array(rep["input"]["concentration"].split()).astype(float)
        data["composition"] = {x: float(y) for x, y in zip(el_arr, comp_arr)}
    else:
        data["status"] = False
        data["composition"] = None
        rep = None

    if mode in ["ts", "tscale"]:
        tsfile = os.path.join(wd, "temperature_sweep.dat")
        if os.path.exists(tsfile):
            t, fe, ferr = np.loadtxt(tsfile, unpack=True, usecols=(0, 1, 2))
            data["temperature"] = t
            data["free_energy"] = fe
            data["free_energy_error"] = ferr
        else:
            data["temperature"] = np.asarray(inp["temperature"], dtype=float)
            data["free_energy"] = np.full_like(data["temperature"], np.nan)
            data["free_energy_error"] = np.full_like(data["temperature"], np.nan)

        f_ediffs: List[np.ndarray] = []
        b_ediffs: List[np.ndarray] = []
        f_lambdas: List[np.ndarray] = []
        b_lambdas: List[np.ndarray] = []
        i = 1
        while True:
            fwdfile = os.path.join(wd, f"ts.forward_{i}.dat")
            bkdfile = os.path.join(wd, f"ts.backward_{i}.dat")
            if not (os.path.exists(fwdfile) and os.path.exists(bkdfile)):
                break
            fdx, _fp, _fvol, flambda = np.loadtxt(fwdfile, unpack=True, comments="#")
            bdx, _bp, _bvol, blambda = np.loadtxt(bkdfile, unpack=True, comments="#")
            f_ediffs.append(fdx / flambda)
            b_ediffs.append(bdx / blambda)
            f_lambdas.append(flambda)
            b_lambdas.append(blambda)
            i += 1

        data["forward_energy_diff"] = f_ediffs if f_ediffs else None
        data["backward_energy_diff"] = b_ediffs if b_ediffs else None
        data["forward_lambda"] = f_lambdas if f_lambdas else None
        data["backward_lambda"] = b_lambdas if b_lambdas else None

    else:  # fe mode
        data["temperature"] = inp["temperature"]
        data["free_energy"] = rep["results"]["free_energy"] if rep else np.nan
        data["free_energy_error"] = 0.0 if rep else np.nan
        data["forward_energy_diff"] = None
        data["backward_energy_diff"] = None
        data["forward_lambda"] = None
        data["backward_lambda"] = None

    return pd.DataFrame([data])

def calc_free_energy_with_calphy(
    input_structure: Atoms,
    potential_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any],
    working_directory: Optional[str],
    lmp: Optional[Any] = None,
    metadata_dict: Optional[Dict[str, Any]] = None,
    user_dict: Optional[Dict[str, Any]] = None
) -> Tuple[Calculation, pd.DataFrame]:
    """Main function to calculate free energy using calphy with LAMMPS potentials.

    Orchestrates the entire workflow: configures calphy parameters, writes structure
    files in LAMMPS format, executes calphy calculations, and gathers results.

    Parameters
    ----------
    input_structure : Atoms
        ASE Atoms object representing the crystal structure
    potential_df : pd.DataFrame
        DataFrame containing potential information in pyiron-compatible format
    calphy_parameters : Dict[str, Any]
        Dictionary with calphy parameters including:
        - mode: 'fe' (free energy), 'ts' (temperature scaling), or 'composition_scaling'
          (alchemical composition transformation with MC identity-exchange swaps;
          requires 'composition_scaling' and 'monte_carlo' blocks, see calphy.Calculation)
        - temperature: float or list for temperature range
        - reference_phase: 'solid' or 'liquid'
        - n_equilibration_steps, n_switching_steps, n_print_steps
        - equilibration_control: thermostat type (e.g., 'nose-hoover')
        - queue: dict with cores and scheduler info
        - file_format: 'lammps-data'
    working_directory : str
        Directory where calculations will be run
    lmp : Optional[Any], optional
        Optional LAMMPS library object from pylammpsmpi with embedded executor.
        If provided, the calculation will use this lmp object instead of creating
        its own, enabling executor-based parallel execution.
    metadata_dict : Optional[Dict[str, Any]], optional
        Optional dictionary for storing user-defined metadata in executorlib's cache.
        Used when lmp is provided to enable result caching and retrieval.
    user_dict : Optional[Dict[str, Any]], optional
        Optional dictionary for additional user-defined parameters. Accepted but
        currently unused; reserved for future extension.

    Returns
    -------
    Tuple[Calculation, pd.DataFrame]
        Tuple containing:
        - Calculation object: The calphy Calculation instance used
        - pd.DataFrame: Results DataFrame from gather_calphy_results()

    Examples
    --------
    # Basic usage without executor
    result = calc_free_energy_with_calphy(
        input_structure=structure,
        potential_df=potential_df,
        calphy_parameters=params,
        working_directory='output_dir'
    )

    # With executor
    executor = SingleNodeExecutor()
    lmp = LammpsLibrary(cores=1, executor=executor)
    result = calc_free_energy_with_calphy(
        input_structure=structure,
        potential_df=potential_df,
        calphy_parameters=params,
        working_directory='output_dir',
        lmp=lmp,
        metadata_dict={'project': 'my_project', 'version': '1.0'}
    )

    Raises
    ------
    ValueError
        If required parameters are missing or invalid. Type and value issues
        raised during validation are normalized to ValueError.
    RuntimeError
        If calculation execution fails
    """
    try:
        # Validate all inputs
        _validate_input_structure(input_structure)
        _validate_potential_df(potential_df)
        _validate_calphy_parameters(calphy_parameters)
        print("Input validation successful. Proceeding with calculation.")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Input validation failed: {str(e)}") from e

    try:
        if working_directory is None:
            working_directory = os.getcwd()
            print(f"No working directory provided. Using current directory {working_directory} as working directory.")
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)

        with _working_directory_context(working_directory):
            input_class = _build_calphy_config(
                input_structure=input_structure,
                potential_df=potential_df,
                calphy_parameters=calphy_parameters,
                working_directory=working_directory
            )

            _run_calphy(input_class=input_class, lmp=lmp)

        df = gather_calphy_results_detailed(working_directory)

        return input_class, df

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Free energy calculation workflow failed: {type(e).__name__}: {str(e)}"
        ) from e