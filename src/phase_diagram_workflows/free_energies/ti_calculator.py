from __future__ import annotations

import os
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

import numpy as np
import pylammpsmpi
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

# calphy's own internal LAMMPS driver (calphy/helpers.py) constructs
# pylammpsmpi.LammpsLibrary without passing hostname_localhost, so it
# inherits pylammpsmpi's default of advertising the machine's real hostname
# (via socket.gethostname()) for its own internal executor/worker process to
# zmq-connect back to, instead of localhost. On an ephemeral CI runner the
# worker can fail to resolve/connect to that hostname; the connection just
# retries silently rather than raising, so the calculation hangs instead of
# failing. This is the same class of issue pylammpsmpi's own test suite
# works around by passing hostname_localhost=True explicitly wherever it
# nests a LAMMPS wrapper inside its own executor (see pyiron/pylammpsmpi
# tests/test_executor.py). calphy exposes no parameter to pass this through,
# so it's patched here, once, for every calphy calculation this package runs.
_original_lammps_library_init = pylammpsmpi.LammpsLibrary.__init__


def _lammps_library_init_with_localhost_default(self, *args, hostname_localhost=None, **kwargs):
    if hostname_localhost is None:
        hostname_localhost = True
    _original_lammps_library_init(self, *args, hostname_localhost=hostname_localhost, **kwargs)


pylammpsmpi.LammpsLibrary.__init__ = _lammps_library_init_with_localhost_default

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
            from calphy import Solid, Liquid, Alchemy
            from calphy.routines import routine_fe, routine_ts, routine_composition_scaling

            # mode dictates the job class, matching calphy's own dispatch in
            # calphy/queuekernel.py: "alchemy"/"composition_scaling" always run
            # through Alchemy (it owns the switching + MC-swap + mass_integration
            # machinery), regardless of reference_phase.
            if input_class.mode == "composition_scaling":
                if lmp is not None:
                    job = Alchemy(calculation=input_class, simfolder=working_directory, lmp=lmp)
                else:
                    job = Alchemy(calculation=input_class, simfolder=working_directory)
            elif input_class.reference_phase == "solid":
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


def _target_energy_at_end_of_forward_switching(working_directory: str, n_last: int = 100) -> float:
    """Potential energy per atom (eV/atom) of the target composition at the end of the forward switching.

    Read from ``forward_<i>.dat`` of a composition_scaling job (columns dU_1, dU_2, lambda): at lambda = 0 only the
    potential of the target composition is left, so dU_2 over the last ``n_last`` rows is its energy per atom.
    Averaged over the iterations; NaN if the files are not there. A target that melted shows as a jump of this
    energy against its neighbours in concentration or temperature.
    """
    energies = []
    iteration = 1
    while os.path.exists(path := os.path.join(working_directory, f"forward_{iteration}.dat")):
        energies.append(float(np.loadtxt(path, comments="#", usecols=1)[-n_last:].mean()))
        iteration += 1
    return float(np.mean(energies)) if energies else np.nan


def gather_calphy_results_detailed(working_directory: str) -> pd.DataFrame:
    """Read calphy results from a single calculation directory.

    Independent of calphy's gather_results. Reads temperature sweep data
    and forward/backward energy differences for phase-transition checking.
    The three modes give different columns, see Returns.

    Parameters
    ----------
    working_directory : str
        Path to a single calphy calculation directory (contains input_file.yaml)

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        - calculation: folder name
        - calculation_mode: 'ts', 'fe' or 'composition_scaling'
        - reference_phase: 'solid' or 'liquid'
        - pressure: float
        - status: bool
        - composition: dict mapping element to concentration (the target composition for composition_scaling)
        - temperature: array (ts) or scalar (fe, composition_scaling)
        - free_energy: array (ts) or scalar (fe); for composition_scaling see below
        - free_energy_error: array (ts), 0.0 (fe), or the error calphy reports (composition_scaling; 0.0 with one iteration)
        - forward_energy_diff: list of arrays, one per iteration (ts only)
        - backward_energy_diff: list of arrays, one per iteration (ts only)
        - forward_lambda: list of arrays, one per iteration (ts only)
        - backward_lambda: list of arrays, one per iteration (ts only)
        - dissipation: eV/atom, scalar (fe, composition_scaling): half the sum of the forward and backward work; about
          zero when the switching is reversible, large when the phase melts or changes along the way

        Additional columns of ``composition_scaling``, all in eV/atom unless noted. A composition_scaling job gives only
        the free energy of the transformation FROM the reference composition (``reference_composition``), not an
        absolute free energy; the reference comes from a separate fe (or ts) calculation and is added with
        ``add_reference_free_energy``.

        - reference_composition: c(Mg) of the starting structure
        - delta_free_energy: free-energy difference to the reference composition, work + mass correction, WITHOUT the
          ideal-mixing term
        - work, mass_correction: the two parts of delta_free_energy (delta = work - mass_correction)
        - ideal_entropy: eV/atom/K, positive; calphy's convention (calphy.postprocessing), minus the
          ``entropy_contribution`` of calphy's report. The ideal-mixing term of the transformation is
          ``-temperature * ideal_entropy`` (it lowers the free energy), as in calphy's ``fix_composition_scaling``
        - delta_free_energy_with_ideal_entropy: delta_free_energy - temperature * ideal_entropy
        - energy_target: potential energy per atom of the target composition at the end of the forward switching
          (average over the iterations; NaN if forward_<i>.dat is missing). A melted target shows as a jump of it
          against its neighbours in concentration or temperature
        - reference_free_energy, free_energy, free_energy_with_ideal_entropy: NaN here; filled by
          ``add_reference_free_energy`` (reference + delta, without and with the ideal-mixing term)

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

    elif mode == "composition_scaling":
        # the job gives only the free energy of the transformation from the reference composition;
        # the free energy of the reference itself comes from a separate fe job, see add_reference_free_energy
        results = rep["results"] if rep else {}
        data["temperature"] = inp["temperature"]
        data["reference_composition"] = inp["reference_composition"]
        data["delta_free_energy"] = results.get("free_energy", np.nan)    # work + mass correction
        data["work"] = results.get("work", np.nan)
        data["mass_correction"] = results.get("mass_correction", np.nan)
        data["dissipation"] = results.get("dissipation", np.nan)    # eV/atom, half the forward + backward work: ~0 when reversible
        data["energy_target"] = _target_energy_at_end_of_forward_switching(wd)   # eV/atom of the target composition at the end of the forward switching
        # calphy's convention (calphy.postprocessing): ideal_entropy = -entropy_contribution of the report, positive, eV/atom/K;
        # the ideal-mixing term enters as F - T * ideal_entropy (calphy's fix_composition_scaling)
        data["ideal_entropy"] = -results.get("entropy_contribution", np.nan)
        data["delta_free_energy_with_ideal_entropy"] = data["delta_free_energy"] - inp["temperature"] * data["ideal_entropy"]
        data["reference_free_energy"] = np.nan
        data["free_energy"] = np.nan                       # reference + delta, without the ideal-entropy term
        data["free_energy_with_ideal_entropy"] = np.nan    # free_energy - temperature * ideal_entropy
        data["free_energy_error"] = results.get("error", np.nan)
        data["forward_energy_diff"] = None
        data["backward_energy_diff"] = None
        data["forward_lambda"] = None
        data["backward_lambda"] = None

    else:  # fe mode
        data["temperature"] = inp["temperature"]
        data["free_energy"] = rep["results"]["free_energy"] if rep else np.nan
        data["free_energy_error"] = 0.0 if rep else np.nan
        data["dissipation"] = rep["results"].get("dissipation", np.nan) if rep else np.nan
        data["forward_energy_diff"] = None
        data["backward_energy_diff"] = None
        data["forward_lambda"] = None
        data["backward_lambda"] = None

    return pd.DataFrame([data])


def _same_value(a: Any, b: Any) -> bool:
    return bool(np.allclose(np.atleast_1d(a).astype(float), np.atleast_1d(b).astype(float)))


def add_reference_free_energy(composition_scaling_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Fill in the free energy of the reference composition for composition_scaling results.

    A composition_scaling job gives only the free energy of the transformation from the reference
    composition (``delta_free_energy``). The free energy of the reference itself comes from an ``fe`` job
    at the same temperature, pressure and reference phase, read with gather_calphy_results_detailed.

    Parameters
    ----------
    composition_scaling_df : pd.DataFrame
        Rows of composition_scaling jobs from gather_calphy_results_detailed
    reference_df : pd.DataFrame
        Rows of the fe jobs of the reference composition from gather_calphy_results_detailed

    Returns
    -------
    pd.DataFrame
        A copy of composition_scaling_df with ``reference_free_energy``, ``free_energy`` (reference +
        ``delta_free_energy``, without the ideal-mixing term) and ``free_energy_with_ideal_entropy`` (reference +
        ``delta_free_energy_with_ideal_entropy``)

    Raises
    ------
    ValueError
        If a row has no fe job, or more than one, with its temperature, pressure and reference phase
    """
    result = composition_scaling_df.copy()
    fe_jobs = reference_df[(reference_df["calculation_mode"] == "fe") & reference_df["status"]]
    for index, row in result.iterrows():
        if row["calculation_mode"] != "composition_scaling":
            raise ValueError(f"row {index} is a '{row['calculation_mode']}' job, not composition_scaling")
        match = fe_jobs[
            fe_jobs.apply(
                lambda ref: _same_value(ref["temperature"], row["temperature"])
                and _same_value(ref["pressure"], row["pressure"])
                and ref["reference_phase"] == row["reference_phase"],
                axis=1,
            )
        ]
        if len(match) != 1:
            raise ValueError(
                f"expected exactly one finished fe job at T={row['temperature']}, pressure={row['pressure']}, "
                f"reference phase {row['reference_phase']} for row {index}, found {len(match)}"
            )
        reference_free_energy = float(match.iloc[0]["free_energy"])
        result.at[index, "reference_free_energy"] = reference_free_energy
        result.at[index, "free_energy"] = reference_free_energy + row["delta_free_energy"]
        result.at[index, "free_energy_with_ideal_entropy"] = reference_free_energy + row["delta_free_energy_with_ideal_entropy"]
    return result


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