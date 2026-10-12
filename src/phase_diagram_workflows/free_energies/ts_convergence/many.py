"""Temperature-scaling bracket convergence for MANY structures of one phase, with one tolerance for all.

``single.py`` narrows the bracket of one structure against a tolerance you give it. For a phase there is one
structure per concentration, and the right tolerance is visible in the first bracket of all of them: the
criteria fall into a low group (the crystal survived the heating) and a high group (it did not), see
``discover_tolerance``. ``refine_many_structures_with_discovered_tolerance`` therefore

1. runs the first bracket of every structure,
2. once all of them are done, finds the tolerance from their criteria **once** and stores it in
   ``tolerance.json`` in the phase folder, and
3. narrows every structure that is above it, exactly as ``refine_temperature_bracket_with_resubmission`` does.

The tolerance is not recomputed for the narrower brackets: most of those are good, so their criteria have no
gap, and a tolerance that moved would change the verdict on brackets that were already judged.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase_diagram_workflows.free_energies.ts_convergence.base import (
    discover_tolerance,
    pick_best_converged_bracket,
)
from phase_diagram_workflows.free_energies.ts_convergence.single import (
    DEFAULT_MAX_ITERATIONS,
    ExecutorSpec,
    _check_max_iterations,
    current_bracket_resource_dict,
    load_bracket_history,
    refine_temperature_bracket_with_resubmission,
)

TOLERANCE_FILE = "tolerance.json"

# the job that waits for the first brackets does no calculation: it reads a few files and submits jobs
_DEFAULT_COORDINATOR_RESOURCES = {"cores": 1, "threads_per_core": 1, "run_time_limit": 3600}


def _tolerance_path(working_directory: str) -> str:
    return os.path.join(working_directory, TOLERANCE_FILE)


def load_discovered_tolerance(working_directory: str) -> Optional[float]:
    """The tolerance stored for this phase folder, or None if it has not been determined yet."""
    try:
        with open(_tolerance_path(working_directory)) as handle:
            return float(json.load(handle)["tolerance"])
    except FileNotFoundError:
        return None


def _first_bracket_criteria(
    structure_roots: Dict[str, str], initial_bracket: Tuple[float, float]
) -> Dict[str, Optional[float]]:
    return {
        name: load_bracket_history(root)[1].get(initial_bracket) for name, root in structure_roots.items()
    }


def _submit_narrowing(
    structures: Dict[str, Any],
    structure_roots: Dict[str, str],
    tolerance: float,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor_spec: ExecutorSpec,
    initial_bracket: Tuple[float, float],
    step_lower: Optional[float],
    step_upper: Optional[float],
    scale_switching_steps_with_range: bool,
    max_iterations: Optional[int],
) -> List[str]:
    """Submit `refine_temperature_bracket_with_resubmission` for every structure that has not converged yet."""
    executor = executor_spec.create()
    submitted, futures = [], []
    for name, atoms in structures.items():
        root = structure_roots[name]
        if pick_best_converged_bracket(load_bracket_history(root)[1], tolerance) is not None:
            continue
        futures.append(executor.submit(
            refine_temperature_bracket_with_resubmission,
            resource_dict=current_bracket_resource_dict(root, initial_bracket, executor_spec),
            input_structure=atoms,
            calphy_parameters=calphy_parameters,
            potential_df=potential_df,
            executor_spec=executor_spec,
            working_directory_root=root,
            initial_bracket=initial_bracket,
            tolerance=tolerance,
            step_lower=step_lower,
            step_upper=step_upper,
            scale_switching_steps_with_range=scale_switching_steps_with_range,
            max_iterations=max_iterations,
        ))
        submitted.append(name)
    if executor_spec.detaches:
        executor.shutdown(wait=False)
    else:
        for future in futures:
            future.result()
        executor.shutdown(wait=True)
    return submitted


def _discover_tolerance_and_narrow(
    structures: Dict[str, Any],
    working_directory: str,
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor_spec: ExecutorSpec,
    initial_bracket: Tuple[float, float],
    step_lower: Optional[float],
    step_upper: Optional[float],
    scale_switching_steps_with_range: bool,
    max_iterations: Optional[int],
    min_gap_ratio: float,
    noise_floor: float,
    fallback_tolerance: Optional[float],
    **first_bracket_results: Any,
) -> Dict[str, Any]:
    """The job that runs once every first bracket has finished (they arrive as `first_bracket_results`).

    Never recomputes: if `tolerance.json` exists, it is used as it is.
    """
    structure_roots = {name: os.path.join(working_directory, name) for name in structures}

    tolerance = load_discovered_tolerance(working_directory)
    if tolerance is None:
        criteria = _first_bracket_criteria(structure_roots, initial_bracket)
        missing = [name for name, criterion in criteria.items() if criterion is None]
        if missing:
            raise RuntimeError(f"The first bracket has no result for {missing}; nothing was decided.")
        try:
            tolerance, method = discover_tolerance(list(criteria.values()), min_gap_ratio, noise_floor), "gap"
        except ValueError as no_gap:
            if fallback_tolerance is None:
                raise ValueError(
                    f"{no_gap} Give a fallback_tolerance to use when the first brackets show no failing group."
                ) from no_gap
            tolerance, method = float(fallback_tolerance), "fallback (no gap in the first brackets)"
        temporary = _tolerance_path(working_directory) + ".tmp"
        with open(temporary, "w") as handle:
            json.dump({"tolerance": tolerance, "method": method, "initial_bracket": list(initial_bracket),
                       "first_bracket_criteria": criteria}, handle, indent=1)
        os.replace(temporary, _tolerance_path(working_directory))

    submitted = _submit_narrowing(
        structures, structure_roots, tolerance, calphy_parameters, potential_df, executor_spec,
        initial_bracket, step_lower, step_upper, scale_switching_steps_with_range, max_iterations,
    )
    return {"tolerance": tolerance, "narrowing_submitted": submitted}


def refine_many_structures_with_discovered_tolerance(
    structures: Dict[str, Any],
    calphy_parameters: Dict[str, Any],
    potential_df: Any,
    executor_spec: ExecutorSpec,
    working_directory: str,
    initial_bracket: Tuple[float, float],
    step_lower: Optional[float] = None,
    step_upper: Optional[float] = None,
    scale_switching_steps_with_range: bool = True,
    max_iterations: Optional[int] = DEFAULT_MAX_ITERATIONS,
    fallback_tolerance: Optional[float] = None,
    min_gap_ratio: float = 3.0,
    noise_floor: float = 2e-4,
    coordinator_resource_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Submit the temperature-bracket convergence of every structure of a phase, with a tolerance found from the data.

    Call it once and close everything. First, the ``initial_bracket`` of every structure is submitted as a job.
    A further small job that starts when all of those have finished computes the tolerance from their criteria
    (``discover_tolerance``), writes it to ``<working_directory>/tolerance.json`` and submits the narrowing of
    every structure that is above it; from there each bracket is a job that submits the next itself
    (``refine_temperature_bracket_with_resubmission``). The tolerance is determined once and never recomputed.

    The state lives on disk, so calling this again resumes: structures whose first bracket has a result are not
    run again, and once ``tolerance.json`` exists only the narrowing is submitted. Delete ``tolerance.json`` to
    have the tolerance determined again. Only re-run when nothing is queued or running any more, a job that is
    still waiting leaves no trace on disk.

    Parameters
    ----------
    structures : Dict[str, Any]
        Structure (e.g. ase.Atoms) by name; each name is the sub-folder of `working_directory` that
        structure lives in.
    calphy_parameters, potential_df, executor_spec, initial_bracket, step_lower, step_upper,
    scale_switching_steps_with_range, max_iterations
        As in ``refine_temperature_bracket_with_resubmission``. `max_iterations` counts the narrowing steps
        after the first bracket. The executor needs to resolve Future arguments (a cluster executor does) and
        its queue template must render the dependency.
    working_directory : str
        Folder of the phase; ``tolerance.json`` and one folder per structure live here.
    fallback_tolerance : Optional[float], optional
        Used when the first brackets show no failing group, e.g. a phase that survives the whole initial range
        (every criterion is then noise and any value above the noise will do). Without it, that case raises
        in the job that decides.
    min_gap_ratio, noise_floor : float, optional
        See ``discover_tolerance``.
    coordinator_resource_dict : Optional[Dict[str, Any]], optional
        Resources of the job that waits for the first brackets (default: one core, one hour).

    Returns
    -------
    Dict[str, Any]
        ``tolerance``: the stored tolerance, or None while it has not been determined; ``first_brackets_submitted``
        and ``narrowing_submitted``: names of the structures that were submitted.
        With an executor that does not detach (e.g. ``SingleNodeExecutor``) this returns when everything is done.
    """
    if step_lower is None and step_upper is None:
        raise ValueError("At least one of step_lower or step_upper must be given.")
    _check_max_iterations(max_iterations)

    structure_roots = {name: os.path.join(working_directory, name) for name in structures}
    narrowing = (
        calphy_parameters, potential_df, executor_spec, initial_bracket,
        step_lower, step_upper, scale_switching_steps_with_range, max_iterations,
    )

    tolerance = load_discovered_tolerance(working_directory)
    if tolerance is not None:
        submitted = _submit_narrowing(structures, structure_roots, tolerance, *narrowing)
        return {"tolerance": tolerance, "first_brackets_submitted": [], "narrowing_submitted": submitted}

    executor = executor_spec.create()
    first_bracket_futures = {}
    for name, atoms in structures.items():
        root = structure_roots[name]
        if initial_bracket in load_bracket_history(root)[1]:
            continue
        first_bracket_futures[f"first_bracket_{name}"] = executor.submit(
            refine_temperature_bracket_with_resubmission,
            resource_dict=current_bracket_resource_dict(root, initial_bracket, executor_spec),
            input_structure=atoms,
            calphy_parameters=calphy_parameters,
            potential_df=potential_df,
            executor_spec=executor_spec,
            working_directory_root=root,
            initial_bracket=initial_bracket,
            tolerance=np.inf,          # accept whatever comes out: only this first bracket is wanted here
            step_lower=step_lower,
            step_upper=step_upper,
            scale_switching_steps_with_range=scale_switching_steps_with_range,
            max_iterations=0,
        )

    task_name = "discover_tolerance_" + hashlib.sha1(os.path.abspath(working_directory).encode()).hexdigest()[:12]
    cache_directory = os.path.abspath(executor_spec.kwargs.get("cache_directory", "executorlib_cache"))
    stale_result = os.path.join(cache_directory, task_name + "_o.h5")
    if os.path.exists(stale_result):
        os.remove(stale_result)   # left by an earlier attempt (e.g. one that raised): the tolerance file, not the cache, is the state

    coordinator = executor.submit(
        _discover_tolerance_and_narrow,
        resource_dict={**(coordinator_resource_dict or _DEFAULT_COORDINATOR_RESOURCES), "cache_key": task_name},
        structures=structures,
        working_directory=working_directory,
        calphy_parameters=calphy_parameters,
        potential_df=potential_df,
        executor_spec=executor_spec,
        initial_bracket=initial_bracket,
        step_lower=step_lower,
        step_upper=step_upper,
        scale_switching_steps_with_range=scale_switching_steps_with_range,
        max_iterations=max_iterations,
        min_gap_ratio=min_gap_ratio,
        noise_floor=noise_floor,
        fallback_tolerance=fallback_tolerance,
        **first_bracket_futures,
    )
    if executor_spec.detaches:
        executor.shutdown(wait=False)
        decided = {}
    else:
        decided = coordinator.result()
        executor.shutdown(wait=True)
    return {
        "tolerance": decided.get("tolerance"),
        "first_brackets_submitted": [name[len("first_bracket_"):] for name in first_bracket_futures],
        "narrowing_submitted": decided.get("narrowing_submitted", []),
    }
