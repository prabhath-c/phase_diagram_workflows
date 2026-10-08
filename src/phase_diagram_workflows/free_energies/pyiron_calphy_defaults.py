"""The calphy parameters that pyiron_atomistics' ``Calphy`` job uses, as a plain dict.

calphy's own API defaults differ from the values pyiron_atomistics supplies (``Calphy._default_input`` and the
``calc_mode_*`` methods in ``pyiron_atomistics/calphy/job.py``), e.g. a different thermostat damping, spring-constant
and pressure tolerance, and ``solid_fraction``. To reproduce a pyiron run, or to keep results comparable with
earlier ones, start from these values and add what you need::

    from phase_diagram_workflows.free_energies.pyiron_calphy_defaults import pyiron_atomistics_calphy_parameters

    parameters = pyiron_atomistics_calphy_parameters()
    parameters["mode"] = "ts"
    parameters["temperature"] = [300, 1000]
    parameters["reference_phase"] = "solid"
    parameters["md"]["seed"] = 1
"""

from __future__ import annotations

import copy
from typing import Any, Dict

# Mode-independent values of pyiron_atomistics (job.py, `_default_input`), with pressure 0 and npt as for a pyiron
# run that is given a pressure. Not included, because they are set per calculation: mode, temperature and
# reference_phase. n_print_steps is 0 as in every ``calc_mode_*`` of pyiron (``_default_input`` itself lists 1000,
# which no mode uses).
_PYIRON_ATOMISTICS_CALPHY_PARAMETERS: Dict[str, Any] = {
    "pressure": 0,
    "npt": True,
    "n_equilibration_steps": 15000,
    "n_switching_steps": 25000,
    "n_print_steps": 0,
    "n_iterations": 1,
    "spring_constants": None,
    "equilibration_control": "nose-hoover",
    "melting_cycle": True,
    "fix_potential_path": False,
    "md": {
        "timestep": 0.001,
        "n_small_steps": 10000,
        "n_every_steps": 10,
        "n_repeat_steps": 10,
        "n_cycles": 100,
        "thermostat_damping": 0.5,
        "barostat_damping": 0.1,
    },
    "tolerance": {
        "lattice_constant": 0.0002,
        "spring_constant": 0.01,
        "solid_fraction": 0.7,
        "liquid_fraction": 0.05,
        "pressure": 0.5,
    },
    "nose_hoover": {"thermostat_damping": 0.1, "barostat_damping": 0.1},
    "berendsen": {"thermostat_damping": 100.0, "barostat_damping": 100.0},
}


def pyiron_atomistics_calphy_parameters() -> Dict[str, Any]:
    """The pyiron_atomistics calphy parameters, as a new dict that is yours to change.

    Every call returns an independent deep copy, so editing it (also a nested entry such as
    ``parameters["md"]["seed"] = 1``) never affects another call or the defaults themselves.

    For a constant-volume run set ``parameters["npt"] = False`` and delete ``parameters["pressure"]``.

    Returns
    -------
    Dict[str, Any]
        Without ``mode``, ``temperature`` and ``reference_phase`` (calphy needs them: add them), the structure, the
        potential, or anything about how calphy is run (``execution_mode``, ``queue``, ``file_format``).
    """
    return copy.deepcopy(_PYIRON_ATOMISTICS_CALPHY_PARAMETERS)
