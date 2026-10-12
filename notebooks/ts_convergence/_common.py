"""Settings shared by the ts_convergence notebooks, so every notebook runs the same small, fast Al case.

The numbers are chosen for speed (a bracket takes seconds), not for production-quality free energies.
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from ase.build import bulk
from lammpsparser import get_potential_by_name

PROJECT_ROOT = next(
    (parent for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents] if (parent / "pyproject.toml").is_file()),
    Path.cwd(),
)
RESULTS = Path(__file__).resolve().parent / "results"

INITIAL_BRACKET = (700.0, 720.0)


def setup_environment() -> None:
    """Make the checkout importable and the kernel work without an activated environment."""
    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    # lammpsparser looks for the potential library under CONDA_PREFIX, and pylammpsmpi starts LAMMPS through `mpiexec`
    # from the environment's bin directory: both are missing when the kernel was started without activating the environment
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    # the bracket jobs run in worker processes that inherit these: keep TensorFlow start-up notices and deprecation
    # warnings out of the notebook
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"


def results_directory(name: str) -> Path:
    """Folder for the results of one notebook; delete it to start that notebook from scratch."""
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def al_structure(lattice_constant: float = 4.05, repeat: int = 2):
    """Cubic fcc Al, ``4 * repeat**3`` atoms (32 by default)."""
    return bulk("Al", a=lattice_constant, cubic=True).repeat(repeat)


def mishin_al_potential() -> pd.DataFrame:
    return get_potential_by_name("1999--Mishin-Y--Al--LAMMPS--ipr1").to_frame().transpose()


def calphy_parameters() -> dict:
    """Reversible scaling ("ts" mode) on one core. ``temperature`` is overwritten for every bracket."""
    return {
        "mode": "ts",
        "pressure": 0,
        "temperature": list(INITIAL_BRACKET),
        "n_equilibration_steps": 100,
        "n_switching_steps": 100,
        "n_print_steps": 0,
        "equilibration_control": "berendsen",
        "md": {"thermostat_damping": 0.5},
        "tolerance": {"spring_constant": 0.01, "pressure": 0.5},
        "queue": {"cores": 1, "scheduler": "local"},   # MPI ranks of the LAMMPS run
        "execution_mode": "library",                    # LAMMPS through pylammpsmpi: no `lmp` binary needed
        "reference_phase": "solid",
        "file_format": "lammps-data",
    }
