"""
Energies of pristine and defected structures: relaxation on a potential, and (building on it)
point-defect size convergence and formation energies.

- `relaxation`: relax many structures (cell and/or positions) with a LAMMPS potential, in the
  notebook or as fire-and-forget SLURM jobs, with the results kept on disk.
- `defect_spec`: describe a defect for the size convergence from a sublattice orbit.
- `jobs`: state of every job of a run (queued, running, done, failed) kept as files, and a
  `follow_jobs` that prints the changes so the submitting cell shows what is going on.
- `phase_inputs`: read the relaxed unit cell, the pure-element energies and the sublattices a phase's earlier steps saved.
- `formation_energies`: the formation energy of every defect of one type of a phase at one supercell size, and
  storing the records.
- `size_convergence`: formation energy of one point defect against supercell size (pristine and
  defect energy at every size), one job per size, results on disk; plus its table, plot and converged size.

Pure arithmetic on the energies (`compute_formation_energy`) lives in
`phase_diagram_workflows.structures.point_defects.formation_energy`.
"""

from .defect_spec import build_defect, defect_from_orbit
from .formation_energies import (
    calculate_formation_energies,
    load_defect_results,
    load_phase_defect_results,
    save_defect_results,
)
from .jobs import follow_jobs, load_job_status
from .phase_inputs import PhaseInputs, load_phase_inputs
from .relaxation import load_relaxed_structures, relax_structures
from .size_convergence import (
    converge_formation_energy,
    load_size_convergence,
    plot_size_convergence,
    select_converged_size,
    summarize_size_convergence,
)

__all__ = [
    "PhaseInputs",
    "build_defect",
    "calculate_formation_energies",
    "converge_formation_energy",
    "defect_from_orbit",
    "follow_jobs",
    "load_defect_results",
    "load_job_status",
    "load_phase_defect_results",
    "load_phase_inputs",
    "load_relaxed_structures",
    "load_size_convergence",
    "plot_size_convergence",
    "relax_structures",
    "save_defect_results",
    "select_converged_size",
    "summarize_size_convergence",
]
