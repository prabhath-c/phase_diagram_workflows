"""
Integration tests for phase_diagram_workflows.phase_diagram.convex_hull.

Real LAMMPS runs via atomistics -- no mocking. Complements
tests/unit/phase_diagram/test_convex_hull.py's fast mocked tests (which
check our own argument-forwarding contract in isolation) with genuine
end-to-end confidence that the wrappers actually work against a real
LAMMPS build.
"""

import os
import sys

import pytest
from ase.build import bulk

from phase_diagram_workflows.phase_diagram.convex_hull import (
    compute_energy_per_atom,
    optimize_and_compute_energy_per_atom,
    optimize_structure,
)


@pytest.fixture(scope="module")
def liu_al_mg_eam_potential_df():
    """The real 1998 Liu Al-Mg EAM potential, via lammpsparser + iprpy-data.

    Same potential used throughout the example notebooks and verified
    there against real Materials Project structures. Requires the
    iprpy-data conda package (see .ci_support/environment.yml); resource
    path is derived from sys.prefix rather than the CONDA_PREFIX env var,
    since the latter isn't reliably set in every process that runs
    pytest (e.g. a Jupyter kernel -- see notebooks/ConvexHull_MaterialsProject
    .ipynb's setup cell for the same fix).
    """
    from lammpsparser import get_potential_by_name

    resource_path = os.path.join(sys.prefix, "share", "iprpy")
    potential_df = get_potential_by_name("1998--Liu-X-Y--Al-Mg--LAMMPS--ipr1", resource_path=resource_path)
    potential_df = potential_df.to_frame().transpose()
    potential_df["Config"] = potential_df["Config"].apply(
        lambda cfg: [s if s.endswith("\n") else s + "\n" for s in cfg]
    )
    return potential_df


class TestOptimizeAndComputeEnergyPerAtomRealLammps:
    """-3.359966 eV/atom is the same value independently verified earlier
    against a real Materials Project pure-Al structure with this exact
    potential (see notebooks/ConvexHull_MaterialsProject.ipynb) --
    bulk("Al") should relax to the same cohesive energy, since both are
    just perfect FCC Al under the same potential.
    """

    def test_optimize_structure_relaxes_bulk_al(self, liu_al_mg_eam_potential_df):
        atoms = bulk("Al")

        relaxed = optimize_structure(atoms, liu_al_mg_eam_potential_df)

        assert len(relaxed) == len(atoms)
        assert relaxed.get_chemical_symbols() == atoms.get_chemical_symbols()

    def test_compute_energy_per_atom_matches_known_value(self, liu_al_mg_eam_potential_df):
        atoms = bulk("Al")

        epa = compute_energy_per_atom(atoms, liu_al_mg_eam_potential_df)

        assert epa == pytest.approx(-3.359966, abs=1e-3)

    def test_optimize_and_compute_energy_per_atom_end_to_end(self, liu_al_mg_eam_potential_df):
        atoms = bulk("Al")

        relaxed, epa = optimize_and_compute_energy_per_atom(atoms, liu_al_mg_eam_potential_df)

        assert len(relaxed) == len(atoms)
        assert epa == pytest.approx(-3.359966, abs=1e-3)
