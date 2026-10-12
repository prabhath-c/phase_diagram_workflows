"""Unit tests for defect_energies.defect_spec.defect_from_orbit."""

import numpy as np
import pytest
from ase.build import bulk

from phase_diagram_workflows.defect_energies import defect_from_orbit
from phase_diagram_workflows.structures.point_defects import (
    discover_atomic_sublattices,
    discover_interstitial_sublattices,
)

unit_cell = bulk("Al", cubic=True)
atomic = discover_atomic_sublattices(unit_cell)[0]
void = next(o for o in discover_interstitial_sublattices(unit_cell) if o["label"] == "int_8c")


def test_vacancy():
    assert defect_from_orbit(atomic, "vacancy") == {"type": "vacancy", "atom_index": atomic["atom_indices"][0], "label": "Al_vac_4a"}


def test_substitution():
    spec = defect_from_orbit(atomic, "substitution", element="Mg")
    assert spec["type"] == "substitution" and spec["to_element"] == "Mg" and spec["label"] == "Mg_on_Al_4a"


def test_interstitial_takes_the_first_void_position():
    spec = defect_from_orbit(void, "interstitial", element="Mg")
    assert spec["type"] == "interstitial" and spec["label"] == "Mg_i_int_8c"
    assert np.allclose(spec["position"], void["cart_positions"][0])


def test_site_with_a_suffix_is_kept_whole():
    assert defect_from_orbit({"label": "Al_18f_3", "species": "Al", "atom_indices": [5]}, "vacancy")["label"] == "Al_vac_18f_3"


@pytest.mark.parametrize("kwargs", [
    dict(orbit=atomic, kind="dislocation"),
    dict(orbit=void, kind="vacancy"),                      # a void orbit is not an atomic one
    dict(orbit=atomic, kind="interstitial", element="Mg"),
    dict(orbit=atomic, kind="substitution"),               # no element
    dict(orbit=atomic, kind="substitution", element="Al"), # Al by Al is no defect
    dict(orbit=void, kind="interstitial"),                 # no element
])
def test_invalid_requests_raise(kwargs):
    with pytest.raises(ValueError):
        defect_from_orbit(**kwargs)
