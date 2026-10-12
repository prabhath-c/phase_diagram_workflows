"""Unit tests for phase_diagram.point_defect_phase_diagram on synthetic host pickles and defect records (no real data)."""

import os

import numpy as np
import pandas as pd
import pytest

from phase_diagram_workflows.utils.executor_spec import ExecutorSpec
from phase_diagram_workflows.defect_energies.formation_energies import save_defect_results
from phase_diagram_workflows.phase_diagram.point_defect_phase_diagram import (
    add_point_defects,
    build_defected_phase,
    build_interpolated_phase,
    check_records,
    SUBMITTED_SUFFIX,
    compute_phase_diagram,
    load_line_phase,
    sublattices_from_records,
    temperature_range,
)


def _save(records, folder, name):
    os.makedirs(folder, exist_ok=True)
    save_defect_results(records, str(folder), name)


def _record(label, kind, site, species, dn_mg, energy, multiplicity, n_atoms):
    return {
        "defect_label": label, "defect_type": kind, "species": species, "delta_n_Mg": dn_mg, "intercept_landau": energy,
        "sublattice_ref": None if kind == "interstitial" else site, "sublattice_int": site if kind == "interstitial" else None,
        "multiplicity": multiplicity, "n_atoms": n_atoms, "intercept": energy, "defect_latex": label, "delta_n_Al": 0, "delta_n_total": 0,
    }


def _host_pickle(path, mg_fraction, t_min=300.0, t_max=900.0, n=1000):
    temperature = np.linspace(t_min, t_max, n)
    table = pd.DataFrame({"temperature": temperature, "free_energy": -3.0 - 1e-3 * (temperature - 300) - 0.1 * mg_fraction, "free_energy_error": 0.0})
    table.attrs = {"composition": {"Al": 1 - mg_fraction, "Mg": mg_fraction}}
    table.to_pickle(path)


@pytest.fixture
def inputs(tmp_path):
    hosts = {}
    for phase, x in (("FCC", 0.0), ("HCP", 1.0)):
        hosts[phase] = str(tmp_path / f"{phase}_free_energy.pkl")
        _host_pickle(hosts[phase], x, t_max=900.0 if phase == "FCC" else 700.0)
        folder = tmp_path / "defects" / phase
        other, mine = ("Mg", "Al") if phase == "FCC" else ("Al", "Mg")
        # vacancy and substitutional of one site, computed in supercells of different size, plus two interstitial sites
        _save([_record(f"{mine}_vac_4a", "vacancy", f"{mine}_4a", mine, 0, 0.7, 256, 256 * 1)], folder / "Vacancy", "vacancy")
        _save([_record(f"{other}_on_{mine}_4a", "substitutional", f"{mine}_4a", other, 1, 0.3, 500, 500)], folder / "Substitutional", "substitutional")
        _save(
            [_record("i_4b", "interstitial", "int_4b", mine, 0, 2.0, 4, 4), _record("i_8c", "interstitial", "int_8c", mine, 0, 2.5, 8, 4 * 2)],
            folder / "Interstitial", "interstitial",
        )
    return hosts, str(tmp_path / "defects")


def test_load_line_phase_takes_concentration_from_attrs_and_thins(inputs):
    hosts, _ = inputs
    phase = load_line_phase(hosts["HCP"], "HCP", n_points=50)
    assert phase.fixed_concentration == 1.0
    assert len(phase.temperatures) == 50
    assert phase.temperatures[0] == 300.0 and phase.temperatures[-1] == 700.0


def test_load_line_phase_without_composition_raises(tmp_path):
    path = str(tmp_path / "bare.pkl")
    pd.DataFrame({"temperature": [300.0, 400.0], "free_energy": [-1.0, -1.1], "free_energy_error": 0.0}).to_pickle(path)
    with pytest.raises(ValueError, match="composition"):
        load_line_phase(path, "X")


def test_sublattices_group_vacancy_and_substitutional_of_one_site(inputs):
    from phase_diagram_workflows.defect_energies.formation_energies import load_phase_defect_results

    _, defects = inputs
    sublattices = sublattices_from_records(load_phase_defect_results(f"{defects}/FCC"))
    by_name = {s.name: s for s in sublattices}
    assert set(by_name) == {"Al_4a", "int_4b", "int_8c"}
    site = by_name["Al_4a"]
    assert len(site.defects) == 2
    assert site.sublattice_fraction == pytest.approx(1.0)
    assert {d.excess_solutes for d in site.defects} == {0, 1}


def test_sublattices_disagreeing_fractions_raise():
    records = [_record("a", "vacancy", "s", "Al", 0, 0.5, 4, 4), _record("b", "substitutional", "s", "Mg", 1, 0.3, 3, 4)]
    with pytest.raises(ValueError, match="disagree"):
        sublattices_from_records(records)


def _liquid_pickles(tmp_path, concentrations, t_min=700.0, t_max=1300.0):
    paths = []
    for c in concentrations:
        paths.append(str(tmp_path / f"liquid_{c:.2f}.pkl"))
        _host_pickle(paths[-1], c, t_min, t_max)
    return paths


def test_interpolated_phase_rejects_duplicate_concentrations_and_empty(tmp_path):
    with pytest.raises(ValueError, match="same composition"):
        build_interpolated_phase(_liquid_pickles(tmp_path, [0.5, 0.5]), "Liquid")
    with pytest.raises(ValueError, match="no pickles"):
        build_interpolated_phase([], "Liquid")


def test_compute_phase_diagram_reads_cache(tmp_path):
    cache = str(tmp_path / "cache" / "df.pkl")
    table = pd.DataFrame({"T": [300.0], "phase": ["FCC"]})
    os.makedirs(os.path.dirname(cache))
    table.to_pickle(cache)
    assert compute_phase_diagram([], [300.0], cache).equals(table)


def test_compute_phase_diagram_runs_and_caches(inputs, tmp_path):
    hosts, defects = inputs
    phases = [build_defected_phase(hosts[p], f"{defects}/{p}", p, n_points=20) for p in hosts]
    cache = str(tmp_path / "df.pkl")
    table = compute_phase_diagram(phases, np.linspace(300, 700, 3), cache, n_mu=20)
    assert set(table["phase"]) <= {"FCC", "HCP"} and len(table) > 0
    assert pd.read_pickle(cache).equals(table)

def test_temperature_range_of_one_and_of_several(inputs, tmp_path):
    hosts, _ = inputs
    assert temperature_range(hosts["HCP"]) == (300.0, 700.0)
    assert temperature_range(_liquid_pickles(tmp_path, [0.0, 1.0], t_min=700.0, t_max=1300.0)) == (700.0, 1300.0)


def test_build_defected_phase_has_host_and_sublattices(inputs):
    hosts, defects = inputs
    phase = build_defected_phase(hosts["FCC"], f"{defects}/FCC", "fcc", n_points=30)
    assert phase.name == "fcc" and phase.line_phase.fixed_concentration == 0.0
    assert sorted(s.name for s in phase.sublattices) == ["Al_4a", "int_4b", "int_8c"]


def test_build_defected_phase_missing_files_raise(inputs, tmp_path):
    hosts, defects = inputs
    with pytest.raises(FileNotFoundError, match="not found"):
        build_defected_phase(str(tmp_path / "gone.pkl"), f"{defects}/FCC", "x")
    with pytest.raises(FileNotFoundError):
        build_defected_phase(hosts["FCC"], str(tmp_path / "no_defects"), "x")


def test_check_records_lists_every_incomplete_record():
    good = _record("a", "vacancy", "s", "Al", 0, 0.5, 4, 4)
    bad_one = {k: v for k, v in good.items() if k != "n_atoms"} | {"defect_label": "bad1"}
    bad_two = {k: v for k, v in good.items() if k not in ("multiplicity", "delta_n_Mg")} | {"defect_label": "bad2"}
    check_records([good])
    with pytest.raises(ValueError) as error:
        check_records([good, bad_one, bad_two])
    assert "bad1: missing ['n_atoms']" in str(error.value) and "bad2" in str(error.value) and "a:" not in str(error.value)


def test_phase_diagram_with_interpolated_phase(inputs, tmp_path):
    hosts, defects = inputs
    phases = [build_defected_phase(hosts[p], f"{defects}/{p}", p, n_points=20) for p in hosts]
    liquid = build_interpolated_phase(_liquid_pickles(tmp_path, [1.0, 0.0, 0.5, 0.25, 0.75]), "Liquid", n_points=20)
    assert [line.fixed_concentration for line in liquid.phases] == [0.0, 0.25, 0.5, 0.75, 1.0]
    table = compute_phase_diagram([*phases, liquid], np.linspace(700, 900, 2), str(tmp_path / "df.pkl"), n_mu=20)
    assert "Liquid" in set(table["phase"])


class _RecordingExecutor:
    """Runs a submitted call at once, here, and records it (stands in for a SLURM executor)."""

    calls = []

    def submit(self, function, *args, **kwargs):
        _RecordingExecutor.calls.append(function.__name__)
        function(*args, **kwargs)

    def shutdown(self, wait=True):
        pass


def test_compute_phase_diagram_submits_once_and_then_reads(inputs, tmp_path):
    hosts, defects = inputs
    phases = [build_defected_phase(hosts[p], f"{defects}/{p}", p, n_points=20) for p in hosts]
    cache = str(tmp_path / "out" / "df.pkl")
    spec = ExecutorSpec(_RecordingExecutor)
    _RecordingExecutor.calls.clear()

    assert compute_phase_diagram(phases, np.linspace(300, 700, 2), cache, n_mu=20, executor_spec=spec) is None   # submitted, not waited for
    assert _RecordingExecutor.calls == ["save_phase_diagram"] and os.path.isfile(cache + SUBMITTED_SUFFIX)
    assert os.path.isfile(cache) and not os.path.exists(cache + ".unfinished")                                 # the job saved it itself

    table = compute_phase_diagram(phases, np.linspace(300, 700, 2), cache, n_mu=20, executor_spec=spec)         # rerun: reads, no new job
    assert len(table) > 0 and _RecordingExecutor.calls == ["save_phase_diagram"]


def test_compute_phase_diagram_does_not_resubmit_while_pending(inputs, tmp_path):
    hosts, defects = inputs
    phases = [build_defected_phase(hosts["FCC"], f"{defects}/FCC", "FCC", n_points=20)]
    cache = str(tmp_path / "df.pkl")
    open(cache + SUBMITTED_SUFFIX, "w").close()                       # a job was submitted and has not finished
    _RecordingExecutor.calls.clear()
    assert compute_phase_diagram(phases, [300.0], cache, executor_spec=ExecutorSpec(_RecordingExecutor)) is None
    assert _RecordingExecutor.calls == [] and not os.path.exists(cache)


def test_add_point_defects_on_a_host_from_any_source(inputs):
    from landau.phases import TemperatureDependentLinePhase

    _, defects = inputs
    host = TemperatureDependentLinePhase(name="other", fixed_concentration=0.0, temperatures=[500.0, 600.0, 700.0], free_energies=[-3.0, -3.1, -3.2])
    phase = add_point_defects(host, f"{defects}/FCC", "fcc")
    assert phase.line_phase is host and phase.name == "fcc"
    assert sorted(s.name for s in phase.sublattices) == ["Al_4a", "int_4b", "int_8c"]
