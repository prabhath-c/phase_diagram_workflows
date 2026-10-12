"""
Unit tests for phase_diagram_workflows.structures.point_defects.formation_energy.

Expected values are hand-computed from the same formulas documented in
compute_formation_energy's docstring (and matching the "Formation energies
and save" cells in the source all_vacancies/all_substitutionals/
all_interstitials notebooks) rather than copied from its implementation.
"""

import pytest

from phase_diagram_workflows.structures.point_defects.formation_energy import (
    compute_formation_energy,
    delta_n_for_interstitial,
    delta_n_for_substitution,
    delta_n_for_vacancy,
)


class TestDeltaNForVacancy:
    def test_al_vacancy(self):
        assert delta_n_for_vacancy("Al") == {"Al": -1}

    def test_arbitrary_element(self):
        assert delta_n_for_vacancy("Ti") == {"Ti": -1}


class TestDeltaNForSubstitution:
    def test_al_to_mg(self):
        assert delta_n_for_substitution("Al", "Mg") == {"Al": -1, "Mg": 1}

    def test_mg_to_al(self):
        assert delta_n_for_substitution("Mg", "Al") == {"Mg": -1, "Al": 1}

    def test_same_element_is_a_no_op(self):
        assert delta_n_for_substitution("Al", "Al") == {}


class TestDeltaNForInterstitial:
    def test_al_interstitial(self):
        assert delta_n_for_interstitial("Al") == {"Al": 1}

    def test_arbitrary_element(self):
        assert delta_n_for_interstitial("Zn") == {"Zn": 1}


class TestComputeFormationEnergy:
    def test_vacancy_case(self):
        # Al vacancy (pinned=Al, swept=Mg): delta_n={"Al": -1} -- not a swap,
        # landau["exact"] is False.
        E_defect, E_pristine = -430.0, -432.0
        mu0 = {"Al": -3.5, "Mg": -1.6}
        result = compute_formation_energy(
            E_defect, E_pristine, {"Al": -1}, mu0, pinned_element="Al", swept_element="Mg"
        )

        assert result["delta_n_pinned"] == -1
        assert result["delta_n_swept"] == 0
        assert result["delta_n_total"] == -1
        assert result["slope"] == 0  # -delta_n_swept
        assert result["slope_pinned"] == 1  # -delta_n_pinned
        assert result["intercept"] == pytest.approx(E_defect - E_pristine - (-1) * mu0["Al"])
        assert result["landau"]["intercept"] == pytest.approx(E_defect - E_pristine - (-1) * mu0["Al"])
        # For this defect delta_n_total == delta_n_pinned, so intercept and
        # landau["intercept"] coincide numerically -- but landau["exact"] must
        # still read False, since that equality is a coincidence of this specific
        # defect (delta_n_swept=0), not a guarantee the two quantities mean the
        # same thing in general.
        assert result["landau"]["exact"] is False

    def test_substitution_case_is_landau_exact(self):
        # Al->Mg substitution: delta_n={"Al": -1, "Mg": 1} -- a swap, delta_n_total=0.
        E_defect, E_pristine = -428.0, -432.0
        mu0 = {"Al": -3.5, "Mg": -1.6}
        result = compute_formation_energy(
            E_defect, E_pristine, {"Al": -1, "Mg": 1}, mu0, pinned_element="Al", swept_element="Mg"
        )

        assert result["delta_n_total"] == 0
        assert result["landau"]["exact"] is True
        assert result["slope"] == -1
        assert result["slope_pinned"] == 1
        assert result["intercept"] == pytest.approx(E_defect - E_pristine - (-1) * mu0["Al"] - 1 * mu0["Mg"])
        # delta_n_total == 0 -- landau's mu_pinned term drops out entirely.
        assert result["landau"]["intercept"] == pytest.approx(E_defect - E_pristine)

    def test_interstitial_case(self):
        # Mg interstitial (pinned=Al, swept=Mg): delta_n={"Mg": 1} -- not a swap.
        E_defect, E_pristine = -429.0, -432.0
        mu0 = {"Al": -3.5, "Mg": -1.6}
        result = compute_formation_energy(
            E_defect, E_pristine, {"Mg": 1}, mu0, pinned_element="Al", swept_element="Mg"
        )

        assert result["delta_n_total"] == 1
        assert result["landau"]["exact"] is False
        assert result["slope"] == -1
        assert result["slope_pinned"] == 0
        assert result["intercept"] == pytest.approx(E_defect - E_pristine - 1 * mu0["Mg"])
        assert result["landau"]["intercept"] == pytest.approx(E_defect - E_pristine - 1 * mu0["Al"])

    def test_arbitrary_element_pair_not_hardcoded(self):
        # Same math, completely different element pair -- nothing here should
        # be special-cased to Al/Mg.
        E_defect, E_pristine = -900.0, -905.0
        mu0 = {"Ti": -7.8, "Zn": -1.3}
        result = compute_formation_energy(
            E_defect, E_pristine, {"Ti": -1}, mu0, pinned_element="Ti", swept_element="Zn"
        )
        assert result["pinned_element"] == "Ti"
        assert result["swept_element"] == "Zn"
        assert result["intercept"] == pytest.approx(E_defect - E_pristine - (-1) * mu0["Ti"])

    def test_record_carries_inputs_through(self):
        result = compute_formation_energy(
            -10.0, -12.0, {"Al": -1}, {"Al": -3.5, "Mg": -1.6}, pinned_element="Al", swept_element="Mg"
        )
        assert result["E_defect"] == -10.0
        assert result["delta_n"] == {"Al": -1}
        assert result["mu0_pinned"] == -3.5
        assert result["mu0_swept"] == -1.6

    def test_zero_defect_matches_pristine_gives_zero_formation_energy(self):
        # A no-op "defect" (same energy, no atoms changed) should cost nothing.
        result = compute_formation_energy(
            -100.0, -100.0, {}, {"Al": -3.5, "Mg": -1.6}, pinned_element="Al", swept_element="Mg"
        )
        assert result["intercept"] == pytest.approx(0.0)
        assert result["landau"]["intercept"] == pytest.approx(0.0)
        assert result["landau"]["exact"] is True  # delta_n_total == 0

    def test_rejects_delta_n_key_outside_the_pinned_swept_pair(self):
        with pytest.raises(ValueError, match="neither pinned_element"):
            compute_formation_energy(
                -10.0, -12.0, {"Cu": -1}, {"Al": -3.5, "Mg": -1.6, "Cu": -2.0},
                pinned_element="Al", swept_element="Mg",
            )

    def test_missing_mu0_entry_raises(self):
        with pytest.raises(KeyError):
            compute_formation_energy(
                -10.0, -12.0, {"Al": -1}, {"Al": -3.5}, pinned_element="Al", swept_element="Mg"
            )


class TestMatchesRealFccProductionRecords:
    """
    Regression test against real ACE-potential FCC results from
    Single_defects/FCC/{Vacancy,Substitutional,Interstitial}/*.pkl (mu0_Al,
    mu0_Mg, E_pristine_FCC from reference_energies.pkl) -- E_defect and the
    expected intercept/slope/landau fields are copied verbatim from those
    pickles, computed by the original (pre-port) notebook formula. Confirms
    compute_formation_energy reproduces production numbers exactly, not just
    the hand-derived formula in the other tests above.
    """

    mu0 = {"Al": -3.737530763558215, "Mg": -1.501550231853002}
    E_pristine = -403.6533224642872

    def test_al_vacancy_4a(self):
        # Al_vac_4a.pkl
        result = compute_formation_energy(
            E_defect=-399.2621738501879,
            E_pristine=self.E_pristine,
            delta_n={"Al": -1},
            mu0=self.mu0,
            pinned_element="Al",
            swept_element="Mg",
        )
        assert result["intercept"] == pytest.approx(0.6536178505411212)
        assert result["slope"] == 0
        assert result["slope_pinned"] == 1
        assert result["landau"]["intercept"] == pytest.approx(0.6536178505411212)
        assert result["landau"]["exact"] is False

    def test_mg_on_al_4a_substitution(self):
        # Mg_on_Al_4a.pkl
        result = compute_formation_energy(
            E_defect=-401.28419443178797,
            E_pristine=self.E_pristine,
            delta_n={"Al": -1, "Mg": 1},
            mu0=self.mu0,
            pinned_element="Al",
            swept_element="Mg",
        )
        assert result["intercept"] == pytest.approx(0.1331475007940348)
        assert result["slope"] == -1
        assert result["slope_pinned"] == 1
        assert result["landau"]["intercept"] == pytest.approx(2.3691280324992476)
        assert result["landau"]["exact"] is True

    def test_al_interstitial_4b(self):
        # Al_i_int_4b.pkl
        result = compute_formation_energy(
            E_defect=-404.81417358804435,
            E_pristine=self.E_pristine,
            delta_n={"Al": 1},
            mu0=self.mu0,
            pinned_element="Al",
            swept_element="Mg",
        )
        assert result["intercept"] == pytest.approx(2.5766796398010743)
        assert result["slope"] == 0
        assert result["slope_pinned"] == -1
        assert result["landau"]["intercept"] == pytest.approx(2.5766796398010743)
        assert result["landau"]["exact"] is False
