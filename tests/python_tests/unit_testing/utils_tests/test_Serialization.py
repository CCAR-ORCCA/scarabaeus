# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==========================================#
#  Unit Tests for the Serialization helpers #
# ==========================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
from scarabaeus.utils.Serialization import (
    safe_initial_mass,
    owner_to_dict,
    resolve_format,
    UnknownSerializationFormatError,
)
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])

INITIAL_MASS = 1000.0

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def sc():
    """Spacecraft whose mass is depleted leg by leg."""
    return scb.Spacecraft(
        "ORCCA", -1234, awu(INITIAL_MASS, kg), awu(1e-06, km**2), awu(1.5, None)
    )


def burn_leg(spacecraft, t_start, t_end, m_start, m_end):
    """Append a stored mass profile for one burn leg."""
    t_array = np.linspace(t_start, t_end, 60)
    m_array = m_start + (m_end - m_start) * (t_array - t_start) / (t_end - t_start)
    spacecraft.fit_mass_profile_from_data(
        t_array=t_array, m_array=m_array, domain=(t_start, t_end), units=kg
    )


# endregion Fixtures  #
# --------------------#


# --------------------#
# region       Tests #
# --------------------#
class TestSafeInitialMass:
    """Cases where safe_initial_mass must report the mass at t0, not a later leg's."""

    def test_returns_constant_mass_directly(self, sc):
        """A spacecraft that never propagated reports its constant mass."""
        assert float(safe_initial_mass(sc).values) == pytest.approx(INITIAL_MASS)

    def test_survives_a_single_profile(self, sc):
        """One stored profile reports the mass at its start."""
        burn_leg(sc, 0.0, 600.0, INITIAL_MASS, 900.0)

        assert float(safe_initial_mass(sc).values) == pytest.approx(INITIAL_MASS, abs=1e-6)

    def test_survives_several_legs(self, sc):
        """Later legs do not overwrite the initial mass with a partly depleted one."""
        burn_leg(sc, 0.0, 600.0, INITIAL_MASS, 900.0)
        burn_leg(sc, 1000.0, 1600.0, 900.0, 850.0)

        # 900 kg is where the *second* leg starts, which is the trap
        assert float(safe_initial_mass(sc).values) == pytest.approx(INITIAL_MASS, abs=1e-6)

    def test_matches_via_serialization(self, sc):
        """owner_to_dict records the initial mass, not the latest leg's."""
        burn_leg(sc, 0.0, 600.0, INITIAL_MASS, 900.0)
        burn_leg(sc, 1000.0, 1600.0, 900.0, 850.0)

        info = owner_to_dict(sc)

        assert float(np.asarray(info["tot_mass"]["values"]).ravel()[0]) == pytest.approx(
            INITIAL_MASS, abs=1e-6
        )



class TestResolveFormat:
    """fmt wins over the extension; an unknown fmt raises the custom error."""

    @pytest.mark.parametrize(
        "path, fmt, expected",
        [
            ("a.json", None, "json"),
            ("a.pkl", None, "pickle"),
            ("a.pickle", None, "pickle"),
            ("a.txt", None, "json"),
            ("a.json", "PKL", "pickle"),
            ("a.pkl", "json", "json"),
        ],
    )
    def test_resolves(self, path, fmt, expected):
        assert resolve_format(path, fmt) == expected

    def test_unknown_fmt_raises(self):
        with pytest.raises(UnknownSerializationFormatError):
            resolve_format("a.json", "yaml")


# endregion Tests     #
# --------------------#
