# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ======================================#
#  Unit Tests for the nPlateSRP class  #
# ======================================#
import src.scarabaeus as scb
import scarabaeus.dynamics.nPlateSRP as _nplate_module
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


# setup
@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc_area():
    return awu(1e-06, km**2)


@pytest.fixture
def sc_ref_coeff():
    return 1.5


@pytest.fixture
def n_model(nplate_config_file):
    """Define n-plate model."""
    # construct N-plate model using a configuration file
    plate_config = str(nplate_config_file)
    return scb.nPlateModel(plate_config)


# construction
@pytest.fixture
def sc(sc_area, sc_ref_coeff, n_model):
    """Define spacecraft."""
    return scb.Spacecraft(
        "ORCCA", -1234, area=sc_area, ref_coeff=sc_ref_coeff, n_plate_model=n_model)


@pytest.fixture
def state(epoch, origin, sc):
    """Define StateArray object."""
    pos_0 = scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)
    vel_0 = scb.ArrayWFrame(np.array([0.0, 70.0, 0.0]), km / sec, J2000)

    return scb.StateArray(epoch, origin, state=scb.StateDefinition()
                                                .position(sc, pos_0)
                                                .velocity(sc, vel_0),)


@pytest.fixture
def nplate_srp(state, sc):
    return scb.nPlateSRP(state.origin, sc, J2000)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(sc_area, sc_ref_coeff, origin, state, sc):
    """
    Verifies that object is constructed correctly, including the
    P_SRP getter and the eta_srp property's setter/getter (None -> 0,
    explicit value stored as-is).
    """
    # initialize
    nplate_srp = scb.nPlateSRP(state.origin, sc, J2000)

    # ensure correct properties
    assert nplate_srp.origin == origin
    assert nplate_srp.origin_name == origin.name
    assert nplate_srp._spacecraft == sc
    assert nplate_srp._frame == J2000

    assert nplate_srp.area == scb.ArrayWUnits.get_value_in_target_units( sc_area, km**2)  # stored in km^2
    assert nplate_srp.ref_coeff == sc_ref_coeff

    assert nplate_srp._plate_model == sc.n_plate_model
    assert nplate_srp.P_SRP > 0

    # eta_srp property: None clears to 0, explicit value stored as-is
    nplate_srp.eta_srp = None
    assert nplate_srp.eta_srp == 0
    nplate_srp.eta_srp = 0.7
    assert nplate_srp.eta_srp == pytest.approx(0.7)


def test_inertial_to_body_DCM(sc, state, epoch):
    """
    Verifies that inertial_to_body_DCM returns a valid 3x3 DCM.

    No CK attitude kernel is loaded for spacecraft -1234, so the method falls
    back to the identity matrix.  The test confirms the shape and the fallback
    values.
    """
    nplate_srp = scb.nPlateSRP(state.origin, sc, J2000)
    dcm = nplate_srp.inertial_to_body_DCM(epoch)

    assert dcm.shape == (3, 3)
    assert np.allclose(dcm, np.eye(3))


class TestEtaSrp:
    @pytest.mark.parametrize(
        "case",
        ["cached", "dynamic_key", "fallback"],
        ids=["Cached from StateArray", "Dynamic current_state Key", "Fallback (1.0)"],
    )
    def test_get_eta_srp(self, sc, state, case):
        """
        Verifies _get_eta_srp's priority order: a dynamic ("eta_srp",
        spice_id) key in current_state, a value pushed directly into
        self._eta_srp_map by a caller, and the 1.0 fallback when neither
        is present.
        """
        nplate_srp = scb.nPlateSRP(state.origin, sc, J2000)

        match case:
            case "cached":
                nplate_srp._eta_srp_map[sc.spice_id] = 0.5
                result = nplate_srp._get_eta_srp({}, sc)
                assert result == pytest.approx(0.5)
            case "dynamic_key":
                result = nplate_srp._get_eta_srp({("eta_srp", sc.spice_id): awu(0.42, None)}, sc)
                assert result == pytest.approx(0.42)
            case "fallback":
                result = nplate_srp._get_eta_srp({}, sc)
                assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vec_1, vec_2, expected",
    [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.pi / 2),
        (np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), 0.0),
        (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.pi),
    ],
    ids=["Perpendicular", "Same Direction", "Opposite"],
)
def test_angle_between(sc, state, vec_1, vec_2, expected):
    """Verifies _angle_between for perpendicular, parallel, and opposite vectors."""
    nplate_srp = scb.nPlateSRP(state.origin, sc, J2000)
    assert nplate_srp._angle_between(vec_1, vec_2) == pytest.approx(expected)


class TestErrorHandling:
    def test_bad_plate_area_units(self, state, sc, n_model):
        """
        Verifies ValueError when a plate area isn't expressed in km^2.

        nPlateModel.areas always stores km^2 through its own setter, so the
        underlying attribute is tampered with directly to exercise this
        defensive check.
        """
        n_model._areas[0] = awu(1e-06, km)  # wrong units: km instead of km^2
        with pytest.raises(ValueError, match="units of nPlate areas not compliant"):
            scb.nPlateSRP(state.origin, sc, J2000)

    @pytest.mark.parametrize(
        "method_name",
        ["compute_acceleration", "_compute_partial_by_position"],
        ids=["compute_acceleration", "_compute_partial_by_position"],
    )
    def test_fixed_inertial_attitude_malformed_matrix(self, sc, state, method_name):
        """
        Verifies ValueError when attitudeMode='inertialFixedAttitude' but
        fixedInertialAttitude isn't a 3x3 numpy array, for both
        compute_acceleration and _compute_partial_by_position.
        """
        nplate_srp = scb.nPlateSRP(state.origin, sc, J2000)
        sc.attitudeMode = "inertialFixedAttitude"
        sc.fixedInertialAttitude = "not a matrix"

        method = getattr(nplate_srp, method_name)
        position = np.array([7000.0, 0.0, 0.0])
        et0 = scb.SpiceManager.cal2et("2000 JAN 01 12:00:00.000")
        with pytest.raises(ValueError, match="fixedInertialAttitude must be a 3x3 numpy array"):
            method(position, {}, et0, sc)

    def test_missing_rust_core_is_reported(self):
        """
        A model without the extension raises the documented error: the rust
        kernels are the only implementation of the acceleration.
        """
        model = _nplate_module.nPlateSRP.__new__(_nplate_module.nPlateSRP)
        model._rust_cores = None

        with pytest.raises(
            _nplate_module.nPlateSRP.RustCoreUnavailableError, match="rust extension"
        ):
            model._require_rust_core(np.zeros(3), 1.0)

    @pytest.mark.parametrize(
        "traced", ["position", "eta_srp"], ids=["position", "eta_srp"]
    )
    def test_autograd_tracing_is_reported(self, sc, state, monkeypatch, traced):
        """
        An input autograd is tracing raises, naming the dual-number check,
        instead of failing on the ArrayBox inside the rust call. The plate
        sums run in rust, which autograd cannot trace.
        """
        from autograd import jacobian
        from autograd.numpy.numpy_boxes import ArrayBox

        monkeypatch.setattr(_nplate_module.np, "autograd_types", (ArrayBox,))
        nplate_srp = scb.nPlateSRP(state.origin, sc, J2000)
        sc.attitudeMode = "inertialFixedAttitude"
        sc.fixedInertialAttitude = np.eye(3)

        position = np.array([7000.0, 0.0, 0.0])
        et0 = scb.SpiceManager.cal2et("2000 JAN 01 12:00:00.000")
        if traced == "position":
            func, seed = (
                lambda p: nplate_srp.compute_acceleration(p, {}, et0, sc),
                position,
            )
        else:
            func, seed = (
                lambda e: nplate_srp.compute_acceleration(
                    position, {("eta_srp", sc.spice_id): e}, et0, sc
                ),
                1.0,
            )

        with pytest.raises(
            _nplate_module.nPlateSRP.AutogradTracingError,
            match="compute_partials_ad",
        ):
            jacobian(func)(seed)

    def test_invalid_attitude_mode(self, sc, state):
        """Verifies ValueError for an unrecognised attitudeMode."""
        nplate_srp = scb.nPlateSRP(state.origin, sc, J2000)
        sc.attitudeMode = "bogus_mode"

        position = np.array([7000.0, 0.0, 0.0])
        et0 = scb.SpiceManager.cal2et("2000 JAN 01 12:00:00.000")
        with pytest.raises(ValueError, match="Invalid attitudeMode"):
            nplate_srp.compute_acceleration(position, {}, et0, sc)


# NOTE: the rest of this class' methods (compute_acceleration's actual
# CK/nadir/inertial-fixed acceleration math, _compute_partial_by_position's
# Jacobian, and _compute_partial_by_eta) are covered in functional testing.
