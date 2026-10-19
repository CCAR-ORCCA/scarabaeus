# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ===================================#
#  Unit Tests for the Bplane class  #
# ===================================#
import datetime
import src.scarabaeus as scb
import pytest
import numpy as np
import pathlib

km, kg, sec, hr, day, m, rad, unitless = scb.Units.get_units(
    ["km", "kg", "sec", "hr", "day", "m", "rad", "unitless"]
)

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def epoch():
    """Epoch to define state."""
    epoch_0 = scb.EpochArray("2000 JAN 01 00:00:00.000", "TDB")
    return epoch_0


@pytest.fixture
def bplane_file(earth_bplane_fk):
    file = str(earth_bplane_fk)

    return file


@pytest.fixture
def bplane(epoch, bplane_file):
    """Constructed Bplane for reuse across tests."""
    return scb.Bplane(
        epoch,
        "EARTH_BPLANE",
        12345678,
        "MOON",
        301,
        "EARTH_BARYCENTER",
        3,
        bplane_file,
        v_hat=False,
        new_bplane=False,
    )


@pytest.fixture
def hyperbolic_state():
    """Relative state (sc - target) giving a hyperbolic Earth flyby (ecc > 1)."""
    # At r = 400,000 km, v_esc ≈ 1.41 km/s; v = 3.0 km/s → C3 > 0
    return scb.ArrayWUnits(
        [400000.0, 0.0, 0.0, 0.0, 3.0, 0.5],
        [km, km, km, km / sec, km / sec, km / sec],
    )


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization(epoch, bplane_file):
    """
    Verifies correct attribute assignment on construction and all
    read-only properties, rotation matrices, and B-plane reference parameters.
    """
    bp = scb.Bplane(
        epoch,
        "EARTH_BPLANE",
        12345678,
        "MOON",
        301,
        "EARTH_BARYCENTER",
        3,
        bplane_file,
        v_hat=False,
        new_bplane=False,
    )

    # Properties
    assert bp.epoch is bp._epoch
    assert bp.bplane_name == "EARTH_BPLANE"
    assert bp.bplane_spice_id == 12345678
    assert bp.sc_name == "MOON"
    assert bp.sc_spice_id == "MOON"
    assert bp.target_name == "EARTH_BARYCENTER"
    assert bp.target_spice_id == "EARTH BARYCENTER"
    assert bp.v_hat is False
    assert bp.new_bplane is False
    assert bp.fk_file == pathlib.Path(bplane_file)

    # Rotation matrices
    assert isinstance(bp.rot_to_j2000, scb.ArrayWUnits)
    assert bp.rot_to_j2000.values.shape == (3, 3)
    assert isinstance(bp.rot_to_bplane, scb.ArrayWUnits)
    assert bp.rot_to_bplane.values.shape == (3, 3)
    assert np.allclose(bp.rotmat_to_j2000.values, bp.rot_to_j2000.values)
    assert np.allclose(bp.rotmat_to_bplane.values, bp.rot_to_bplane.values)

    # B-plane reference parameters
    assert isinstance(bp.bt_ref, scb.ArrayWUnits)
    assert isinstance(bp.br_ref, scb.ArrayWUnits)
    assert isinstance(bp.ltof_ref, scb.ArrayWUnits)
    assert isinstance(bp.time_close_approach_ref, scb.ArrayWUnits)
    assert isinstance(bp.bpar_ref, scb.ArrayWUnits)
    assert bp.bpar_ref.values.shape == (3,)


class TestCoerceEpoch:
    def test_passthrough(self, epoch):
        """EpochArray input is returned as the same object."""
        result = scb.Bplane._coerce_epoch(epoch)
        assert result is epoch

    @pytest.mark.parametrize(
        "val", [0.0, 42, datetime.datetime(2024, 1, 1, 0, 0, 0)],
        ids=["float", "int", "datetime"],
    )
    def test_numeric_types(self, val):
        """Float, int, and datetime inputs are all wrapped in an EpochArray
        (datetime is converted to ET via spice.str2et)."""
        result = scb.Bplane._coerce_epoch(val)
        assert isinstance(result, scb.EpochArray)

    def test_invalid_raises(self):
        """Unsupported input types raise TypeError."""
        with pytest.raises(TypeError):
            scb.Bplane._coerce_epoch({"not": "valid"})


@pytest.mark.parametrize(
    "body, expected",
    [
        ("MOON", "MOON"),
        ("NOT_A_REAL_BODY_XYZ", ValueError),
        (999999999, 999999999),
        (3.14, TypeError),
    ],
    ids=["Valid Name", "Invalid Name", "Int Without Name", "Invalid Type"],
)
def test_resolve_body_id(body, expected):
    """A resolvable name string is returned unchanged; an unresolvable
    name raises ValueError; an int with no SPICE name falls back to the
    raw int; any other type raises TypeError."""
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            scb.Bplane._resolve_body_id(body)
    else:
        assert scb.Bplane._resolve_body_id(body) == expected


@pytest.mark.parametrize(
    "method_name, arg",
    [("_check_frame_name", "J2000"), ("_check_frame_id", 1)],
    ids=["Frame Name", "Frame ID"],
)
def test_check_frame_already_registered_raises(method_name, arg):
    """J2000 (name and frame ID 1) is always registered, so both checks
    must raise RuntimeError for it."""
    method = getattr(scb.Bplane, method_name)
    with pytest.raises(RuntimeError, match="already"):
        method(arg)


def test_create_kernel_existing_file_raises(tmp_path):
    """Constructing a new B-plane (new_bplane=True) over an FK path that
    already exists on disk must raise RuntimeError before anything is
    overwritten."""
    fk_path = tmp_path / "new_bplane.tf"
    fk_path.write_text("dummy")

    with pytest.raises(RuntimeError, match="already exists"):
        scb.Bplane(
            0.0, "SOME_NEW_BPLANE_FRAME", 88887777, "MOON", 301,
            "EARTH_BARYCENTER", 3, fk_path,
            v_hat=False, new_bplane=True, scenario_kernel_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "case",
    ["non_barycenter_target", "near_degenerate_warns", "batch_input"],
)
def test_compute_hyperbolic_parameters_edge_cases(bplane, hyperbolic_state, case):
    """
    Verifies three compute_hyperbolic_parameters branches not exercised
    by the other tests: GM lookup for a non-"*BARYCENTER" target name,
    the near-degenerate T-vector RuntimeWarning, and (N, 6) batch input.
    """
    match case:
        case "non_barycenter_target":
            bplane._target_name = "MOON"
            bpar = bplane.compute_hyperbolic_parameters(hyperbolic_state)
            assert isinstance(bpar, scb.ArrayWUnits)

        case "near_degenerate_warns":
            # r, v chosen (via numerical root-finding) so the incoming
            # asymptote direction S lands within ~1e-16 of the z-axis,
            # well below the method's own 1e-12 degeneracy threshold.
            near_degenerate_state = scb.ArrayWUnits(
                [7000.0, 0.0, 0.0, -2.998868375910746, 0.0, 18.98813559322034],
                [km, km, km, km / sec, km / sec, km / sec],
            )
            with pytest.warns(RuntimeWarning, match="near-degenerate"):
                bplane.compute_hyperbolic_parameters(near_degenerate_state)

        case "batch_input":
            batch_state = scb.ArrayWUnits(
                np.tile(hyperbolic_state.values, (2, 1)),
                [km, km, km, km / sec, km / sec, km / sec],
            )
            bpar_batch = bplane.compute_hyperbolic_parameters(batch_state)
            bpar_single = bplane.compute_hyperbolic_parameters(hyperbolic_state)

            assert bpar_batch.values.shape == (2, 3)
            np.testing.assert_allclose(bpar_batch.values[0], bpar_single.values)
            np.testing.assert_allclose(bpar_batch.values[1], bpar_single.values)


def test_compute_parameters(bplane, hyperbolic_state):
    """
    Verifies that compute_parameters returns the linear (non-conic)
    B-plane projection [B.T, B.R, ltof], matching a direct computation
    from the B-plane rotation matrix.
    """
    bpar = bplane.compute_parameters(hyperbolic_state)

    assert isinstance(bpar, scb.ArrayWUnits)
    assert bpar.values.shape == (3,)

    R = bplane.rot_to_bplane.values
    pos, vel = hyperbolic_state[0:3].values, hyperbolic_state[3:6].values
    rel_pos_bp = R @ pos
    rel_vel_bp = R @ vel
    ltof_val = -rel_pos_bp[2] / rel_vel_bp[2]
    sc_bp_pos = rel_pos_bp + ltof_val * rel_vel_bp

    np.testing.assert_allclose(
        bpar.values, [sc_bp_pos[0], sc_bp_pos[1], ltof_val]
    )

    # ltof is the time left until the crossing: moving the state dt seconds
    # forward along its velocity shortens it by dt (regression: it grew).
    dt = 600.0
    shifted = hyperbolic_state.values.copy()
    shifted[0:3] += dt * vel
    bpar_later = bplane.compute_parameters(
        scb.ArrayWUnits(shifted, hyperbolic_state.units)
    )
    np.testing.assert_allclose(bpar_later.values[2], ltof_val - dt)


def test_compute_jacobian(bplane, hyperbolic_state):
    """
    Verifies that compute_jacobian returns a (3, 3) Jacobian matrix
    for a hyperbolic approach trajectory.
    """
    jac = bplane.compute_jacobian(hyperbolic_state)

    assert isinstance(jac, scb.ArrayWUnits)
    assert jac.values.shape == (3, 3)


class TestCovarianceMapping:
    def test_spacecraft_covariance(self, bplane, hyperbolic_state):
        """
        Verifies that target_covariance maps a 6x6 spacecraft covariance
        to a symmetric (3, 3) B-plane covariance.
        """
        sc_cov = 0.01 * np.eye(6)
        p_bp = bplane.target_covariance(sc_cov, hyperbolic_state)

        assert isinstance(p_bp, scb.ArrayWUnits)
        assert p_bp.values.shape == (3, 3)
        assert np.allclose(p_bp.values, p_bp.values.T)

    def test_joint_covariance(self, bplane, hyperbolic_state):
        """
        Verifies that target_spacecraft_covariance maps a 12x12 joint
        covariance to a symmetric (3, 3) B-plane covariance.
        """
        sc_targ_cov = 0.01 * np.eye(12)
        p_bp = bplane.target_spacecraft_covariance(sc_targ_cov, hyperbolic_state)

        assert isinstance(p_bp, scb.ArrayWUnits)
        assert p_bp.values.shape == (3, 3)
        assert np.allclose(p_bp.values, p_bp.values.T)


def test_linear_targeting_non_convergence_raises(bplane, hyperbolic_state):
    """
    Verifies that linear_targeting raises RuntimeError when it fails to
    converge within max_iterations.

    Uses the plain `bplane` fixture (target_name="EARTH_BARYCENTER"),
    whose bpar_ref reflects the Moon's real (bound, elliptic) orbit, so
    compute_hyperbolic_parameters gives NaN reference B·T/B·R -- the
    residual is NaN every iteration and can never satisfy the
    convergence check.
    """
    with pytest.raises(RuntimeError, match="did not converge"):
        bplane.linear_targeting(hyperbolic_state, max_iterations=3)


def test_propagate_covariance_to_close_approach_raises(bplane, hyperbolic_state):
    """Verifies that the not-yet-implemented method always raises."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        bplane.propagate_covariance_to_close_approach(np.eye(6), hyperbolic_state)


# NOTE: the rest of this class' methods are tested in integration tests
