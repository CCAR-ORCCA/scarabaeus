# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ============================================#
#  Functional Tests for the nPlateSRP class  #
# ============================================#

import json

import numpy as np
import pytest
import spiceypy

import src.scarabaeus as scb
import scarabaeus.dynamics.nPlateSRP as _nplate_module

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

CASE_CK = 0
CASE_NADIR = 1
CASE_INERTIAL_FIXED = 2

# instrument ID of the spacecraft body CK (spice_id * 1000), and the
# CK-based frame kernel frame it defines
SC_BODY_CK_ID = -82000
SC_BODY_FRAME = "ORCCA_NPLATE_BODY"
SC_BODY_FRAME_ID = -82900

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def et0(epochs):
    """Raw ET float derived from the shared `epochs` fixture."""
    return epochs[0].times.values


@pytest.fixture
def plate_ck(tmp_path, et0):
    """
    Synthetic CK segment for nplate_config_file's CK-type plate
    (id -64027), reusing Cassini's real SCLK kernel (spice_id -82)
    already present in the locked generic kernel set. No CK data is
    loaded for -64027 by default, so this doesn't collide with
    anything else in the session. A 180-degree rotation about Z turns
    the plate's [-1, 0, 0] normal to J2000 +x, so the plate is lit.
    """
    sc_id = -82
    plate_inst_id = -64027
    et_list = [et0 + 60.0 * i for i in range(5)]
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
    avvs = np.tile([0.0, 0.0, 0.0], (5, 1))

    ck_path = str(tmp_path / "plate.bc")
    scb.SpiceManager.write_ck_type3(
        ck_path, inst_id=plate_inst_id, ref_frame="J2000", sclk_id=sc_id,
        et_list=et_list, quats=quats, avvs=avvs, segid="PLATE NORMAL CK",
    )
    try:
        yield
    finally:
        scb.SpiceManager.unload_kernel_from_pool(ck_path)


@pytest.fixture
def sc_body_ck(tmp_path, et0):
    """
    Synthetic CK segment for the spacecraft's own body-fixed attitude
    (a 90-degree rotation about Z, so it's distinguishable from the
    identity fallback), keyed by the spice_id*1000 instrument ID
    convention used by SpiceManager.get_attitude.
    """
    sc_id = -82
    body_inst_id = SC_BODY_CK_ID
    et_list = [et0 + 60.0 * i for i in range(5)]
    quats = np.tile([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2], (5, 1))
    avvs = np.tile([0.0, 0.0, 0.0], (5, 1))

    ck_path = str(tmp_path / "body.bc")
    scb.SpiceManager.write_ck_type3(
        ck_path, inst_id=body_inst_id, ref_frame="J2000", sclk_id=sc_id,
        et_list=et_list, quats=quats, avvs=avvs, segid="BODY ATTITUDE CK",
    )
    try:
        yield
    finally:
        scb.SpiceManager.unload_kernel_from_pool(ck_path)


@pytest.fixture
def n_model(nplate_config_file):
    """N-plate model (Fixed + CK plates) from the shared config fixture."""
    return scb.nPlateModel(str(nplate_config_file))


@pytest.fixture
def sc_nplate(n_model):
    """
    Spacecraft with spice_id -82 (to reuse the locked Cassini SCLK
    kernel required by the synthetic CK fixtures) and a dedicated name
    to avoid SPICE kernel pool collisions with the shared session `sc`
    fixture (spice_id -1000).
    """
    return scb.Spacecraft(
        "ORCCA_NPLATESRP_FUNCTIONAL",
        -82,
        tot_mass=scb.ArrayWUnits(1000.0, kg),
        area=scb.ArrayWUnits(1e-06, km**2),
        ref_coeff=1.5,
        n_plate_model=n_model,
    )


@pytest.fixture
def sc_tilted(tmp_path):
    """
    Spacecraft whose two fixed plates face the Sun (body -x in nadir
    pointing) but are tilted off that axis. The nadir attitude only fixes x
    on the Sun line; its roll about that line follows the velocity, and
    matters only when a lit plate is off the axis.
    With the plates of ``sc_nplate`` (normals along +-x) the velocity
    Jacobian is identically zero.
    """
    config = {
        "names": ["tilted 1", "tilted 2"],
        "areas": [1e-06, 2e-06],
        "ref coeffs": [1.5, 1.1],
        "abs coeffs": [0.3, 0.2],
        "dref coeffs": [0.3, 0.5],
        "sref coeffs": [0.4, 0.3],
        "normal vectors": [[-0.8, 0.6, 0.0], [-0.6, 0.0, 0.8]],
        "types": ["Fixed", "Fixed"],
        "ids": [None, None],
    }
    config_path = tmp_path / "tilted_plates.json"
    config_path.write_text(json.dumps(config))
    return scb.Spacecraft(
        "ORCCA_NPLATESRP_TILTED",
        -82,
        tot_mass=scb.ArrayWUnits(1000.0, kg),
        area=scb.ArrayWUnits(3e-06, km**2),
        ref_coeff=1.5,
        n_plate_model=scb.nPlateModel(str(config_path)),
    )


@pytest.fixture
def sc_body_frame(tmp_path, sc_body_ck):
    """
    Frame kernel defining SC_BODY_FRAME as the CK-based frame of the
    sc_body_ck attitude, as a mission FK defines the spacecraft body frame.
    """
    fk_path = tmp_path / "body_frame.tf"
    fk_path.write_text(
        "\\begindata\n"
        f"FRAME_{SC_BODY_FRAME} = {SC_BODY_FRAME_ID}\n"
        f"FRAME_{SC_BODY_FRAME_ID}_NAME = '{SC_BODY_FRAME}'\n"
        f"FRAME_{SC_BODY_FRAME_ID}_CLASS = 3\n"
        f"FRAME_{SC_BODY_FRAME_ID}_CLASS_ID = {SC_BODY_CK_ID}\n"
        f"FRAME_{SC_BODY_FRAME_ID}_CENTER = -82\n"
        f"CK_{SC_BODY_CK_ID}_SCLK = -82\n"
        f"CK_{SC_BODY_CK_ID}_SPK = -82\n"
        "\\begintext\n"
    )
    scb.SpiceManager.load_kernel(str(fk_path))
    try:
        yield
    finally:
        scb.SpiceManager.unload_kernel_from_pool(str(fk_path))


def _plate_spacecraft(name, normals, types=None, ids=None, **optional):
    """
    Spacecraft (spice_id -82) with one plate per normal; ``optional`` sets
    the "frames" / "double_sided" keys.
    """
    n = len(normals)
    config = {
        "names": [f"plate {i}" for i in range(n)],
        "areas": [1e-06 * (i + 1) for i in range(n)],
        "ref coeffs": [1.5] * n,
        "abs coeffs": [0.3, 0.2, 0.4, 0.25][:n],
        "dref coeffs": [0.3, 0.5, 0.2, 0.35][:n],
        "sref coeffs": [0.4, 0.3, 0.4, 0.4][:n],
        "normal vectors": [list(map(float, normal)) for normal in normals],
        "types": types or ["Fixed"] * n,
        "ids": ids or [None] * n,
    }
    if "frames" in optional:
        config["frames"] = optional["frames"]
    if "double_sided" in optional:
        config["double sided"] = optional["double_sided"]
    return scb.Spacecraft(
        name,
        -82,
        tot_mass=scb.ArrayWUnits(1000.0, kg),
        area=scb.ArrayWUnits(1e-06 * n, km**2),
        ref_coeff=1.5,
        n_plate_model=scb.nPlateModel(plates_data=config),
    )


def _evaluate(sc, origin, position, current_state, epoch):
    """Acceleration and position Jacobian of a fresh nPlateSRP model."""
    model = scb.nPlateSRP(origin, sc, J2000)
    acc = np.asarray(model.compute_acceleration(position, current_state, epoch, sc))
    jac = np.asarray(
        model._compute_partial_by_position(position, current_state, epoch, sc)
    )
    return acc, jac


def _sun_direction_body(sc, origin, position, current_state, epoch):
    """Unit spacecraft -> Sun vector in the body frame of ``sc``'s attitude."""
    model = scb.nPlateSRP(origin, sc, J2000)
    (_, (sun2body, dcm, *_), _, _), *_ = model._rust_terms(
        position, current_state, epoch, sc
    )
    s = dcm.T @ -np.asarray(sun2body, dtype=float)
    return s / np.linalg.norm(s)


def _set_attitude_mode(sc_nplate, case):
    """Configures the spacecraft's attitudeMode/current_state for the given case."""
    position = np.array([7000.0, 0.0, 0.0])
    current_state = {}
    match case:
        case 0:  # CASE_CK
            sc_nplate.attitudeMode = "CK"
        case 1:  # CASE_NADIR
            sc_nplate.attitudeMode = "nadirPointingSun"
            current_state = {
                ("velocity", sc_nplate.spice_id): scb.ArrayWUnits(np.array([0.0, 7.5, 0.0]), km / sec)
            }
        case 2:  # CASE_INERTIAL_FIXED
            sc_nplate.attitudeMode = "inertialFixedAttitude"
            sc_nplate.fixedInertialAttitude = np.eye(3)
    return position, current_state


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestnPlateSRP:
    """All DynamicModel-inherited methods, across every attitude mode."""

    @pytest.mark.parametrize(
        "case",
        [CASE_CK, CASE_NADIR, CASE_INERTIAL_FIXED],
        ids=["CK", "nadirPointingSun", "inertialFixedAttitude"],
    )
    def test_compute_acceleration(
        self, request, origin, plate_ck, sc_body_ck, et0, case
    ):
        """
        Verifies that the acceleration due to solar radiation pressure
        is finite and 3-dimensional, for each attitude mode. Nadir pointing
        runs on the body-frame plates of sc_tilted (CK plates mixed with
        nadir: test_mixed_plates_under_nadir).
        """
        sc_nplate = request.getfixturevalue(
            "sc_tilted" if case == CASE_NADIR else "sc_nplate"
        )
        position, current_state = _set_attitude_mode(sc_nplate, case)
        nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)

        acc = nplate_srp.compute_acceleration(position, current_state, et0, sc_nplate)

        assert acc.shape == (3,)
        assert np.all(np.isfinite(acc))

    @pytest.mark.parametrize(
        "case",
        [CASE_CK, CASE_NADIR, CASE_INERTIAL_FIXED],
        ids=["CK", "nadirPointingSun", "inertialFixedAttitude"],
    )
    def test_compute_partial_by_position(
        self, request, origin, plate_ck, sc_body_ck, et0, case
    ):
        """
        Verifies that the partial derivative of the SRP acceleration
        with respect to position is a finite 3x3 Jacobian, for each
        attitude mode. Nadir pointing runs on the body-frame plates of
        sc_tilted.
        """
        sc_nplate = request.getfixturevalue(
            "sc_tilted" if case == CASE_NADIR else "sc_nplate"
        )
        position, current_state = _set_attitude_mode(sc_nplate, case)
        nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)

        jac = nplate_srp._compute_partial_by_position(position, current_state, et0, sc_nplate)

        assert jac.shape == (3, 3)
        assert np.all(np.isfinite(jac))

    @pytest.mark.parametrize(
        "case, velocity_dependent",
        [(CASE_CK, False), (CASE_NADIR, True), (CASE_INERTIAL_FIXED, False)],
        ids=["CK", "nadirPointingSun", "inertialFixedAttitude"],
    )
    def test_depends_on_velocity(
        self, sc_tilted, origin, sc_body_ck, et0, case, velocity_dependent
    ):
        """
        Only the nadir attitude follows the velocity; the other modes report a
        zero velocity Jacobian.
        """
        sc_nplate = sc_tilted
        position, current_state = _set_attitude_mode(sc_nplate, case)
        nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)

        assert nplate_srp.depends_on_velocity is velocity_dependent
        jac_v = nplate_srp._compute_partial_by_velocity(
            position, current_state, et0, sc_nplate
        )
        assert jac_v.shape == (3, 3)
        assert bool(np.any(jac_v != 0.0)) is velocity_dependent

    def test_nadir_partials_match_finite_differences(self, sc_tilted, origin, et0):
        """
        The nadir attitude is built from the Sun->spacecraft position and
        velocity, so both Jacobians carry an attitude term. Central differences
        of the acceleration check them independently of the DCM derivatives
        that the analytical and dual-number sides share. The tilted plates
        make the roll about the Sun line, and so the velocity, matter.
        """
        sc_nplate = sc_tilted
        position, current_state = _set_attitude_mode(sc_nplate, CASE_NADIR)
        nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)
        vel_key = ("velocity", sc_nplate.spice_id)
        velocity = np.asarray(current_state[vel_key].values, dtype=float)

        def acc_at(pos, vel):
            state = {vel_key: scb.ArrayWUnits(vel, km / sec)}
            return np.asarray(
                nplate_srp.compute_acceleration(pos, state, et0, sc_nplate)
            )

        def central(f, x0, h):
            cols = []
            for k in range(3):
                dx = np.zeros(3)
                dx[k] = h
                cols.append((f(x0 + dx) - f(x0 - dx)) / (2.0 * h))
            return np.column_stack(cols)

        fd_pos = central(lambda p: acc_at(p, velocity), position, 10.0)
        fd_vel = central(lambda v: acc_at(position, v), velocity, 1e-3)

        an_pos = nplate_srp._compute_partial_by_position(
            position, current_state, et0, sc_nplate
        )
        an_vel = nplate_srp._compute_partial_by_velocity(
            position, current_state, et0, sc_nplate
        )

        for an, fd in ((an_pos, fd_pos), (an_vel, fd_vel)):
            scale = np.max(np.abs(fd))
            assert scale > 0.0
            np.testing.assert_allclose(an, fd, rtol=1e-6, atol=1e-6 * scale)

    def test_compute_partial_by_eta(self, sc_nplate, origin, plate_ck, et0):
        """
        Verifies that the partial derivative of SRP acceleration with
        respect to the SRP scale factor equals compute_acceleration's
        result divided by the (fallback) eta_srp of 1.0.
        """
        position, current_state = _set_attitude_mode(sc_nplate, CASE_INERTIAL_FIXED)
        nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)

        acc = nplate_srp.compute_acceleration(position, current_state, et0, sc_nplate)
        partial_eta = nplate_srp._compute_partial_by_eta(position, current_state, et0, sc_nplate)

        assert np.allclose(partial_eta, acc)


class TestPartialsCheck:
    """
    check_force_model_partials verifies the N-plate partials against the Rust
    dual-number derivatives of the very same kernel, so it runs with or
    without UseAutoGrad (autograd cannot trace into Rust).
    """

    @pytest.mark.parametrize(
        "case",
        [CASE_CK, CASE_NADIR, CASE_INERTIAL_FIXED],
        ids=["CK", "nadirPointingSun", "inertialFixedAttitude"],
    )
    def test_partials_match_dual_numbers_without_autograd(
        self, request, origin, plate_ck, sc_body_ck, et0, tmp_path, monkeypatch, case
    ):
        """
        Without UseAutoGrad the N-plate partials are still checked, and the
        analytical position Jacobian and d(acc)/d(eta) match the dual numbers.
        The attitude is a generic rotation, so a transposed or unrotated
        mapping between the SC-fixed and inertial Jacobians would show up.
        In nadir pointing the attitude follows the state, so the velocity
        Jacobian is checked too, on tilted plates where it is not zero.
        """
        import types

        sc_nplate = request.getfixturevalue(
            "sc_tilted" if case == CASE_NADIR else "sc_nplate"
        )
        position, current_state = _set_attitude_mode(sc_nplate, case)
        if case == CASE_INERTIAL_FIXED:
            sc_nplate.fixedInertialAttitude = spiceypy.eul2m(0.3, -1.1, 0.7, 3, 1, 3)
        # eta != 1 so a d(acc)/d(eta) that forgot the division stands out
        current_state[("eta_srp", sc_nplate.spice_id)] = scb.ArrayWUnits(0.8, None)

        nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)
        fm = types.SimpleNamespace(
            force_models={"nPlateSRP": nplate_srp},
            _dynamics_mode_list=["nPlateSRP"],
            _custom_models={},
            _check_nplate_partials=scb.ForceModelTranslation._check_nplate_partials,
        )

        monkeypatch.delenv("UseAutoGrad", raising=False)
        monkeypatch.chdir(tmp_path)
        scb.ForceModelTranslation._check_all_partials(
            fm, scb.ArrayWUnits(position, km), current_state, et0, sc_nplate
        )

        lines = [
            line.split(",")
            for line in (tmp_path / "partials_FMoutput.txt").read_text().splitlines()
        ]
        assert {line[1] for line in lines} == {"nPlateSRP"}
        if case == CASE_NADIR:
            assert len(lines) == 21  # 3 components x (3 position + 3 velocity + 1 eta)
        else:
            assert len(lines) == 12  # 3 components x (3 position + 1 eta)

        # position, velocity and eta are compared separately: d(acc)/d(position)
        # is ~8 orders of magnitude smaller than d(acc)/d(eta), so a shared
        # absolute tolerance would swallow it whole
        def block_of(line):
            indep = line[2].rsplit("/d(", 1)[1].rstrip(")")
            return "eta" if indep == "eta" else indep[0] if indep[0] == "v" else "r"

        blocks = {block_of(line) for line in lines}
        assert blocks == ({"r", "v", "eta"} if case == CASE_NADIR else {"r", "eta"})
        for block in blocks:
            rows = [line for line in lines if block_of(line) == block]
            analytical = np.array([float(line[3]) for line in rows])
            algo = np.array([float(line[4]) for line in rows])
            # non-vacuous: at least one plate is lit, so nothing here is all zero
            scale = np.max(np.abs(algo))
            assert scale > 0.0
            assert analytical == pytest.approx(algo, rel=1e-10, abs=1e-12 * scale)

    def test_autograd_backend_evaluates_the_model(
        self, sc_nplate, origin, plate_ck, sc_body_ck, et0, monkeypatch
    ):
        """
        Under the autograd backend the rust core is built — at construction and
        on unpickling — and plain arrays evaluate exactly as without it. Only a
        traced input is refused.
        """
        import pickle
        from autograd.numpy.numpy_boxes import ArrayBox

        position, current_state = _set_attitude_mode(sc_nplate, CASE_CK)
        nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)
        expected = np.asarray(
            nplate_srp.compute_acceleration(position, current_state, et0, sc_nplate)
        )
        expected_jac = np.asarray(
            nplate_srp._compute_partial_by_position(
                position, current_state, et0, sc_nplate
            )
        )

        monkeypatch.setattr(_nplate_module.np, "autograd_types", (ArrayBox,))
        built = scb.nPlateSRP(origin, sc_nplate, J2000)
        restored = pickle.loads(pickle.dumps(built))

        for model in (built, restored):
            assert model._rust_cores is not None
            np.testing.assert_array_equal(
                np.asarray(
                    model.compute_acceleration(position, current_state, et0, sc_nplate)
                ),
                expected,
            )
            np.testing.assert_array_equal(
                np.asarray(
                    model._compute_partial_by_position(
                        position, current_state, et0, sc_nplate
                    )
                ),
                expected_jac,
            )


class TestPlateFramesAndSides:
    """
    Plates fixed in a named SPICE frame (the "frames" key) and double-sided
    plates (the "double sided" key).
    """

    # body-frame normals of the two plates, both lit under the sc_body_ck
    # attitude
    BODY_NORMALS = [[0.0, 1.0, 0.0], [-0.6, 0.8, 0.0]]
    # a node of sc_body_ck: its per-sample interpolation intervals hold one
    # record each, so the CK-based frame is undefined between nodes
    EPOCH_OFFSET = 120.0

    def test_named_frame_matches_body_frame(
        self, origin, sc_body_frame, et0
    ):
        """
        Plates fixed in the CK-based body frame of an FK (by name or ID), or
        given as CK plates of the body CK, are the body-frame plates of a
        Spacecraft in the CK attitude. Plates oriented by SPICE alone are the
        same under any attitudeMode, or none.
        """
        epoch = et0 + self.EPOCH_OFFSET
        position = np.array([7000.0, 0.0, 0.0])
        n = len(self.BODY_NORMALS)

        sc_body = _plate_spacecraft("ORCCA_NPLATE_FRAMES_BODY", self.BODY_NORMALS)
        sc_body.attitudeMode = "CK"
        expected_acc, expected_jac = _evaluate(sc_body, origin, position, {}, epoch)
        assert np.any(expected_acc != 0.0)

        variants = {
            "frame name": dict(frames=[SC_BODY_FRAME] * n),
            "frame ID": dict(frames=[SC_BODY_FRAME_ID] * n),
            "CK plates": dict(types=["CK"] * n, ids=[SC_BODY_CK_ID] * n),
        }
        for label, kwargs in variants.items():
            sc = _plate_spacecraft(
                "ORCCA_NPLATE_FRAMES_" + label.replace(" ", "_").upper(),
                self.BODY_NORMALS,
                **kwargs,
            )
            for mode in ("CK", "inertialFixedAttitude", None):
                sc.attitudeMode = mode
                sc.fixedInertialAttitude = spiceypy.eul2m(0.3, -1.1, 0.7, 3, 1, 3)
                acc, jac = _evaluate(sc, origin, position, {}, epoch)
                np.testing.assert_allclose(
                    acc, expected_acc, rtol=1e-12, err_msg=f"{label}, {mode}"
                )
                np.testing.assert_allclose(
                    jac, expected_jac, rtol=1e-12,
                    atol=1e-12 * np.max(np.abs(expected_jac)),
                    err_msg=f"{label}, {mode}",
                )

    def test_named_frame_plates_under_nadir(self, origin, sc_body_frame, et0):
        """
        A model whose plates are all oriented by SPICE does not follow the
        nadir attitude: it does not depend on the velocity, and evaluates as
        with no attitudeMode.
        """
        epoch = et0 + self.EPOCH_OFFSET
        sc = _plate_spacecraft(
            "ORCCA_NPLATE_FRAMES_NADIR",
            self.BODY_NORMALS,
            frames=[SC_BODY_FRAME] * len(self.BODY_NORMALS),
        )
        sc.attitudeMode = None
        position = np.array([7000.0, 0.0, 0.0])
        expected_acc, expected_jac = _evaluate(sc, origin, position, {}, epoch)

        position, current_state = _set_attitude_mode(sc, CASE_NADIR)
        model = scb.nPlateSRP(origin, sc, J2000)
        assert model.depends_on_velocity is False
        np.testing.assert_array_equal(
            model._compute_partial_by_velocity(position, current_state, epoch, sc),
            np.zeros((3, 3)),
        )
        acc, jac = _evaluate(sc, origin, position, current_state, epoch)
        np.testing.assert_array_equal(acc, expected_acc)
        np.testing.assert_array_equal(jac, expected_jac)

    def test_mixed_plates_under_nadir(self, origin, plate_ck, et0):
        """
        A nadir-pointed bus with a CK plate and a plate in a named frame (solar
        arrays on their own CK, say): the bus plate follows the attitude, the
        others keep their SPICE orientation. The acceleration is the one of
        the same plates all in the body frame, and the Jacobians, which hold
        the SPICE plates fixed in the frame, match central differences.
        """
        probe = _plate_spacecraft("ORCCA_NPLATE_MIXED_PROBE", [[-1.0, 0.0, 0.0]])
        position, current_state = _set_attitude_mode(probe, CASE_NADIR)
        # the nadir body -> J2000 DCM at this state
        (_, (sun2body, C, *_), _, _), = scb.nPlateSRP(
            origin, probe, J2000
        )._rust_terms(position, current_state, et0, probe)
        sun_dir = -sun2body / np.linalg.norm(sun2body)

        bus_normal = np.array([-0.8, 0.6, 0.0])
        frame_normal_body = np.array([-0.6, 0.0, 0.8])
        sc = _plate_spacecraft(
            "ORCCA_NPLATE_MIXED_NADIR",
            [bus_normal, [-1.0, 0.0, 0.0], C @ frame_normal_body],
            types=["Fixed", "CK", "Fixed"],
            ids=[None, -64027, None],
            frames=[None, None, "J2000"],
        )
        _set_attitude_mode(sc, CASE_NADIR)
        model = scb.nPlateSRP(origin, sc, J2000)
        assert model.depends_on_velocity
        ck_normal = model.plate_model.get_all_normals(
            et0, sc.spice_id, inertial_frame="J2000"
        )[1]
        # every plate is lit, the CK one included
        for normal in (C @ bus_normal, ck_normal, C @ frame_normal_body):
            assert normal @ sun_dir > 0.1

        same_in_body = _plate_spacecraft(
            "ORCCA_NPLATE_MIXED_BODY", [bus_normal, C.T @ ck_normal, frame_normal_body]
        )
        _set_attitude_mode(same_in_body, CASE_NADIR)
        body_model = scb.nPlateSRP(origin, same_in_body, J2000)
        np.testing.assert_allclose(
            model.compute_acceleration(position, current_state, et0, sc),
            body_model.compute_acceleration(
                position, current_state, et0, same_in_body
            ),
            rtol=1e-12,
        )

        vel_key = ("velocity", sc.spice_id)
        velocity = np.asarray(current_state[vel_key].values, dtype=float)

        def acc_at(pos, vel):
            state = {vel_key: scb.ArrayWUnits(vel, km / sec)}
            return np.asarray(model.compute_acceleration(pos, state, et0, sc))

        def central(f, x0, h):
            cols = []
            for k in range(3):
                dx = np.zeros(3)
                dx[k] = h
                cols.append((f(x0 + dx) - f(x0 - dx)) / (2.0 * h))
            return np.column_stack(cols)

        an_pos = model._compute_partial_by_position(position, current_state, et0, sc)
        an_vel = model._compute_partial_by_velocity(position, current_state, et0, sc)
        fd_pos = central(lambda r: acc_at(r, velocity), position, 10.0)
        fd_vel = central(lambda v: acc_at(position, v), velocity, 1e-3)
        for an, fd in ((an_pos, fd_pos), (an_vel, fd_vel)):
            scale = np.max(np.abs(fd))
            assert scale > 0.0
            np.testing.assert_allclose(an, fd, rtol=1e-6, atol=1e-6 * scale)

        # the SPICE plates do not turn with the velocity, the all-body ones do
        body_vel = body_model._compute_partial_by_velocity(
            position, current_state, et0, same_in_body
        )
        assert not np.allclose(an_vel, body_vel, rtol=1e-3, atol=0.0)

    def test_double_sided_flips_back_lit_plate(self, origin, et0):
        """
        Lit from behind, a double-sided plate pushes as the same plate with
        the opposite normal; a single-sided one not at all.
        """
        position, current_state = np.array([7000.0, 0.0, 0.0]), {}

        def spacecraft(name, normals, double_sided):
            sc = _plate_spacecraft(name, normals, double_sided=double_sided)
            sc.attitudeMode = "inertialFixedAttitude"
            sc.fixedInertialAttitude = spiceypy.eul2m(0.3, -1.1, 0.7, 3, 1, 3)
            return sc

        s = _sun_direction_body(
            spacecraft("ORCCA_NPLATE_SIDES_PROBE", [[1.0, 0.0, 0.0]], [False]),
            origin, position, current_state, et0,
        )
        u = np.cross(s, [0.0, 0.0, 1.0])
        u /= np.linalg.norm(u)
        lit = 0.8 * s + 0.6 * u
        back_lit = -0.6 * s + 0.8 * u

        double = spacecraft(
            "ORCCA_NPLATE_SIDES_DOUBLE", [lit, back_lit], [False, True]
        )
        flipped = spacecraft(
            "ORCCA_NPLATE_SIDES_FLIPPED", [lit, -back_lit], [False, False]
        )
        single = spacecraft("ORCCA_NPLATE_SIDES_SINGLE", [back_lit], [False])

        acc, jac = _evaluate(double, origin, position, current_state, et0)
        expected_acc, expected_jac = _evaluate(
            flipped, origin, position, current_state, et0
        )
        lit_acc, _ = _evaluate(
            spacecraft("ORCCA_NPLATE_SIDES_LIT", [lit], [False]),
            origin, position, current_state, et0,
        )
        # the back-lit plate does contribute
        assert not np.allclose(acc, lit_acc, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(acc, expected_acc, rtol=1e-14)
        np.testing.assert_allclose(
            jac, expected_jac, rtol=1e-14, atol=1e-14 * np.max(np.abs(expected_jac))
        )
        single_acc, single_jac = _evaluate(
            single, origin, position, current_state, et0
        )
        np.testing.assert_array_equal(single_acc, np.zeros(3))
        np.testing.assert_array_equal(single_jac, np.zeros((3, 3)))

    @pytest.mark.parametrize(
        "case",
        [CASE_NADIR, CASE_INERTIAL_FIXED],
        ids=["nadirPointingSun", "inertialFixedAttitude"],
    )
    def test_double_sided_partials_match_dual_numbers(self, origin, et0, case):
        """
        The analytical position (and, in nadir pointing, velocity) Jacobians
        match the dual-number derivatives on a model with lit, back-lit
        double-sided and back-lit single-sided plates.
        """
        probe = _plate_spacecraft("ORCCA_NPLATE_SIDES_PARTIALS", [[1.0, 0.0, 0.0]])
        position, current_state = _set_attitude_mode(probe, case)
        attitude = spiceypy.eul2m(0.3, -1.1, 0.7, 3, 1, 3)
        probe.fixedInertialAttitude = attitude
        s = _sun_direction_body(probe, origin, position, current_state, et0)
        u = np.cross(s, [0.0, 0.0, 1.0])
        u /= np.linalg.norm(u)
        w = np.cross(s, u)

        sc = _plate_spacecraft(
            "ORCCA_NPLATE_SIDES_PARTIALS",
            [0.8 * s + 0.6 * u, 0.6 * s + 0.8 * w, -0.6 * s + 0.8 * u, -0.8 * s + 0.6 * w],
            double_sided=[False, False, True, False],
        )
        _set_attitude_mode(sc, case)
        sc.fixedInertialAttitude = attitude
        model = scb.nPlateSRP(origin, sc, J2000)

        jac_ad, jac_v_ad = np.zeros(9), np.zeros(9)
        for core, inputs, D_r, D_v in model._rust_terms(
            position, current_state, et0, sc, derivatives=True
        ):
            jac, jac_v, _ = core.compute_partials_ad(*inputs, D_r, D_v)
            jac_ad = jac_ad + np.asarray(jac)
            jac_v_ad = jac_v_ad + np.asarray(jac_v)
        pairs = [
            (
                model._compute_partial_by_position(position, current_state, et0, sc),
                jac_ad.reshape(3, 3),
            )
        ]
        if case == CASE_NADIR:
            assert model.depends_on_velocity
            pairs.append(
                (
                    model._compute_partial_by_velocity(position, current_state, et0, sc),
                    jac_v_ad.reshape(3, 3),
                )
            )
        for analytical, dual in pairs:
            scale = np.max(np.abs(dual))
            assert scale > 0.0
            np.testing.assert_allclose(analytical, dual, rtol=1e-10, atol=1e-12 * scale)


def test_inertial_to_body_DCM_real_ck(sc_nplate, origin, sc_body_ck, et0):
    """
    Verifies that inertial_to_body_DCM returns the actual attitude from
    a loaded C-kernel (a 90-degree rotation about Z) rather than the
    identity fallback used when no CK is available.
    """
    nplate_srp = scb.nPlateSRP(origin, sc_nplate, J2000)

    dcm = nplate_srp.inertial_to_body_DCM(scb.EpochArray(et0, "TDB"))

    assert dcm.shape == (3, 3)
    assert not np.allclose(dcm, np.eye(3))


# endregion Tests #
# ----------------#
