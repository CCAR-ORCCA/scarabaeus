# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ====================================================================#
#  Unit Tests for the CustomDynamicModel / CustomParameter classes    #
# ====================================================================#
import src.scarabaeus as scb
from scarabaeus.units.ArrayWUnits import ArrayWUnits as awu
import pytest
import numpy as np

km, kg, sec = scb.Units.get_units(["km", "kg", "sec"])
J2000 = scb.Frame("J2000")

# Reference acceleration used across the tests: a velocity-proportional drag
# whose analytic Jacobians are trivial, so the finite differences can be checked
# against exact values.
DRAG_SCALE = 1e-9


def drag_acceleration(position, velocity, params):
    """a = -k * scale * |r|^-1 * v — depends on position, velocity and a parameter."""
    return -params["drag_k"][0] * DRAG_SCALE * velocity / np.linalg.norm(position)


def drag_partial_by_position(position, velocity, params):
    """da/dr = k * scale * |r|^-3 * v r^T."""
    radius = np.linalg.norm(position)

    return (
        params["drag_k"][0]
        * DRAG_SCALE
        * np.outer(velocity, position)
        / radius**3
    )


def drag_partial_by_velocity(position, params):
    """da/dv = -k * scale * |r|^-1 * I."""
    return -params["drag_k"][0] * DRAG_SCALE * np.eye(3) / np.linalg.norm(position)


def drag_partial_by_k(position, velocity):
    """da/dk = -scale * |r|^-1 * v."""
    return (-DRAG_SCALE * velocity / np.linalg.norm(position)).reshape(3, 1)


# --------------------#
# region    Fixtures #
# --------------------#
@pytest.fixture(autouse=True)
def isolate_custom_registry():
    """Keep the process-wide custom-parameter registry from leaking between tests."""
    saved = dict(scb.StateDefinition._custom_state_components)
    yield
    scb.StateDefinition._custom_state_components.clear()
    scb.StateDefinition._custom_state_components.update(saved)


@pytest.fixture
def epoch():
    """Epoch to define state."""
    return scb.EpochArray("2000 JAN 01 12:00:00.000", "UTC")


@pytest.fixture
def origin():
    """Origin body to define state."""
    return scb.CelestialBody.from_constants("Earth")


@pytest.fixture
def sc():
    """Define spacecraft."""
    return scb.Spacecraft(
        "ORCCA",
        -1234,
        awu(1000.0, kg),
        awu(1e-06, km**2),
        awu(1.5, None),
    )


@pytest.fixture
def pos_0():
    """Initial position."""
    return scb.ArrayWFrame(np.array([7000.0, 100.0, -200.0]), km, J2000)


@pytest.fixture
def vel_0():
    """Initial velocity."""
    return scb.ArrayWFrame(np.array([0.5, 7.5, 0.3]), km / sec, J2000)


@pytest.fixture
def drag_parameter():
    """Scalar drag parameter with finite-difference partials."""
    return scb.CustomParameter("drag_k", default=2.2)


@pytest.fixture
def state_with_parameter(epoch, origin, sc, pos_0, vel_0, drag_parameter):
    """State containing position, velocity and the custom drag parameter."""
    state_def = (
        scb.StateDefinition()
        .position(sc, pos_0)
        .velocity(sc, vel_0)
        .custom_param("drag_k", sc, scb.ArrayWFrame(2.2, None, None))
    )

    return scb.StateArray(epoch=epoch, origin=origin, state=state_def)


@pytest.fixture
def bound_model(state_with_parameter, sc, drag_parameter):
    """Custom model bound to the state, as ForceModelTranslation would do."""
    model = scb.CustomDynamicModel(
        "Drag", drag_acceleration, parameters=[drag_parameter]
    )
    model._bind(
        origin=state_with_parameter.origin, base_frame=J2000, primary_body=sc
    )

    return model


@pytest.fixture
def current_state(state_with_parameter):
    """Direct representation of the state, as handed to the force models."""
    return state_with_parameter._make_direct_representation(
        state_with_parameter.values0.values
    )


# endregion Fixtures #
# --------------------#


# --------------------------#
# region    Registration   #
# --------------------------#
class TestRegistration:
    def test_makes_parameter_usable(self, epoch, origin, sc, pos_0, vel_0):
        """A declared CustomParameter is accepted as a state component."""
        scb.CustomParameter("thermal_k", default=1.0)

        state_def = (
            scb.StateDefinition()
            .position(sc, pos_0)
            .velocity(sc, vel_0)
            .param("thermal_k", sc, scb.ArrayWFrame(1.0, None, None), dynamics="static")
        )
        state = scb.StateArray(epoch=epoch, origin=origin, state=state_def)

        assert ("thermal_k", sc.spice_id) in state.index_rep

    def test_rejects_reserved_name(self):
        """Built-in component names cannot be shadowed by a custom parameter."""
        with pytest.raises(scb.StateDefinition.ReservedComponentNameError):
            scb.StateDefinition.register_custom_parameter("eta_srp")

    def test_rejects_conflicting_redeclaration(self):
        """Re-registering a name with a different definition is refused."""
        scb.StateDefinition.register_custom_parameter("conflicting", dimension=1)

        with pytest.raises(scb.StateDefinition.CustomParameterConflictError):
            scb.StateDefinition.register_custom_parameter("conflicting", dimension=3)

    def test_is_idempotent(self):
        """Re-registering an identical definition leaves the registry unchanged."""
        scb.StateDefinition.register_custom_parameter("repeated", dimension=2)
        scb.StateDefinition.register_custom_parameter("repeated", dimension=2)

        assert scb.StateDefinition.custom_parameters()["repeated"][1] == 2

    def test_unknown_component_rejected(self, sc, pos_0, vel_0):
        """An unregistered name is refused, with a pointer to the registration API."""
        state_def = scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0)

        with pytest.raises(scb.StateDefinition.UnknownComponentError):
            state_def.param(
                "never_registered", sc, scb.ArrayWFrame(1.0, None, None), dynamics="static"
            )


# endregion Registration #
# --------------------------#


# ------------------------#
# region    Signatures   #
# ------------------------#
class TestSignatures:
    def test_rejects_unknown_argument(self):
        """Callables may only declare names present in the evaluation context."""
        with pytest.raises(scb.CustomDynamicModel.SignatureError, match="altitude"):
            scb.CustomDynamicModel("Bad", lambda altitude: np.zeros(3))

    def test_rejects_var_positional(self):
        """Callables are invoked by keyword, so ``*args`` cannot be honoured."""
        with pytest.raises(scb.CustomDynamicModel.SignatureError, match=r"\*args"):
            scb.CustomDynamicModel("Bad", lambda *args: np.zeros(3))

    def test_accepts_var_keyword(self, state_with_parameter, sc, current_state, epoch):
        """A callable declaring ``**kwargs`` receives the whole context."""
        seen = {}

        def acceleration(**context):
            seen.update(context)
            return np.zeros(3)

        model = scb.CustomDynamicModel("All", acceleration)
        model._bind(origin=state_with_parameter.origin, base_frame=J2000, primary_body=sc)
        model.compute_acceleration(np.zeros(3), current_state, 0.0, sc)

        assert {"position", "velocity", "epoch", "params", "state"} <= set(seen)


# endregion Signatures #
# ------------------------#


# ----------------------------#
# region    Model evaluation #
# ----------------------------#
class TestModelEvaluation:
    def test_acceleration_is_evaluated(self, bound_model, current_state, pos_0, vel_0, sc):
        """The user acceleration is evaluated with the state's position and velocity."""
        position = pos_0.quantity.values
        acceleration = bound_model.compute_acceleration(position, current_state, 0.0, sc)

        expected = drag_acceleration(
            position, vel_0.quantity.values, {"drag_k": np.array([2.2])}
        )

        assert np.allclose(acceleration, expected)

    def test_acceleration_shape_is_validated(self, state_with_parameter, sc, current_state):
        """A wrong-sized acceleration is reported instead of silently broadcast."""
        model = scb.CustomDynamicModel("Bad", lambda position: np.zeros(2))
        model._bind(origin=state_with_parameter.origin, base_frame=J2000, primary_body=sc)

        with pytest.raises(scb.CustomDynamicModel.ShapeError):
            model.compute_acceleration(np.zeros(3), current_state, 0.0, sc)

    def test_position_partial_matches_analytic(self, bound_model, current_state, pos_0, vel_0, sc):
        """The finite-difference position Jacobian matches the analytic one."""
        position = pos_0.quantity.values

        numerical = bound_model._compute_partial_by_position(
            position, current_state, 0.0, sc
        )
        analytic = drag_partial_by_position(
            position, vel_0.quantity.values, {"drag_k": np.array([2.2])}
        )

        assert np.allclose(numerical, analytic, rtol=1e-5, atol=1e-20)

    def test_velocity_partial_matches_analytic(self, bound_model, current_state, pos_0, sc):
        """The finite-difference velocity Jacobian matches the analytic one."""
        position = pos_0.quantity.values

        numerical = bound_model._compute_partial_by_velocity(
            position, current_state, 0.0, sc
        )
        analytic = drag_partial_by_velocity(position, {"drag_k": np.array([2.2])})

        assert np.allclose(numerical, analytic, rtol=1e-5, atol=1e-20)

    def test_velocity_partial_absent_when_independent(self):
        """A model whose acceleration ignores velocity reports no velocity dependence."""
        model = scb.CustomDynamicModel("Radial", lambda position: np.zeros(3))

        assert model.depends_on_position is True
        assert model.depends_on_velocity is False

    def test_parameter_partial_matches_analytic(
        self, bound_model, current_state, pos_0, vel_0, sc
    ):
        """The finite-difference parameter Jacobian matches the analytic one."""
        position = pos_0.quantity.values

        numerical = bound_model._compute_partial_by_parameter(
            "drag_k", position, current_state, 0.0, sc
        )
        analytic = drag_partial_by_k(position, vel_0.quantity.values)

        assert numerical.shape == (3, 1)
        assert np.allclose(numerical, analytic, rtol=1e-5, atol=1e-20)

    def test_analytic_partials_are_used(self, state_with_parameter, sc, current_state, pos_0):
        """User-supplied Jacobians are returned as given, not re-derived."""
        parameter = scb.CustomParameter(
            "drag_k", default=2.2, partial=drag_partial_by_k, register=False
        )
        model = scb.CustomDynamicModel(
            "Drag",
            drag_acceleration,
            parameters=[parameter],
            partial_by_position=drag_partial_by_position,
            partial_by_velocity=drag_partial_by_velocity,
        )
        model._bind(origin=state_with_parameter.origin, base_frame=J2000, primary_body=sc)

        position = pos_0.quantity.values
        reference = scb.CustomDynamicModel(
            "Ref",
            drag_acceleration,
            parameters=[scb.CustomParameter("drag_k", default=2.2, register=False)],
        )
        reference._bind(
            origin=state_with_parameter.origin, base_frame=J2000, primary_body=sc
        )

        for method in ("_compute_partial_by_position", "_compute_partial_by_velocity"):
            analytic = getattr(model, method)(position, current_state, 0.0, sc)
            numerical = getattr(reference, method)(position, current_state, 0.0, sc)
            assert np.allclose(analytic, numerical, rtol=1e-5, atol=1e-20)

    def test_parameter_defaults_when_not_estimated(self, epoch, origin, sc, pos_0, vel_0):
        """A parameter absent from the state falls back to its declared default."""
        parameter = scb.CustomParameter("drag_k", default=3.7)
        model = scb.CustomDynamicModel("Drag", drag_acceleration, parameters=[parameter])

        state_def = scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0)
        state = scb.StateArray(epoch=epoch, origin=origin, state=state_def)
        model._bind(origin=state.origin, base_frame=J2000, primary_body=sc)

        current_state = state._make_direct_representation(state.values0.values)
        position = pos_0.quantity.values

        acceleration = model.compute_acceleration(position, current_state, 0.0, sc)
        expected = drag_acceleration(
            position, vel_0.quantity.values, {"drag_k": np.array([3.7])}
        )

        assert np.allclose(acceleration, expected)

    def test_dynamic_parameter_rate_and_jacobian(self, epoch, origin, sc, pos_0, vel_0):
        """A parameter with its own ODE reports both its rate and its Jacobian."""
        tau = 1800.0
        parameter = scb.CustomParameter(
            "emp_a",
            default=1e-9,
            dynamics=lambda params: -params["emp_a"] / tau,
        )
        model = scb.CustomDynamicModel(
            "Empirical",
            lambda velocity, params: params["emp_a"][0] * velocity,
            parameters=[parameter],
        )

        state_def = scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0)
        state = scb.StateArray(epoch=epoch, origin=origin, state=state_def)
        model._bind(origin=state.origin, base_frame=J2000, primary_body=sc)

        current_state = state._make_direct_representation(state.values0.values)
        position = pos_0.quantity.values

        rate = model._compute_parameter_dynamics(
            "emp_a", position, current_state, 0.0, sc
        )
        jacobian = model._compute_parameter_dynamics_partial(
            "emp_a", "emp_a", position, current_state, 0.0, sc
        )

        assert np.allclose(rate, -1e-9 / tau)
        assert jacobian.shape == (1, 1)
        assert np.allclose(jacobian, -1.0 / tau, rtol=1e-5)


# endregion Model evaluation #
# ----------------------------#


# --------------------------------#
# region    Force model wiring   #
# --------------------------------#
class TestForceModelWiring:
    def test_registers_custom_model(
        self, sc, state_with_parameter, drag_parameter
    ):
        """The custom model appears in the dynamics list and in the instantiated models."""
        model = scb.CustomDynamicModel(
            "Drag", drag_acceleration, parameters=[drag_parameter]
        )
        force_model = scb.ForceModelTranslation(primary_body=sc, custom_models=[model])
        force_model.initialize_force_models(state_with_parameter, J2000)

        assert "Drag" in force_model.dynamics_mode_list
        assert force_model.force_models["Drag"] is model
        assert force_model.custom_models["Drag"] is model

    def test_sums_custom_acceleration(
        self, sc, state_with_parameter, current_state, drag_parameter, pos_0
    ):
        """The custom acceleration is added to the built-in point-mass contribution."""
        model = scb.CustomDynamicModel(
            "Drag", drag_acceleration, parameters=[drag_parameter]
        )
        with_drag = scb.ForceModelTranslation(primary_body=sc, custom_models=[model])
        with_drag.initialize_force_models(state_with_parameter, J2000)

        without_drag = scb.ForceModelTranslation(primary_body=sc)
        without_drag.initialize_force_models(state_with_parameter, J2000)

        position = scb.ArrayWUnits(pos_0.quantity.values, km)
        total = with_drag._compute_total_acceleration(position, current_state, 0.0, sc)
        baseline = without_drag._compute_total_acceleration(
            position, current_state, 0.0, sc
        )
        contribution = model.compute_acceleration(
            pos_0.quantity.values, current_state, 0.0, sc
        )

        assert np.allclose(total - baseline, contribution)

    def test_collects_custom_partials(
        self, sc, state_with_parameter, current_state, drag_parameter, pos_0
    ):
        """_collect_partials routes acceleration-by-parameter and -by-velocity keys."""
        model = scb.CustomDynamicModel(
            "Drag", drag_acceleration, parameters=[drag_parameter]
        )
        force_model = scb.ForceModelTranslation(primary_body=sc, custom_models=[model])
        force_model.initialize_force_models(state_with_parameter, J2000)

        position = scb.ArrayWUnits(pos_0.quantity.values, km)
        sid = sc.spice_id

        by_parameter = force_model._collect_partials(
            f"acceleration@{sid}_by_drag_k@{sid}", position, current_state, 0.0, sc
        )
        by_velocity = force_model._collect_partials(
            f"acceleration@{sid}_by_velocity@{sid}", position, current_state, 0.0, sc
        )

        assert by_parameter is not None
        assert by_parameter.values.shape == (3, 1)
        assert by_velocity is not None
        assert np.allclose(
            by_velocity.values,
            drag_partial_by_velocity(pos_0.quantity.values, {"drag_k": np.array([2.2])}),
            rtol=1e-5,
            atol=1e-20,
        )

    def test_no_velocity_partial_without_custom_models(
        self, sc, state_with_parameter, current_state, pos_0
    ):
        """Built-in models alone contribute no d(acceleration)/d(velocity) block."""
        force_model = scb.ForceModelTranslation(primary_body=sc)
        force_model.initialize_force_models(state_with_parameter, J2000)

        position = scb.ArrayWUnits(pos_0.quantity.values, km)
        sid = sc.spice_id

        assert (
            force_model._collect_partials(
                f"acceleration@{sid}_by_velocity@{sid}", position, current_state, 0.0, sc
            )
            is None
        )

    def test_rebinds_to_same_spacecraft(self, sc, state_with_parameter, drag_parameter):
        """Re-initializing the same propagation re-binds without complaint."""
        model = scb.CustomDynamicModel(
            "Drag", drag_acceleration, parameters=[drag_parameter]
        )
        force_model = scb.ForceModelTranslation(primary_body=sc, custom_models=[model])

        force_model.initialize_force_models(state_with_parameter, J2000)
        force_model.initialize_force_models(state_with_parameter, J2000)

        assert model.primary_body is sc

    def test_accepts_single_model(self, sc, drag_parameter):
        """A lone model may be passed without wrapping it in a list."""
        model = scb.CustomDynamicModel(
            "Drag", drag_acceleration, parameters=[drag_parameter]
        )
        force_model = scb.ForceModelTranslation(primary_body=sc, custom_models=model)

        assert "Drag" in force_model.dynamics_mode_list


class TestForceModelErrors:
    def test_rejects_duplicate_name(self, sc, drag_parameter):
        """Two active models may not share a name."""
        first = scb.CustomDynamicModel("Drag", drag_acceleration, parameters=[drag_parameter])
        second = scb.CustomDynamicModel("Drag", lambda position: np.zeros(3))

        with pytest.raises(scb.ForceModelTranslation.DuplicateCustomModelError):
            scb.ForceModelTranslation(primary_body=sc, custom_models=[first, second])

    def test_rejects_duplicate_dynamics(self, sc):
        """Two models may not both integrate the same parameter."""
        parameter = scb.CustomParameter(
            "shared_p", default=1.0, dynamics=lambda params: -params["shared_p"]
        )
        first = scb.CustomDynamicModel(
            "A", lambda params: np.zeros(3), parameters=[parameter]
        )
        second = scb.CustomDynamicModel(
            "B", lambda params: np.zeros(3), parameters=[parameter]
        )

        with pytest.raises(scb.ForceModelTranslation.DuplicateCustomDynamicsError):
            scb.ForceModelTranslation(primary_body=sc, custom_models=[first, second])

    def test_rejects_wrong_type(self, sc):
        """custom_models only accepts CustomDynamicModel instances."""
        with pytest.raises(scb.ForceModelTranslation.CustomModelTypeError):
            scb.ForceModelTranslation(primary_body=sc, custom_models=["Drag"])

    def test_cannot_be_shared_between_spacecraft(
        self, sc, state_with_parameter, drag_parameter
    ):
        """Re-binding one instance to another spacecraft is refused, not silently done."""
        other = scb.Spacecraft("OTHER", -4321, awu(1000.0, kg), awu(1e-06, km**2))
        model = scb.CustomDynamicModel(
            "Drag", drag_acceleration, parameters=[drag_parameter]
        )

        scb.ForceModelTranslation(
            primary_body=sc, custom_models=[model]
        ).initialize_force_models(state_with_parameter, J2000)

        with pytest.raises(scb.CustomDynamicModel.RebindError):
            scb.ForceModelTranslation(
                primary_body=other, custom_models=[model]
            ).initialize_force_models(state_with_parameter, J2000)


# endregion Force model wiring #
# --------------------------------#


# ----------------------------#
# region    Propagation      #
# ----------------------------#
@pytest.fixture
def tspan():
    """Two-hour propagation grid."""
    t0 = scb.SpiceManager.jd2et(2461809.72995654)

    return scb.EpochArray(np.arange(t0, t0 + 7200.0, 120.0), sys="TDB")


def _propagate_with_drag(sc, origin, tspan, pos_0, vel_0, drag_k):
    """Propagate the drag-perturbed orbit for a given drag coefficient."""
    parameter = scb.CustomParameter("drag_k", default=drag_k)
    model = scb.CustomDynamicModel(
        "Drag", drag_acceleration, parameters=[parameter]
    )

    state_def = (
        scb.StateDefinition()
        .position(sc, pos_0)
        .velocity(sc, vel_0)
        .custom_param("drag_k", sc, scb.ArrayWFrame(drag_k, None, None))
    )
    state = scb.StateArray(epoch=tspan[0], origin=origin, state=state_def)

    propagator = scb.Propagator(
        primary_body=sc,
        state_vector=state,
        tspan=tspan,
        force_models=scb.ForceModelTranslation(
            primary_body=sc, custom_models=[model]
        ),
    )
    propagator.propagate(display_progress=False)

    return propagator


class TestPropagation:
    def test_sensitivity_matches_finite_difference(
        self, sc, origin, tspan, pos_0, vel_0
    ):
        """The STM column of the custom parameter matches a trajectory finite difference."""
        nominal = _propagate_with_drag(sc, origin, tspan, pos_0, vel_0, 2.2)

        # Central difference of the whole trajectory: the one-sided form would carry
        # a second-order error of the same size as the tolerance below.
        delta = 1e-2
        plus = _propagate_with_drag(sc, origin, tspan, pos_0, vel_0, 2.2 + delta)
        minus = _propagate_with_drag(sc, origin, tspan, pos_0, vel_0, 2.2 - delta)

        size = nominal.full_state_vector.size
        stm = np.asarray(nominal.STM.quantity.values)[-1].reshape(size, size)
        analytic = stm[:3, 6]
        numerical = (plus.ys[-1][:3] - minus.ys[-1][:3]) / (2.0 * delta)

        assert np.linalg.norm(numerical) > 0.0
        assert np.allclose(analytic, numerical, rtol=1e-4)

    def test_dynamic_parameter_propagates(self, sc, origin, tspan, pos_0, vel_0):
        """A dynamic custom parameter integrates to its analytic solution."""
        tau = 1800.0
        initial = 1e-9

        parameter = scb.CustomParameter(
            "emp_a",
            units=km / sec**2,
            default=initial,
            dynamics=lambda params: -params["emp_a"] / tau,
        )
        model = scb.CustomDynamicModel(
            "Empirical",
            lambda velocity, params: params["emp_a"][0]
            * velocity
            / np.linalg.norm(velocity),
            parameters=[parameter],
        )

        state_def = (
            scb.StateDefinition()
            .position(sc, pos_0)
            .velocity(sc, vel_0)
            .custom_param(
                "emp_a",
                sc,
                scb.ArrayWFrame(np.array([initial]), km / sec**2, J2000),
                dynamics="dynamic",
            )
        )
        state = scb.StateArray(epoch=tspan[0], origin=origin, state=state_def)

        propagator = scb.Propagator(
            primary_body=sc,
            state_vector=state,
            tspan=tspan,
            force_models=scb.ForceModelTranslation(
                primary_body=sc, custom_models=[model]
            ),
        )
        propagator.propagate(display_progress=False)

        elapsed = tspan.et[-1] - tspan.et[0]
        expected = initial * np.exp(-elapsed / tau)

        assert np.isclose(propagator.ys[-1][6], expected, rtol=1e-8)


# endregion Propagation #
# ----------------------------#


# ------------------------------------------#
# region    Third-body typing (regression)  #
# ------------------------------------------#
# 'third_bodies' takes CelestialBody objects only. Names used to reach
# ThreeBodyGravity and fail there with "'str' object has no attribute 'name'";
# worse, a name like 'MARSBARYCENTER' has no entry in the constants library at
# all, so it could never have been resolved.
class TestThirdBodyTyping:
    def test_rejects_names(self, sc):
        """A body name is refused, with the call that builds the object."""
        with pytest.raises(
            scb.ForceModelTranslation.ThirdBodyTypeError, match="from_constants"
        ):
            scb.ForceModelTranslation(primary_body=sc, third_bodies=["MERCURY"])

    def test_suggests_barycenter_for_barycenter_names(self, sc):
        """A barycenter name is pointed at .barycenter(), not from_constants alone."""
        for name in ("MARSBARYCENTER", "MARS BARYCENTER", "Mars_Barycenter"):
            with pytest.raises(
                scb.ForceModelTranslation.ThirdBodyTypeError,
                match=r'from_constants\("MARS"\)\.barycenter\(\)',
            ):
                scb.ForceModelTranslation(primary_body=sc, third_bodies=[name])

    def test_rejects_other_types(self, sc):
        """Anything that is not a CelestialBody is refused."""
        with pytest.raises(scb.ForceModelTranslation.ThirdBodyTypeError):
            scb.ForceModelTranslation(primary_body=sc, third_bodies=[42])

    def test_accepts_bodies_and_barycenters(self, sc):
        """CelestialBody objects, including barycenters, are accepted."""
        mars = scb.CelestialBody.from_constants("MARS")
        force_model = scb.ForceModelTranslation(
            primary_body=sc,
            third_bodies=[scb.CelestialBody.from_constants("VENUS"), mars.barycenter()],
        )

        assert "ThreeBodyGravity" in force_model.dynamics_mode_list

    def test_survives_serialization_round_trip(self, sc):
        """to_dict/from_dict rebuild a barycenter as a barycenter."""
        mars = scb.CelestialBody.from_constants("MARS")
        original = scb.ForceModelTranslation(
            primary_body=sc,
            third_bodies=[scb.CelestialBody.from_constants("VENUS"), mars.barycenter()],
        )

        restored = scb.ForceModelTranslation.from_dict(original.to_dict(), primary_body=sc)
        names = [body.name for body in restored.third_bodies]

        assert names == ["Venus", "Mars Barycenter"]
        assert restored.third_bodies[1].grav_param == mars.barycenter().grav_param

    def test_accepts_legacy_serialized_names(self, sc):
        """Files written before the barycenter flag still load."""
        restored = scb.ForceModelTranslation.from_dict(
            {"third_bodies": ["Venus"]}, primary_body=sc
        )

        assert [body.name for body in restored.third_bodies] == ["Venus"]


# endregion Third-body typing #
# ------------------------------------------#
