# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =============================================#
#  Unit Tests for the StateDefinition class  #
# =============================================#
import src.scarabaeus as scb
import pytest
import numpy as np

km, sec, kg, rad = scb.Units.get_units(["km", "sec", "kg", "rad"])
J2000 = scb.Frame("J2000")

# --------------------#
# region    Fixtures #
# --------------------#


@pytest.fixture
def sc():
    return scb.Spacecraft("ORCCA", -1234, tot_mass=scb.ArrayWUnits(1500.0, kg))


@pytest.fixture
def sc2():
    return scb.Spacecraft("ORCCA2", -5678, tot_mass=scb.ArrayWUnits(500.0, kg))


@pytest.fixture
def pos_0():
    return scb.ArrayWFrame(np.array([7000.0, 0.0, 0.0]), km, J2000)


@pytest.fixture
def vel_0():
    return scb.ArrayWFrame(np.array([0.0, 7.0, 0.0]), km / sec, J2000)


# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization():
    """
    Verifies that a fresh StateDefinition has empty components, zeroed
    counters, and all ordering flags set to False.
    """
    sd = scb.StateDefinition()

    assert sd._components == []
    assert sd._position_count == 0
    assert sd._velocity_count == 0
    assert sd._last_component_was_position is False
    assert sd._seen_static is False
    assert sd._seen_considered is False
    assert len(sd) == 0


def test_position_velocity_chaining(sc, pos_0, vel_0):
    """
    Verifies position/velocity chaining stores correct tuples, updates
    counters, and exposes the list interface correctly.
    """
    sd = scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0)

    # Tuple content
    assert sd[0][0] == "position" and sd[0][2] == "estimated" and sd[0][3] == "dynamic"
    assert sd[1][0] == "velocity" and sd[1][4] is sc and sd[1][5] is vel_0

    # Counters
    assert sd._position_count == 1
    assert sd._velocity_count == 1
    assert sd._last_component_was_position is False

    # List interface
    assert len(sd) == 2
    assert [c[0] for c in sd] == ["position", "velocity"]
    assert sd[-1][0] == "velocity"
    assert "2" in repr(sd)
    assert sd.components is sd._components


def test_param(sc, pos_0, vel_0):
    """
    Verifies that param() appends a scalar static parameter with the
    correct name, estimation, and dynamics values.
    """
    eta_val = scb.ArrayWUnits(np.array([1.0]), None)
    sd = (scb.StateDefinition()
          .position(sc, pos_0)
          .velocity(sc, vel_0)
          .param("eta_srp", sc, eta_val, dynamics="static"))

    assert len(sd) == 3
    name, dim, est, dyn, owner, _ = sd[2]
    assert name == "eta_srp"
    assert dim == 1
    assert est == "estimated"
    assert dyn == "static"
    assert owner is sc

    # a_pfogm/beta_pfogm accept any dimension that's a multiple of 3
    pfogm = scb.ArrayWUnits(np.zeros(6), None)
    sd_pfogm = scb.StateDefinition().param("a_pfogm", sc, pfogm, dimension=6)
    assert sd_pfogm[0][1] == 6


class TestErrors:
    """Groups all validation and ordering error cases."""

    def test_velocity_without_position(self, sc, vel_0):
        """velocity() without a preceding position() raises ValueError."""
        with pytest.raises(ValueError, match="must appear immediately after"):
            scb.StateDefinition().velocity(sc, vel_0)

    @pytest.mark.parametrize(
        "est, dyn, match",
        [
            ("bad_est", "dynamic", "Estimation must be"),
            ("estimated", "bad_dyn", "Dynamics must be"),
        ],
        ids=["Bad Estimation", "Bad Dynamics"],
    )
    def test_validate_modes(self, est, dyn, match):
        """Unrecognised estimation or dynamics strings raise ValueError."""
        with pytest.raises(ValueError, match=match):
            scb.StateDefinition._validate_modes(est, dyn)

    def test_invalid_param_name(self, sc):
        """An unrecognised parameter name raises ValueError."""
        with pytest.raises(ValueError, match="not allowed in StateArray"):
            scb.StateDefinition().param(
                "unknown_param_xyz", sc, scb.ArrayWUnits(np.array([1.0]), None)
            )

    def test_duplicate_component(self, sc, pos_0, vel_0):
        """Adding the same component/owner pair twice raises ValueError."""
        eta = scb.ArrayWUnits(np.array([1.0]), None)
        with pytest.raises(ValueError, match="Duplicate component"):
            (scb.StateDefinition()
             .position(sc, pos_0)
             .velocity(sc, vel_0)
             .param("eta_srp", sc, eta, dynamics="static")
             .param("eta_srp", sc, eta, dynamics="static"))

    def test_duplicate_dv_man(self, sc, pos_0, vel_0):
        """Adding the same dv_man variant twice raises ValueError."""
        dv = scb.ArrayWUnits(np.array([0.1, 0.0, 0.0]), km / sec)
        with pytest.raises(ValueError, match="Duplicate maneuver"):
            (scb.StateDefinition()
             .position(sc, pos_0)
             .velocity(sc, vel_0)
             .param("dv_man", sc, dv)
             .param("dv_man", sc, dv))

    def test_duplicate_dra(self, sc, pos_0, vel_0):
        """Adding the same dRA component for the same owner twice raises ValueError."""
        angle = scb.ArrayWUnits(np.array([0.01]), rad)
        with pytest.raises(ValueError, match="Duplicate component"):
            (scb.StateDefinition()
             .position(sc, pos_0)
             .velocity(sc, vel_0)
             .param("dRA1", sc, angle, dynamics="static")
             .param("dRA1", sc, angle, dynamics="static"))

    @pytest.mark.parametrize(
        "case, match",
        [
            ("dynamic_after_static", "dynamic components must appear before static"),
            ("estimated_after_considered", "estimated components must appear before considered"),
        ],
        ids=["Dynamic After Static", "Estimated After Considered"],
    )
    def test_enforce_global_order_violations(self, sc, pos_0, vel_0, case, match):
        """
        Verifies both _enforce_global_order ordering-rule violations:
        a dynamic component after a static one, and an estimated
        component after a considered one.
        """
        eta = scb.ArrayWUnits(np.array([1.0]), None)
        gm = scb.ArrayWUnits(np.array([398600.4]), None)
        sd = scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0)

        with pytest.raises(ValueError, match=match):
            if case == "dynamic_after_static":
                sd.param("eta_srp", sc, eta, dynamics="static")
                sd.param("gm", sc, gm, dynamics="dynamic")
            else:
                sd.param("eta_srp", sc, eta, estimation="considered", dynamics="static")
                sd.param("gm", sc, gm, estimation="estimated", dynamics="static")

    def test_validate_component_dimension_mismatch(self, sc):
        """A fixed-dimension component with the wrong dimension raises ValueError."""
        eta = scb.ArrayWUnits(np.array([1.0, 2.0]), None)
        with pytest.raises(ValueError, match="incorrect dimension"):
            scb.StateDefinition().param("eta_srp", sc, eta, dimension=2)

    def test_validate_component_pfogm_dimension_not_multiple_of_3(self, sc):
        """a_pfogm/beta_pfogm with a dimension that isn't a multiple of 3 raises ValueError."""
        bad = scb.ArrayWUnits(np.array([1.0, 2.0]), None)
        with pytest.raises(ValueError, match=r"3 \* n_batches"):
            scb.StateDefinition().param("a_pfogm", sc, bad, dimension=4)

    def test_validate_component_fourier_dimension_not_multiple_of_3(self, sc):
        """a_fourier with a dimension that isn't a multiple of 3 raises ValueError."""
        bad = scb.ArrayWUnits(np.array([1.0, 2.0]), None)
        with pytest.raises(ValueError, match=r"3 \* \(2 \* degree \+ 1\)"):
            scb.StateDefinition().param("a_fourier", sc, bad, dimension=4)

    def test_validate_component_ppoly_dimension_not_multiple_of_3(self, sc):
        """a_ppoly with a dimension that isn't a multiple of 3 raises ValueError."""
        bad = scb.ArrayWUnits(np.array([1.0, 2.0]), None)
        with pytest.raises(ValueError, match=r"3 \* \(n_batches \+ 1\)"):
            scb.StateDefinition().param("a_ppoly", sc, bad, dimension=4)

    @pytest.mark.parametrize(
        "case, expected",
        [
            ("invalid_tuple_shape", "Invalid component at index 0"),
            ("incomplete_dRA_dDEC", "Incomplete finite-burn angular pair"),
            ("a_fogm_too_few_preceding", "must appear after 'position' and 'velocity'"),
            ("a_fogm_wrong_predecessor", "must appear immediately after"),
            ("a_fogm_correctly_placed", None),
            ("a_fogm_misplaced_not_last_is_unchecked", None),
        ],
        ids=[
            "Invalid Tuple Shape",
            "Incomplete dRA/dDEC Pair",
            "a_fogm Too Few Preceding",
            "a_fogm Wrong Predecessor",
            "a_fogm Correctly Placed",
            "a_fogm Misplaced But Not Last (unchecked)",
        ],
    )
    def test_extend_from_components_errors(self, sc, pos_0, vel_0, case, expected):
        """
        Verifies extend_from_components' validation branches: an invalid
        raw tuple is wrapped in a ValueError naming its index, an
        incomplete dRA/dDEC pair raises after the full list is processed,
        and a_fogm ordering is enforced when a_fogm is the LAST tuple
        processed (too few preceding components, or a wrong immediate
        predecessor).

        The last case documents a known limitation: the a_fogm ordering
        check only runs using the loop's leftover name/owner from
        whichever tuple was processed LAST, so a misplaced a_fogm earlier
        in the list is silently accepted instead of raising ValueError.
        """
        a_fogm_val = scb.ArrayWUnits(np.array([0.0, 0.0, 0.0]), None)
        angle = scb.ArrayWUnits(np.array([0.01]), rad)
        eta = scb.ArrayWUnits(np.array([1.0]), None)

        match case:
            case "invalid_tuple_shape":
                components = [("position", 3, "estimated", "dynamic", sc)]  # 5-tuple
            case "incomplete_dRA_dDEC":
                components = [
                    ("position", 3, "estimated", "dynamic", sc, pos_0),
                    ("velocity", 3, "estimated", "dynamic", sc, vel_0),
                    ("dRA1", 1, "estimated", "static", sc, angle),
                ]
            case "a_fogm_too_few_preceding":
                components = [("a_fogm", 3, "estimated", "dynamic", sc, a_fogm_val)]
            case "a_fogm_wrong_predecessor":
                components = [
                    ("position", 3, "estimated", "dynamic", sc, pos_0),
                    ("velocity", 3, "estimated", "dynamic", sc, vel_0),
                    ("eta_srp", 1, "estimated", "dynamic", sc, eta),
                    ("a_fogm", 3, "estimated", "dynamic", sc, a_fogm_val),
                ]
            case "a_fogm_correctly_placed":
                components = [
                    ("position", 3, "estimated", "dynamic", sc, pos_0),
                    ("velocity", 3, "estimated", "dynamic", sc, vel_0),
                    ("a_fogm", 3, "estimated", "dynamic", sc, a_fogm_val),
                ]
            case "a_fogm_misplaced_not_last_is_unchecked":
                components = [
                    ("a_fogm", 3, "estimated", "dynamic", sc, a_fogm_val),  # invalid placement
                    ("position", 3, "estimated", "dynamic", sc, pos_0),
                    ("velocity", 3, "estimated", "dynamic", sc, vel_0),
                ]

        if expected is not None:
            with pytest.raises(ValueError, match=expected):
                scb.StateDefinition.from_components(components)
        elif case == "a_fogm_correctly_placed":
            sd = scb.StateDefinition.from_components(components)
            assert len(sd) == 3
        else:
            sd = scb.StateDefinition.from_components(components)
            assert [c[0] for c in sd] == ["a_fogm", "position", "velocity"]

    @pytest.mark.parametrize(
        "bad_tuple, match",
        [
            (["position", 3, "estimated", "dynamic", None, None], "6-tuple"),
            ((123, 3, "estimated", "dynamic", None, None), "name must be a str"),
            (("position", 3.0, "estimated", "dynamic", None, None), "dim must be an int"),
            (("position", 3, 123, "dynamic", None, None), "est and dyn must be str"),
        ],
        ids=["Not a Tuple", "Name Not Str", "Dim Not Int", "Est/Dyn Not Str"],
    )
    def test_normalize_tuple_type_errors(self, bad_tuple, match):
        """Verifies TypeError for each malformed raw component tuple shape."""
        with pytest.raises(TypeError, match=match):
            scb.StateDefinition._normalize_tuple(bad_tuple)

    def test_validate_component_wrong_physical_type(self, sc, pos_0):
        """A component whose value's physical dimension doesn't match the
        allowed state dictionary's expected type raises ValueError."""
        km, sec = scb.Units.get_units(["km", "sec"])
        bad_pos = scb.ArrayWFrame(np.array([1.0, 2.0, 3.0]), km / sec, pos_0.frame)
        with pytest.raises(ValueError, match="incorrect physical type"):
            scb.StateDefinition().position(sc, bad_pos)


def test_empty_pv(sc):
    """
    Verifies that empty_pv returns a two-component StateDefinition whose
    position and velocity values are all-NaN arrays.
    """
    sd = scb.StateDefinition.empty_pv(sc, frame=J2000)

    assert len(sd) == 2
    assert sd[0][0] == "position"
    assert sd[1][0] == "velocity"
    assert np.all(np.isnan(sd[0][5].quantity.values))
    assert np.all(np.isnan(sd[1][5].quantity.values))


@pytest.mark.parametrize(
    "clear, expected_len, expected_owner_idx",
    [(False, 4, 0), (True, 2, 0)],
    ids=["Append", "Reset"],
)
def test_extend_from_components(sc, sc2, pos_0, vel_0, clear, expected_len, expected_owner_idx):
    """
    Verifies that extend_from_components appends when clear=False and
    resets when clear=True.
    """
    sd = scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0)
    extra = [
        ("position", 3, "estimated", "dynamic", sc2, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc2, vel_0),
    ]
    sd.extend_from_components(extra, clear=clear)

    assert len(sd) == expected_len
    assert sd[expected_owner_idx][4] is (sc if not clear else sc2)


def test_from_components_and_to_components(sc, pos_0, vel_0):
    """
    Verifies round-trip: from_components builds the correct StateDefinition
    and to_components recovers the original tuple list. Also verifies
    from_legacy is equivalent to from_components.
    """
    raw = [
        ("position", 3, "estimated", "dynamic", sc, pos_0),
        ("velocity", 3, "estimated", "dynamic", sc, vel_0),
    ]
    sd = scb.StateDefinition.from_components(raw)

    assert len(sd) == 2
    assert sd[0][0] == "position"
    assert sd[1][0] == "velocity"

    recovered = sd.to_components()
    assert isinstance(recovered, list)
    assert recovered[0][0] == "position"
    assert recovered[1][0] == "velocity"

    sd_legacy = scb.StateDefinition.from_legacy(raw)
    assert sd_legacy.to_components() == recovered


class TestAsStateDefinition:
    """Groups as_state_definition's branches for SD/list input and an invalid type."""

    @pytest.mark.parametrize("input_type", ["sd", "list"], ids=["SD", "List"])
    def test_returns_same_or_builds_new(self, sc, pos_0, vel_0, input_type):
        """
        Verifies as_state_definition returns the same object for SD input
        and constructs a new one from a legacy list.
        """
        sd = scb.StateDefinition().position(sc, pos_0).velocity(sc, vel_0)

        if input_type == "sd":
            result = scb.StateDefinition.as_state_definition(sd)
            assert result is sd
        else:
            lst = [
                ("position", 3, "estimated", "dynamic", sc, pos_0),
                ("velocity", 3, "estimated", "dynamic", sc, vel_0),
            ]
            result = scb.StateDefinition.as_state_definition(lst)
            assert isinstance(result, scb.StateDefinition)
            assert len(result) == 2

    def test_invalid_type_raises(self):
        """Verifies TypeError is raised for unsupported input types."""
        with pytest.raises(TypeError):
            scb.StateDefinition.as_state_definition(42)


def test_generate_allowed_state_dictionary():
    """
    Verifies the allowed-state dictionary contains all expected component
    keys used across the codebase.
    """
    allowed = scb.StateDefinition._generate_allowed_state_dictionary()

    for key in [
        "position", "velocity", "eta_srp", "k_thrust", "dRA", "dDEC",
        "dv_man", "gm", "cnm", "snm", "range_bias", "rangerate_bias",
        "gs_delta_location_ECEF", "a_fogm", "beta_fogm",
    ]:
        assert key in allowed


class TestCustomParameterRegistration:
    """Groups register/unregister_custom_parameter behavior."""

    def test_invalid_identifier_raises(self):
        """A name that isn't a valid identifier raises ReservedComponentNameError."""
        with pytest.raises(
            scb.StateDefinition.ReservedComponentNameError, match="not a valid identifier"
        ):
            scb.StateDefinition.register_custom_parameter("bad-name!")

    def test_case_collision_raises(self):
        """
        Registering a name that differs from an already-registered custom
        parameter only by case raises CustomParameterConflictError, since
        partial-derivative keys are case-insensitive.
        """
        scb.StateDefinition.register_custom_parameter("my_custom_param", dimension=1)
        try:
            with pytest.raises(
                scb.StateDefinition.CustomParameterConflictError,
                match="differs from the already",
            ):
                scb.StateDefinition.register_custom_parameter("My_Custom_Param", dimension=1)
        finally:
            scb.StateDefinition.unregister_custom_parameter("my_custom_param")

    def test_unregister(self):
        """
        Verifies unregister_custom_parameter removes a registered component,
        and that an unknown name is silently ignored rather than raising.
        """
        scb.StateDefinition.register_custom_parameter("my_temp_param", dimension=2)
        assert "my_temp_param" in scb.StateDefinition.custom_parameters()

        scb.StateDefinition.unregister_custom_parameter("my_temp_param")
        assert "my_temp_param" not in scb.StateDefinition.custom_parameters()

        scb.StateDefinition.unregister_custom_parameter("never_registered_xyz")

# NOTE: the rest of this class' methods are tested in integration tests