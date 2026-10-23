# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# =======================================#
#  Unit Tests for the Dimensions class  #
# =======================================#
import src.scarabaeus as scb
import numpy as np
import pytest

# --------------------#
# region    Fixtures #
# --------------------#
_length = scb.Dimensions([0, 1, 0, 0])

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
def test_initialization():
    """
    Verifies that object is constructed correctly.
    """
    Dim = scb.Dimensions([0, 1, -1, 0])
    assert np.array_equal(Dim.powers, [0, 1, -1, 0])
    assert Dim.name == "Velocity"
    assert repr(Dim) == "Velocity"


@pytest.mark.parametrize(
    "dim_pwrs, error",
    [
        ("not a list", ValueError),
        ([1, 1, 1], ValueError),
        (["a", "b", "c", "d"], TypeError),
    ],
    ids=["bad type", "bad shape", "bad dtype"],
)
def test_powers_setter_errors(dim_pwrs, error):
    """
    Verifies that constructing a Dimensions object with invalid dim_pwrs
    raises the appropriate exception for a bad type, a bad shape, or a
    non-numeric dtype.
    """
    with pytest.raises(error):
        scb.Dimensions(dim_pwrs)


class TestComparisonOperators:
    @pytest.mark.parametrize(
        "a, b, case",
        [
            (scb.Dimensions([0, 1, 0, 0]), scb.Dimensions([0, 1, 0, 0]), "a = b"),
            (scb.Dimensions([1, 1, 0, 0]), scb.Dimensions([1, 1, 0, 0]), "a = b"),
            (_length, _length, "a = b"),
            (scb.Dimensions([0, 1, 1, 0]), scb.Dimensions([0, 1, 0, 0]), "a != b"),
            (scb.Dimensions([0, 1, 0, 0]), "not a dimension", "error"),
        ],
        ids=[
            "same dim", "same dim mixed unit", "same object", "different dim",
            "non-Dimensions comparison",
        ],
    )
    def test_equality(self, a, b, case):
        """
        Verifies that the equality operator functions as expected.
        """
        match case:
            case "a = b":
                # ensure that a is to b
                assert a == b

            case "a != b":
                assert a != b

            case "error":
                # can't compare a Dimensions object to a non-Dimensions object
                with pytest.raises(TypeError):
                    a == b

    @pytest.mark.parametrize(
        "a, b, case",
        [
            (scb.Dimensions([0, 1, 0, 0]), scb.Dimensions([0, 3, 0, 0]), "a != b"),
            (scb.Dimensions([1, 2, 0, 0]), scb.Dimensions([1, 1, 0, 0]), "a != b"),
        ],
        ids=["same unit", "same unit mixed"],
    )
    def test_inequality(self, a, b, case):
        """
        Verifies that the inequality operator functions as expected.
        """
        match case:
            case "a != b":
                # ensure that a is to b
                assert a != b

            case "error":
                # can't examine inequalities between two different units
                with pytest.raises(Exception):
                    a != b


@pytest.mark.parametrize(
    "dim, expected_name",
    [
        (scb.Dimensions([2, 0, 0, 0]), "Square Mass"),
        (scb.Dimensions([0, 0, 3, 0]), "Cubic Time"),
        (scb.Dimensions([0, 0, 0, 5]), "Angle to the 5th"),
        (scb.Dimensions([-1, 0, 0, 0]), "per Mass"),
        (scb.Dimensions([1, 0, 0, 1]), "Mass times Angle"),
    ],
    ids=["square", "cubic", "higher order", "negative power", "multiple positive powers"],
)
def test_dim_name_power_order(dim, expected_name):
    """
    Verifies that unnamed dimensions are described using the correct
    power order wording (square, cubic, or Nth power) based on the
    magnitude of their dimensional powers.
    """
    assert dim.name == expected_name


@pytest.mark.parametrize(
    "name, powers, case",
    [
        ("Test Dim", [1, 1, 1, 1], "success"),
        ("Test Dim Array", np.array([2, 0, 0, 0]), "success"),
        ("Bad Type", "not a list", "bad type"),
        ("Bad Shape", [1, 1, 1], "bad shape"),
        ("Bad Dtype", [1.5, 0, 0, 0], "bad dtype"),
        ("Duplicate Powers", [1, 1, -2, 0], "duplicate powers"),  # matches Force
        ("Force", [5, 5, 5, 5], "duplicate name"),
    ],
    ids=[
        "success list",
        "success ndarray",
        "bad type",
        "bad shape",
        "bad dtype",
        "duplicate powers",
        "duplicate name",
    ],
)
def test_def_new_named_dim(name, powers, case):
    """
    Verifies that def_new_named_dim registers new named dimensions
    correctly and raises the appropriate exception for invalid or
    duplicate input.
    """
    match case:
        case "success":
            scb.Dimensions.def_new_named_dim(name, powers)
            try:
                dim = scb.Dimensions(list(powers))
                assert dim.name == name
            finally:
                scb.Dimensions._named_dimensions.pop(name, None)

        case "bad type" | "bad shape" | "duplicate powers" | "duplicate name":
            with pytest.raises(ValueError):
                scb.Dimensions.def_new_named_dim(name, powers)

        case "bad dtype":
            with pytest.raises(TypeError):
                scb.Dimensions.def_new_named_dim(name, powers)


@pytest.mark.parametrize("case", ["even", "odd"])
def test_disp_named_dims(capsys, case):
    """
    Verifies that disp_named_dims prints a formatted table containing
    every named dimension in the library without raising, and returns
    nothing -- for both an even number of named dimensions and an odd
    number (which leaves the final row's second column unpaired).
    """
    added_name = None
    if case == "odd":
        added_name = "Test Odd Dim"
        scb.Dimensions.def_new_named_dim(added_name, [3, 3, 3, 3])

    try:
        result = scb.Dimensions.disp_named_dims()
        captured = capsys.readouterr()

        assert result is None
        assert "List of Named Dimensions" in captured.out
        for name in scb.Dimensions._named_dimensions:
            assert name in captured.out
    finally:
        if added_name:
            scb.Dimensions._named_dimensions.pop(added_name, None)
