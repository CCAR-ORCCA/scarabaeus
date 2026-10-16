# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
# ==================================#
#  Unit Tests for the Frame class  #
# ==================================#
import src.scarabaeus as scb
import numpy as np
import pytest

J2000, ITRF93, ECLIPJ2000, IAUEARTH = scb.Frame.generate_common_frames()
km, kg, sec, hr, day, m, rad, unitless = scb.Units.get_units(
    ["km", "kg", "sec", "hr", "day", "m", "rad", "unitless"]
)

# --------------------#
# region    Fixtures #
# --------------------#

# endregion Fixtures #
# --------------------#


# --------------#
# region Tests #
# --------------#
class TestInitialization:
    def test_init(self):
        """
        Verifies that object is constructed correctly.
        """
        Frame_1 = scb.Frame("J2000")

        assert Frame_1._origin == 0
        assert Frame_1._name == "J2000"
        assert Frame_1._frame_class == 1
        assert Frame_1._class_id == 1

        # exercise the public property getters as well as the private attributes
        assert Frame_1.origin == 0
        assert Frame_1.frame_class == 1
        assert Frame_1.class_id == 1
        assert repr(Frame_1) == "J2000 (0 - SOLAR SYSTEM BARYCENTER)"

    def test_invalid_frame_raises(self):
        """
        Verifies that constructing a Frame with an unrecognized name
        raises a ValueError.
        """
        with pytest.raises(ValueError, match="is not in the list of regonized frames"):
            scb.Frame("NOT_A_REAL_FRAME_XYZ")


def test_disp_properties(capsys):
    """
    Verifies that disp_properties prints a formatted block containing
    the frame's name, ID, origin, class, and class ID, and returns
    None.
    """
    Frame_1 = scb.Frame("J2000")
    result = Frame_1.disp_properties()

    assert result is None

    captured = capsys.readouterr()
    assert "Frame Properties" in captured.out
    assert "Frame Name:     J2000" in captured.out
    assert "Frame ID:       1" in captured.out
    assert "Origin Name:    SOLAR SYSTEM BARYCENTER" in captured.out
    assert "Origin ID:      0" in captured.out
    assert "Frame Class:    1" in captured.out
    assert "Class ID:       1" in captured.out


def test_equality_and_hash():
    """
    Verifies that separately built frames with the same SPICE frame ID
    are equal and hash alike, so they can be combined in ArrayWFrame
    arithmetic and used as set or dict keys, and that other frames and
    non-Frame objects are not equal to them.
    """
    first, second = scb.Frame("J2000"), scb.Frame("J2000")

    assert first is not second
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second, J2000}) == 1
    assert first != ITRF93
    assert first != "J2000"


class TestGetDCM:
    @pytest.mark.parametrize(
        "a, b, c, expected",
        [
            (
                J2000,
                ITRF93,
                scb.EpochArray(1096804869.182, "TDB"),
                np.array(
                    [
                        [9.7694412361129e-01, 2.1346994242815e-01, -3.2807043600831e-03],
                        [-2.1346855355350e-01, 9.7694963062890e-01, 7.7191827746788e-04],
                        [3.3698642630361e-03, -5.3793810696780e-05, 9.9999432054441e-01],
                    ]
                ),
            ),
            (
                J2000,
                J2000,
                scb.EpochArray(1096804869.182, "TDB"),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            ),
        ],
        ids=["J2000 to ITRF93", "J2000 to J2000"],
    )
    def test_dcm(self, a, b, c, expected):
        """
        Verifies that the correct DCM is returned.
        """
        assert np.allclose(a.get_DCM(a, b, c), expected)

    @pytest.mark.parametrize(
        "case",
        ["non_tdb_epoch", "non_square", "non_orthonormal"],
        ids=["Non-TDB Epoch", "Non-Square DCM", "Non-Orthonormal DCM"],
    )
    def test_edge_cases(self, mocker, case):
        """
        Verifies that get_DCM converts a non-TDB epoch to TDB before
        computing the transformation, and raises a ValueError when the
        DCM returned by SpiceManager is not square or not orthonormal.

        A real non-TDB EpochArray is avoided for the conversion case since
        converting one to a different moment in UTC lands outside the
        loaded PCK's coverage window; mocking isolates just the conversion
        behavior instead.
        """
        match case:
            case "non_tdb_epoch":
                tdb_epoch = scb.EpochArray(1096804869.182, "TDB")
                non_tdb_epoch = mocker.MagicMock()
                non_tdb_epoch.system = "UTC"
                non_tdb_epoch.to.return_value = tdb_epoch

                expected = np.array(
                    [
                        [9.7694412361129e-01, 2.1346994242815e-01, -3.2807043600831e-03],
                        [-2.1346855355350e-01, 9.7694963062890e-01, 7.7191827746788e-04],
                        [3.3698642630361e-03, -5.3793810696780e-05, 9.9999432054441e-01],
                    ]
                )

                dcm = scb.Frame.get_DCM(J2000, ITRF93, non_tdb_epoch)

                non_tdb_epoch.to.assert_called_once_with(sys="TDB")
                assert np.allclose(dcm, expected)

            case "non_square":
                mocker.patch.object(
                    scb.SpiceManager, "get_xfrm", return_value=np.ones((3, 2))
                )
                with pytest.raises(ValueError, match="should be a square Matrix"):
                    scb.Frame.get_DCM(J2000, ITRF93, scb.EpochArray(1096804869.182, "TDB"))

            case "non_orthonormal":
                mocker.patch.object(
                    scb.SpiceManager, "get_xfrm", return_value=np.eye(3) * 2
                )
                with pytest.raises(ValueError, match="should be an Orthonormal Matrix"):
                    scb.Frame.get_DCM(J2000, ITRF93, scb.EpochArray(1096804869.182, "TDB"))


class TestRelativeTransforms:
    @pytest.mark.parametrize(
        "a, b, c, expected",
        [
            (
                J2000,
                ITRF93,
                scb.EpochArray(1096804869.182, "TDB"),
                scb.ArrayWUnits(
                    [1.4824386025344e08, -7.7148149059754e06, 1.0964030655836e07], km
                ),
            ),
            (
                J2000,
                J2000,
                scb.EpochArray(1096804869.182, "TDB"),
                scb.ArrayWUnits([0, 0, 0], km),
            ),
        ],
        ids=["J2000 to ITRF93", "J2000 to J2000"],
    )
    def test_position(self, a, b, c, expected):
        """
        Verifies that the correct translation is returned.
        """
        assert np.allclose((a.get_relative_pos(a, b, c)).quantity.values, expected.values)
        assert (a.get_relative_pos(a, b, c)).quantity.units == expected.units

    @pytest.mark.parametrize(
        "a, b, c, expected",
        [
            (
                J2000,
                ITRF93,
                scb.EpochArray(1096804869.182, "TDB"),
                scb.ArrayWUnits(
                    [-562.5883813609161, -10782.705827176937, 11.5618578354427], km / sec
                ),
            ),
            (
                J2000,
                J2000,
                scb.EpochArray(1096804869.182, "TDB"),
                scb.ArrayWUnits([0, 0, 0], km / sec),
            ),
        ],
        ids=["J2000 to ITRF93", "J2000 to J2000"],
    )
    def test_velocity(self, a, b, c, expected):
        """
        Verifies that the correct translation velocity is returned.
        """
        assert np.allclose((a.get_relative_vel(a, b, c)).quantity.values, expected.values)
        assert (a.get_relative_vel(a, b, c)).quantity.units == expected.units

    @pytest.mark.parametrize(
        "a, b, c, expected",
        [
            (
                J2000,
                ITRF93,
                scb.EpochArray(1096804869.182, "TDB"),
                [
                    [
                        9.7694412361129e-01,
                        2.1346994242815e-01,
                        -3.2807043600831e-03,
                        1.4824386025344e08,
                    ],
                    [
                        -2.1346855355350e-01,
                        9.7694963062890e-01,
                        7.7191827746788e-04,
                        -7.7148149059754e06,
                    ],
                    [
                        3.3698642630361e-03,
                        -5.3793810696780e-05,
                        9.9999432054441e-01,
                        1.0964030655836e07,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ),
            (
                J2000,
                J2000,
                scb.EpochArray(1096804869.182, "TDB"),
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ),
        ],
        ids=["J2000 to ITRF93", "J2000 to J2000"],
    )
    def test_transformation(self, a, b, c, expected):
        """
        Verifies that the correct transformation is returned.
        """
        assert np.allclose(a.get_transformation(a, b, c)[0].values, np.array(expected))

    @pytest.mark.parametrize(
        "method_name",
        ["get_relative_pos", "get_relative_vel", "get_transformation"],
    )
    def test_non_tdb_epoch_raises(self, method_name):
        """
        Verifies that get_relative_pos, get_relative_vel, and
        get_transformation each raise a ValueError when given a non-TDB
        epoch.
        """
        non_tdb_epoch = scb.EpochArray("2004-10-03T00:00:00", "UTC")

        method = getattr(scb.Frame, method_name)
        with pytest.raises(ValueError, match="must be in TDB"):
            method(J2000, ITRF93, non_tdb_epoch)

    def test_transformation_multi_epoch(self):
        """
        Verifies that get_transformation iterates over each epoch and
        returns one 4x4 transformation matrix per epoch when the given
        EpochArray contains more than one epoch. Uses the J2000-to-J2000
        identity transform, since ITRF93's loaded PCK only covers a
        single instant in this environment and a second, different epoch
        would fail to resolve the Earth-fixed frame's orientation.
        """
        epoch = scb.EpochArray([1096804869.182, 1096804869.182 + 3600], "TDB")

        T = scb.Frame.get_transformation(J2000, J2000, epoch)

        assert len(T) == 2
        identity = np.eye(4)
        for t in T:
            assert np.allclose(t.values, identity)


@pytest.mark.parametrize(
    "method_name, spice_method_name, args",
    [
        ("write_pck_frame", "_write_pck_frame", ("file.pck", "EROS_FIXED", 2, 2000433)),
        (
            "write_ck_frame",
            "_write_ck_frame",
            ("file.tf", "MGS_SPACECRAFT", -94000, -94, -94, -94),
        ),
        (
            "write_tk_frame",
            "_write_tk_frame",
            (
                "file.tf",
                "MARS_FIXED",
                1400499,
                499,
                "IAU_MARS",
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ),
        ),
    ],
)
def test_write_frame_delegates_to_spice_manager(
    mocker, method_name, spice_method_name, args
):
    """
    Verifies that write_pck_frame, write_ck_frame, and write_tk_frame
    each delegate to their corresponding SpiceManager private method
    with the given arguments unchanged.
    """
    mock = mocker.patch.object(scb.SpiceManager, spice_method_name)

    method = getattr(scb.Frame, method_name)
    method(*args)

    mock.assert_called_once_with(*args)
