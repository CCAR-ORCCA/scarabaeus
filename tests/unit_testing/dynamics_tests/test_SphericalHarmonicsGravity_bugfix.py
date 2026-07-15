# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Regression tests for two SphericalHarmonicsGravity bugs.

Bug 1 (_compute_bnm): the ``if not self.normalized:`` branch indexed
``bnm_ext_real``/``bnm_ext_imag`` with tuple subscripts (e.g. ``[n, n]``) while
those objects were Python lists, raising
``TypeError: list indices must be integers or slices, not tuple``. The fix
pre-allocates them as ``(N, N)`` numpy arrays. This test checks the unnormalized
path runs AND yields the same acceleration as the equivalent normalized field
(J2-only: normalized C20 = -J2/sqrt(5) == unnormalized C20 = -J2).

Bug 2 (_rf_transform / the three ``_jacobian_*_transform_rf`` helpers): the
body-fixed frame name was obtained via ``self._body["ref_name"]`` /
``self.SPH_body["ref_name"]`` unconditionally, which raises
``TypeError: 'CelestialBody' object is not subscriptable`` when the force model
is built the normal, documented way with a ``CelestialBody`` (rather than a
dict). This broke the coefficient-partial path (``_compute_partial_by_C``),
i.e. exactly what is needed to estimate J2/C20. This test builds the model with
a ``CelestialBody`` and exercises that partial path.

This module intentionally OVERRIDES the session ``setup`` fixture from the root
conftest: that fixture furnshes a metakernel referencing locked kernels
(de430.bsp, cas00084.tsc, ...) that are not present in this checkout. Here we
load only the (present) supplementary kernels needed for an Earth J2 evaluation.
"""
import os
import glob
import json
import math

import numpy as np
import pytest

import src.scarabaeus as scb
from scarabaeus.dynamics.SphericalHarmonicsGravity import SphericalHarmonicsGravity

_SCB_ROOT = os.path.dirname(scb.__file__)  # .../src/scarabaeus
_KROOT = os.path.join(
    os.path.dirname(os.path.dirname(_SCB_ROOT)),  # repo root
    "tutorials/supplementary/supp_data/kernels/locked",
)

_HAVE_KERNELS = os.path.isdir(_KROOT) and bool(
    glob.glob(os.path.join(_KROOT, "spk/de432s.bsp"))
)

pytestmark = pytest.mark.skipif(
    not _HAVE_KERNELS,
    reason=f"supplementary SPICE kernels not found under {_KROOT}",
)


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Override the root conftest session ``setup`` (locked-kernel metakernel is
    unavailable here); load the present supplementary kernels per-file instead."""
    scb.SpiceManager.clear_kernels()
    for pat in [
        "lsk/*.tls",
        "spk/de432s.bsp",
        "pck/*.tpc",
        "spk/earthstns_fx_201023.bsp",
        "spk/earth_200101_990628_predict.bpc",
        "spk/earth_topo_201023.tf",
    ]:
        for k in glob.glob(os.path.join(_KROOT, pat)):
            scb.SpiceManager.load_kernel_from_mkfile(k)
    yield


_J2 = 0.00108262668355  # Earth J2 (unnormalized -C20)


def _cs_file(tmp_path, normalized):
    """Write a pure-zonal J2 coefficient file (normalized or not)."""
    c20 = (-_J2 / math.sqrt(5.0)) if normalized else (-_J2)
    data = {
        "BodyName": "Earth",
        "ReferenceRadius": 6378.1363,
        "GravitationalParameter": 398600.4415,
        "Normalized": normalized,
        "MaxDegree": 2,
        "Coefficients": [
            {"degree": 0, "order": 0, "Cnm": 1.0, "Snm": 0.0},
            {"degree": 1, "order": 0, "Cnm": 0.0, "Snm": 0.0},
            {"degree": 1, "order": 1, "Cnm": 0.0, "Snm": 0.0},
            {"degree": 2, "order": 0, "Cnm": c20, "Snm": 0.0},
            {"degree": 2, "order": 1, "Cnm": 0.0, "Snm": 0.0},
            {"degree": 2, "order": 2, "Cnm": 0.0, "Snm": 0.0},
        ],
    }
    p = tmp_path / ("earth_j2_%s.json" % ("norm" if normalized else "unnorm"))
    p.write_text(json.dumps(data))
    return str(p)


def _state_vector():
    km, sec, kg = scb.Units.get_units(["km", "sec", "kg"])
    J2000 = scb.Frame("J2000")
    earth = scb.CelestialBody.from_constants("EARTH")
    X0 = np.array(
        [-2629.215543505899, 7931.076007348140, 4727.077214458598,
         -4.210803334341272, -3.864752170605402, 4.142207277141169]
    )
    sc = scb.Spacecraft("SC", -9201, tot_mass=scb.ArrayWUnits(500.0, kg))
    st = scb.StateDefinition.from_components([
        ("position", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(X0[:3].copy(), km, J2000)),
        ("velocity", 3, "estimated", "dynamic", sc, scb.ArrayWFrame(X0[3:].copy(), km / sec, J2000)),
    ])
    # Epoch inside the supplementary Earth binary-PCK (ITRF93) coverage window.
    et = scb.SpiceManager.str2et("2024-01-01 00:00:00 TDB")
    sv = scb.StateArray(epoch=scb.EpochArray(np.array([et]), sys="TDB"), origin=earth, state=st)
    return sv, earth, J2000, X0[:3].copy(), et


def _make(cs, norm_flag, body, sv, J2000):
    return SphericalHarmonicsGravity(
        sph_harm_order=2, sph_harm_cs_file=cs, sph_harm_body=body,
        state_vector=sv, sph_harm_norm_flag=norm_flag, base_frame=J2000,
    )


def test_unnormalized_bnm_matches_normalized(tmp_path):
    """Bug 1: unnormalized _compute_bnm no longer raises, and the unnormalized J2
    acceleration equals the normalized J2 acceleration for the same field."""
    sv, earth, J2000, pos, epoch = _state_vector()
    # dict body isolates Bug 1 from Bug 2
    earth_dict = {"spice_name": "EARTH", "ref_name": earth.base_frame, "SPICE_ID": 399}

    g_un = _make(_cs_file(tmp_path, False), False, earth_dict, sv, J2000)
    g_no = _make(_cs_file(tmp_path, True), True, earth_dict, sv, J2000)

    acc_un = np.asarray(g_un.compute_acceleration(pos, None, epoch, earth_dict), float).ravel()
    acc_no = np.asarray(g_no.compute_acceleration(pos, None, epoch, earth_dict), float).ravel()

    assert acc_un.shape == (3,)
    assert np.linalg.norm(acc_un) > 0.0
    np.testing.assert_allclose(acc_un, acc_no, rtol=0, atol=1e-15)


def test_celestialbody_partial_by_C_runs(tmp_path):
    """Bug 2: the coefficient-partial path runs when SPH_body is a CelestialBody
    (the normal API) rather than a dict."""
    sv, earth, J2000, pos, epoch = _state_vector()
    g = _make(_cs_file(tmp_path, True), True, earth, sv, J2000)  # CelestialBody, not dict
    part = g._compute_partial_by_C(pos, epoch)  # would raise TypeError before the fix
    part = np.asarray(part)
    assert part.shape[0] == 3
    assert np.all(np.isfinite(part))
