.. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
.. SPDX-License-Identifier: ISC

.. meta::
    :description lang=en:
        Ground stations, body orientation, surface tides and their estimation in Scarabaeus.
    :keywords:
        Scarabaeus, ground station, Earth orientation parameters, EOP, solid tide, station estimation

.. _orientation-station-estimation:

.. create a gold color role
.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold

=====================================
:gold:`Stations and Body Orientation`
=====================================
Last revised by G. Fereoli on 2026 OCT 11

A :class:`~scarabaeus.GroundStation` sits on a rotating body. Where it is at an epoch
depends on three things: its coordinates, the orientation of the body-fixed frame, and
the surface displacements (tides) acting on it. Each can be modelled, and most can be
estimated.

Body orientation
----------------
All orientation models live in :class:`~scarabaeus.BodyOrientation`.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Constructor
     - Purpose
   * - ``BodyOrientation(rotation_vector, body_fixed_frame)``
     - Small rigid rotation of a station, in radians, on any body.
   * - ``earth_correction(dxp, dyp, dut1)``
     - The same correction expressed as Earth polar motion and UT1.
   * - ``celestial_pole_correction(dX, dY)``
     - Celestial pole offset (nutation correction).
   * - ``earth_eop(pole_x, pole_y, tai_minus_ut1, model=...)``
     - Absolute Earth orientation, written to an ITRF93 PCK.
   * - ``from_rotation(rotation, frame_class_id, base_frame=...)``
     - Absolute PCK for any body from a user rotation law.

``earth_eop`` takes ``model="EOP1"`` (ITRF93 legacy: IAU1976/1980, equinox-based) or
``model="EOP2"`` (IERS 2010: IAU2006/2000A, CIO/ERA). Both PCKs use class ID 3000, so
existing ITRF93 station SPKs follow the selected orientation.

.. code-block:: python

    scb.SpiceManager.load_kernel("data/kernels/locked/lsk/naif0012.tls")
    orientation = scb.BodyOrientation.earth_eop(
        pole_x=np.deg2rad(6.4125e-5), pole_y=np.deg2rad(1.341694444444444e-4),
        tai_minus_ut1=37.00446633384733, model="EOP1")
    start = scb.SpiceManager.utc2et("2029 APR 25 00:00:00 UTC")
    path = orientation.write_pck("earth_absolute.bpc", start, start + 86400.)
    scb.SpiceManager.load_kernel(str(path))   # load after any other Earth PCK

The PCK must cover every transmit and receive epoch, including Doppler count
intervals.

**Signs.** A correction rotates the station by
:math:`\boldsymbol\varepsilon = (-\Delta y_p,\ -\Delta x_p,\ \omega_\oplus\Delta UT1)`,
with :math:`\Delta\mathbf r = \boldsymbol\varepsilon\times\mathbf r`. An estimated
:math:`\Delta UT1` maps to the absolute input as :math:`\Delta(TAI-UT1) = -\Delta UT1`.

**Time scale.** ``ut1_time_scale="TAI"`` (default) is correct. ``"TDB"`` skips TDB−TT
(up to 1.66 ms, about 0.5 m at the equator). Use it only to reproduce a tool that
makes the same omission.

Station coordinates
-------------------
By default a station is read from its SPK, which also carries plate motion. A station
can instead be given from a table:

.. code-block:: python

    km, deg = scb.Units.get_units(["km", "deg"])
    station = scb.GroundStation.from_cylindrical(
        "DSS-24", 399024, body_fixed_frame="ITRF93",
        longitude=scb.ArrayWUnits(-116.8747939, deg),
        cylindrical_radius=scb.ArrayWUnits(5209.482541999999, km),
        z=scb.ArrayWUnits(3669.24231, km))

``body_fixed_position=`` accepts Cartesian coordinates instead. A table station has no
SPK, so it is queried through ``station.nominal_state``, and plate motion is the
user's to apply. :class:`~scarabaeus.DopplerReal` and
:class:`~scarabaeus.SequentialRangingReal` need a station SPK and raise
``Measurement.StationKernelRequiredError`` without one.

On another body, give the station its central body and frame:

.. code-block:: python

    station = scb.GroundStation("MARS_STATION", 499001,
                                central_body="MARS", body_fixed_frame="IAU_MARS")
    station.body_orientation = scb.BodyOrientation(
        rotation_vector=[1e-7, -2e-7, 3e-7], body_fixed_frame="IAU_MARS")

Surface displacement
--------------------
Tides are part of where the station *is*, so they enter ``nominal_state`` rather than
an estimated correction:

.. code-block:: python

    station.surface_displacement = [scb.SolidTide(),
                                    scb.OceanLoading(amplitudes, phases),
                                    scb.PoleTide(pole_x=xp, pole_y=yp)]

- :class:`~scarabaeus.SolidTide`: IERS degree-2 in-phase, up to about 0.4 m. Keep
  ``remove_permanent_tide=False`` for tide-free coordinates such as ITRF and DSN
  tables.
- The same object also perturbs the gravity field, through :math:`k_2` instead of
  :math:`h_2, l_2`: ``sph_harm.solid_tide = tide``.
- Other bodies have no built-in Love numbers and raise ``UnknownTidalResponseError``.
  Give them explicitly, for example
  ``scb.SolidTide(central_body="MARS", love_h2=0.17, shida_l2=0.02, love_k={(2, 0): 0.169}, tide_bodies=("SUN", "PHOBOS"))``.
- Omitted: degree 3 (~2 mm), out-of-phase terms (a few mm), step-2 corrections
  (up to ~13 mm), atmospheric loading.

Estimable parameters
--------------------
.. list-table::
   :header-rows: 1
   :widths: 40 14 46

   * - Name
     - Units
     - Meaning
   * - ``gs_delta_location_ECEF_<tag>``
     - length (3)
     - Cartesian station offset in its body-fixed frame
   * - ``gs_delta_location_CYL_<tag>``
     - length (3)
     - :math:`(\Delta\rho,\ \rho\Delta\lambda,\ \Delta z)`; set
       ``sigma_east = rho * sigma_lon``
   * - ``eop_delta_pole_x``, ``eop_delta_pole_y``
     - angle
     - Polar motion correction
   * - ``eop_delta_ut1``
     - time
     - UT1 correction
   * - ``eop_delta_celestial_pole_x``, ``_y``
     - angle
     - Celestial pole offset correction
   * - ``body_orientation_delta``
     - angle (3)
     - Rigid rotation of the station frame, any body
   * - ``pole_ra``, ``pole_dec``, ``pole_w0``
     - angle
     - IAU pole and prime meridian of the central body
   * - ``pole_pm``
     - deg/day
     - IAU prime-meridian rate
   * - ``love_h2``, ``shida_l2``
     - —
     - Solid-tide response of the body
   * - ``tropo_zenith_delay_<tag>``, ``iono_zenith_delay_<tag>``
     - length
     - Zenith delays (Chao wet mapping; ionosphere at 2295 MHz, scaled by
       :math:`(2295/f)^2`). Real radiometric models only.

The IAU pole parameters orient the central body's gravity field, its optical shape and
its stations together. Each estimate *replaces* its secular term, so start it from the
secular value at the arc epoch, not from the PCK constant.

.. code-block:: python

    km, sec, rad = scb.Units.get_units(["km", "sec", "rad"])
    itrf, earth = scb.Frame("ITRF93"), scb.Body("EARTH", 399)
    gs = scb.GroundStation("DSS-24")
    state_def = scb.StateDefinition()
    state_def.param("gs_delta_location_CYL_DSS24", gs,
                    scb.ArrayWFrame(np.zeros(3), km, itrf), dynamics="static")
    for name, unit in [("eop_delta_pole_x", rad), ("eop_delta_pole_y", rad),
                       ("eop_delta_ut1", sec)]:
        state_def.param(name, earth, scb.ArrayWFrame(np.array(0.), unit, itrf),
                        dynamics="static")
    reference = scb.StateArray(scb.EpochArray(start, sys="TDB"), earth, state_def)
    model = scb.RangeIdeal("DSS24 range", gs, sigma=scb.ArrayWUnits(.001, km))
    model.update_reference_state(reference)

A parameter owned by the central body applies to all its stations; one owned by a
station applies to that station only.

**Partials** are analytical. The geometric models use
:math:`\partial h/\partial p = -[\partial h/\partial\mathbf r,\ \partial h/\partial\mathbf v]\,\partial\mathbf X_{station}/\partial p`.
The light-time models weight the station Jacobian at both ends of the light path,
:math:`\mathbf g_3\,\partial\mathbf r/\partial p(t_3) + \mathbf g_1\,\partial\mathbf r/\partial p(t_1)`.
``partials_method="fd"`` uses ``eps_dict``, falling back to
``Measurement.station_eps_defaults``. The camera model
(:class:`~scarabaeus.CentroidingIdeal`) has no station, so its station columns are zero.

.. warning::

    Station coordinates and a rigid orientation are strongly correlated: a rotation
    can be absorbed by shifting the stations. Constrain the station priors before
    solving for both.

References
----------
- `IERS Conventions, Chapter 5 <https://iers-conventions.obspm.fr/content/chapter5/icc5.pdf>`_
- `ERFA <https://github.com/liberfa/erfa>`_ (``pyerfa>=2.0``)
- NAIF `pckw02_c <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pckw02_c.html>`_
  and `spkcvo_c <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkcvo_c.html>`_
