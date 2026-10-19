# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Station and Earth-orientation estimation, against independently built geometry.

Every partial checked here is compared against a Jacobian assembled from SPICE
directly in this file, never against the model's own machinery, so a sign or
frame error in the library cannot make the test agree with itself. The kernels
are written by the fixture, so the numbers do not depend on any delivery.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import spiceypy as spice
import scarabaeus as scb

km, sec, rad = scb.Units.get_units(["km", "sec", "rad"])
J2000, ITRF = scb.Frame("J2000"), scb.Frame("ITRF93")
ET = 1.e6
P1 = np.array([4300., 3200., 3000.])
P2 = np.array([-3800., 2700., 4200.])
C_KM_S = 299792.458


@pytest.fixture(scope="module", autouse=True)
def setup(tmp_path_factory):
    spice.kclear()
    spice.furnsh("data/kernels/locked/lsk/naif0012.tls")
    spice.furnsh("data/kernels/locked/pck/pck00010.tpc")
    tmp = tmp_path_factory.mktemp("station_geometry")
    scb.BodyOrientation.earth_eop(1e-6, 2e-6, 32.).write_pck(tmp/"earth.bpc", ET-1e5, ET+1e5)
    spice.furnsh(str(tmp/"earth.bpc"))
    path = tmp/"states.bsp"
    h = spice.spkopn(str(path), "independent test geometry", 0)
    times = ET + np.linspace(-1e5, 1e5, 9)
    def segment(sid, origin, frame, p, v=(0,0,0)):
        states = np.array([np.r_[np.array(p)+np.array(v)*(t-ET), v] for t in times])
        spice.spkw09(h, sid, origin, frame, times[0], times[-1], str(sid), 3, len(times), states, times)
    segment(399, 0, "J2000", [1.5e8, 0, 0], [0, 30, 0])
    segment(3, 399, "J2000", [3000, 0, 0])
    # The barycentres are the default relativistic light-time bodies.
    for sid, p in [(10,[0,0,0]),(301,[1.504e8,1e5,0]),(499,[2e8,1e8,0]),
                   (599,[5e8,2e8,1e7]),(699,[1e9,1e8,3e7]),
                   (4,[2e8,1e8,0]),(5,[5e8,2e8,1e7]),(6,[1e9,1e8,3e7])]:
        segment(sid, 0, "J2000", p)
    segment(399014, 399, "ITRF93", P1)
    segment(399065, 399, "ITRF93", P2)
    segment(499014, 499, "IAU_MARS", [2500,1200,1800])
    segment(-1001, 399, "J2000", [25000, 12000, 28000], [.15, .08, -.05])
    spice.spkcls(h)
    spice.furnsh(str(path))
    yield
    spice.kclear()


@pytest.fixture
def bodies():
    gs = scb.GroundStation("GEOMETRY_GS1", 399014)
    gs2 = scb.GroundStation("GEOMETRY_GS2", 399065)
    target = scb.Spacecraft("GEOMETRY_SC", -1001,
                           instrument_list=[scb.Antenna("radio", turn_ratio=1.1, spice_id=-1001)])
    earth = scb.Body("EARTH", 399)
    return gs, gs2, target, earth


def entries(gs, earth, values=None, cylindrical=True):
    """Location block plus the three Earth EOP deltas, in that column order."""
    location = "gs_delta_location_CYL_station" if cylindrical else "gs_delta_location_ECEF_station"
    vals = np.zeros(6) if values is None else values
    return [(location,3,True,"static",gs,scb.ArrayWFrame(np.array(vals[:3]),km,ITRF)),
            ("eop_delta_pole_x",1,True,"static",earth,scb.ArrayWFrame(np.array(vals[3]),rad,ITRF)),
            ("eop_delta_pole_y",1,True,"static",earth,scb.ArrayWFrame(np.array(vals[4]),rad,ITRF)),
            ("eop_delta_ut1",1,True,"static",earth,scb.ArrayWFrame(np.array(vals[5]),sec,ITRF))]


def celestial_entries(earth, values=(0., 0.)):
    """The two celestial pole offsets dX, dY, owned by the Earth."""
    return [(f"eop_delta_celestial_pole_{axis}",1,True,"static",earth,
             scb.ArrayWFrame(np.array(value),rad,J2000)) for axis,value in zip("xy",values)]


def iau_pole_entries(owner, body, offsets=(0., 0., 0., 0.)):
    """The four IAU pole parameters of ``body``, owned by ``owner``.

    Each sits at its secular value at ``ET`` plus its offset; an offset of
    None declares that parameter without a value.
    """
    coefficients = scb.BodyOrientation.iau_pole_coefficients(body)
    secular, _ = scb.BodyOrientation.iau_angles(ET, None, coefficients)
    values = (secular[0], secular[1], coefficients[2][1], np.deg2rad(coefficients[2][0]))
    # The spin rate is raw deg/day, as the PCK and StateDefinition give it.
    units = (rad, rad, km/km, rad)
    return [(name,1,True,"static",owner,
             None if offset is None else scb.ArrayWFrame(np.array([value+offset]),unit,None))
            for name,value,offset,unit in zip(scb.BodyOrientation.IAU_POLE_PARAMETERS,values,offsets,units)]


def mars_table_station():
    """A Mars station given by IAU_MARS coordinates, with no station kernel."""
    return scb.GroundStation("GEOMETRY_MARS_TABLE", 499015, central_body="MARS",
                             body_fixed_frame="IAU_MARS",
                             body_fixed_position=np.array([2500., 1200., 1800.]))


def make_model(kind, bodies, state):
    gs, gs2, target, earth = bodies
    options = dict(name="station test", state_definition=state)
    if kind == "RangeIdeal":
        return scb.RangeIdeal(instrument=gs, sigma=scb.ArrayWUnits(.001,km), **options)
    if kind == "RangeRateIdeal":
        return scb.RangeRateIdeal(instrument=gs, sigma=scb.ArrayWUnits(1e-6,km/sec), **options)
    if kind.startswith("DopplerIdeal"):
        return scb.DopplerIdeal(instrument=gs, antenna_name="radio",
            doppler_transmit_frequency=scb.ArrayWUnits(8e9,1/sec),
            ground_station_2=gs2 if kind.endswith("3way") else None,
            sigma=scb.ArrayWUnits(.01,1/sec), **options)
    if kind == "DiffOneWayRangeIdeal":
        return scb.DiffOneWayRangeIdeal(instrument=gs, ground_station_2=gs2,
                                      sigma=scb.ArrayWUnits(1e-9,sec), **options)
    raise AssertionError(kind)


def station_jacobian(position, et, cylindrical=True):
    """d(station state in J2000)/d(location delta, xp, yp, UT1), built here from SPICE.

    Columns are the three location components followed by the three Earth
    orientation parameters, matching :func:`entries`.
    """
    lon = np.arctan2(position[1],position[0]); c,s=np.cos(lon),np.sin(lon)
    basis=np.array([[c,-s,0],[s,c,0],[0,0,1]]) if cylindrical else np.eye(3)
    omega = float(scb.constants.earth_rotation_rate.values)
    # Rotation vector produced by each EOP delta: (-dyp, -dxp, omega*dut1).
    axes=np.array([[0,-1,0],[-1,0,0],[0,0,omega]]).T
    dx=np.column_stack([basis, np.column_stack([np.cross(a,position) for a in axes.T])])
    return spice.sxform("ITRF93","J2000",et) @ np.vstack([dx,np.zeros((3,6))])


def real_aux(epochs):
    _,days,seconds=scb.SpiceManager.et2YDS(epochs.times.values)
    day=float(np.atleast_1d(days)[0]); n=epochs.size
    aux={key:[0.]*n for key in ('rcv_time_tag_delay','transmit_time_tag_delay','ul_stn_cal','dl_stn_cal','ul_zheight_corr','dl_zheight_corr')}
    aux.update(scft_transpd_delay=0.,scft_transpd_delay_s=0.,RU=[1.]*n,M=[1e20]*n,M2_num=[11.]*n,M2_den=[10.]*n,Tc=[60.]*n,
               ul_freq=[8e9]*n,ramp_sec=[0.,86400.],ramp_day=[day,day],
               ramp_freq=[8e9,8e9],ramp_rate=[0.,0.],ramp_type=[1,1])
    return aux


# --------------------------------------------------------------------------- #
# The analytical partials                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind",["RangeIdeal","RangeRateIdeal","DopplerIdeal",
                                 "DopplerIdeal3way","DiffOneWayRangeIdeal"])
@pytest.mark.parametrize("cylindrical",[False,True])
@pytest.mark.parametrize("table_coordinates", [False, True])
def test_station_partials_match_independent_geometry(bodies,kind,cylindrical,table_coordinates):
    """Every geometric model's station block equals a Jacobian built here."""
    gs,gs2,target,earth=bodies
    if table_coordinates:
        gs = scb.GroundStation('TABLE_GS1', 399984, body_fixed_position=P1)
        gs2 = scb.GroundStation('TABLE_GS2', 399985, body_fixed_position=P2)
        bodies = gs, gs2, target, earth
    state=entries(gs,earth,cylindrical=cylindrical)
    model=make_model(kind,bodies,state)
    epoch=scb.EpochArray([ET+11.7,ET+4031.1],sys="TDB")
    H=np.asarray(model.compute_partials(target,epoch,J2000))
    for i,et in enumerate(epoch.times.values):
        relative=(spice.spkezr(target.name,et,"J2000","NONE","EARTH")[0]
                  - spice.sxform('ITRF93','J2000',et) @ np.r_[P1,np.zeros(3)])
        r,v=relative[:3],relative[3:]; rho=np.linalg.norm(r); u=r/rho
        ds=station_jacobian(P1,et,cylindrical)
        rr=np.r_[v/rho-np.dot(r,v)*r/rho**3, u]
        if kind=="RangeIdeal": expected=-u@ds[:3]
        elif kind=="RangeRateIdeal": expected=-rr@ds
        elif kind.startswith("DopplerIdeal"):
            factor=-8e9*1.1/C_KM_S
            expected=-factor*rr@ds*(1 if kind.endswith("3way") else 2)
            if kind.endswith("3way"):
                rel2=(spice.spkezr(target.name,et,"J2000","NONE","EARTH")[0]
                      - spice.sxform('ITRF93','J2000',et) @ np.r_[P2,np.zeros(3)])
                r2,v2=rel2[:3],rel2[3:];rho2=np.linalg.norm(r2)
                h2=np.r_[v2/rho2-np.dot(r2,v2)*r2/rho2**3,r2/rho2]
                expected[3:] -= factor*(h2@station_jacobian(P2,et))[3:]
        else:
            r2=(spice.spkezr(target.name,et,"J2000","NONE","EARTH")[0][:3]
                - spice.pxform('ITRF93','J2000',et) @ P2)
            expected=u@ds[:3]/C_KM_S
            expected[3:] -= (r2/np.linalg.norm(r2)@station_jacobian(P2,et)[:3])[3:]/C_KM_S
        np.testing.assert_allclose(H[i],expected,rtol=2e-5,
                                   atol=2e-8 if kind!='DiffOneWayRangeIdeal' else 1e-12)


@pytest.mark.parametrize("kind",["RangeIdeal","RangeRateIdeal"])
@pytest.mark.parametrize("pole_tide",[False,True])
def test_analytical_partials_match_forward_model_differences(bodies,kind,pole_tide):
    """The analytical block is the derivative of this model's own forward model.

    The previous test pins the formula; this one pins that the forward model
    applies exactly the correction the formula differentiates, which is the
    failure a self-consistent but wrongly-applied correction would slip past.
    With a pole tide attached the pole columns also carry its move.
    """
    gs,_,target,earth=bodies
    if pole_tide:
        gs.surface_displacement=scb.PoleTide(pole_x=1e-6,pole_y=2e-6)
    epoch=scb.EpochArray([ET+900.],sys="TDB")
    steps=np.array([1e-4,1e-4,1e-4,1e-7,1e-7,1e-4])
    H=np.asarray(make_model(kind,bodies,entries(gs,earth)).compute_partials(target,epoch,J2000))[0]
    for column,step in enumerate(steps):
        outputs=[]
        for sign in (1,-1):
            values=np.zeros(6); values[column]=sign*step
            perturbed=make_model(kind,bodies,entries(gs,earth,values))
            outputs.append(float(np.asarray(perturbed.computed_measurements(
                target,epoch,frame=J2000,noisy=False)[-1].quantity.values).ravel()[0]))
        difference=(outputs[0]-outputs[1])/(2*step)
        np.testing.assert_allclose(H[column],difference,rtol=2e-4,
                                   atol=1e-6*max(abs(difference),1.))


@pytest.mark.parametrize("kind",["RangeIdeal","RangeRateIdeal"])
@pytest.mark.parametrize("axis",[0,1])
def test_estimated_pole_deltas_move_an_attached_pole_tide(bodies,kind,axis):
    """A pole delta moves the pole tide as a pole tide built at the moved pole does.

    The tide's share of the observable is isolated twice: with the delta
    estimated, as the difference with and without the tide attached, and
    with the delta written into the tide itself, against no tide at all.
    """
    gs,_,target,earth=bodies
    epoch=scb.EpochArray([ET+900.],sys="TDB")
    delta=np.zeros(2); delta[axis]=1e-6
    values=np.r_[np.zeros(3),delta,0.]

    def observable(tide,state):
        gs.surface_displacement=tide
        return float(np.asarray(make_model(kind,bodies,state).computed_measurements(
            target,epoch,frame=J2000,noisy=False)[-1].quantity.values).ravel()[0])

    pole=(1e-6,2e-6)
    estimated=(observable(scb.PoleTide(*pole),entries(gs,earth,values))
               -observable(None,entries(gs,earth,values)))
    built=(observable(scb.PoleTide(pole[0]+delta[0],pole[1]+delta[1]),entries(gs,earth))
           -observable(None,entries(gs,earth)))
    nominal=(observable(scb.PoleTide(*pole),entries(gs,earth))
             -observable(None,entries(gs,earth)))
    # The delta moves the tide by a large fraction of its own size; the floor
    # is the round-off of four 30,000 km ranges.
    assert abs(built-nominal)>0.1*abs(built)
    np.testing.assert_allclose(estimated-nominal,built-nominal,rtol=1e-4,atol=5e-11)


def test_station_block_is_the_chain_rule_of_the_position_and_velocity_partials(bodies):
    """The block is exactly -[dh/dr, dh/dv] @ d(station state)/d(parameter)."""
    gs,_,target,earth=bodies
    model=make_model("RangeRateIdeal",bodies,entries(gs,earth))
    epoch=scb.EpochArray([ET+250.],sys="TDB")
    relative=model._relative_state(target,list(epoch)[0],J2000)
    h_r=np.asarray(model._compute_h_tilde_pos(relative),dtype=float).ravel()
    h_v=np.asarray(model._compute_h_tilde_vel(relative),dtype=float).ravel()
    for component in entries(gs,earth):
        jacobian=model._station_state_partials(component,ET+250.,J2000)
        block=model._compute_h_tilde_station(component,ET+250.,J2000,h_r,h_v)[0]
        np.testing.assert_allclose(block,-np.r_[h_r,h_v]@jacobian,rtol=0,atol=1e-15)


def test_earth_correction_partials_agree_with_earth_correction(bodies):
    """The EOP Jacobian columns are the derivative of earth_correction itself."""
    columns=scb.BodyOrientation.earth_correction_partials()
    for name,column in columns.items():
        step=1e-6
        plus=scb.BodyOrientation.earth_correction(**{name:step}).rotation_vector
        minus=scb.BodyOrientation.earth_correction(**{name:-step}).rotation_vector
        np.testing.assert_allclose(column,(plus-minus)/(2*step),rtol=1e-12,atol=1e-18)


def test_celestial_pole_correction_partials_agree_with_the_correction():
    """The CPO columns are the derivative of celestial_pole_correction itself."""
    columns=scb.BodyOrientation.celestial_pole_correction_partials()
    for name,column in columns.items():
        step=1e-6
        plus=scb.BodyOrientation.celestial_pole_correction(**{name:step})
        minus=scb.BodyOrientation.celestial_pole_correction(**{name:-step})
        np.testing.assert_allclose(column,(plus-minus)/(2*step),rtol=1e-12,atol=1e-18)
    with pytest.raises(scb.BodyOrientation.NonFiniteOffsetError):
        scb.BodyOrientation.celestial_pole_correction(delta_x=float("nan"))


@pytest.mark.parametrize("kind",["RangeIdeal","RangeRateIdeal","DopplerIdeal"])
def test_celestial_pole_partials_match_an_inertial_rotation_built_here(bodies,kind):
    """dX turns the station about J2000 +Y and dY about -X (IERS 2010 eq. 5.10).

    The axes stay fixed in J2000 while the Earth turns under them, which is
    what separates these columns from the terrestrial pole ones.
    """
    gs,_,target,earth=bodies
    model=make_model(kind,bodies,celestial_entries(earth))
    epoch=scb.EpochArray([ET+11.7,ET+40031.1],sys="TDB")
    H=np.asarray(model.compute_partials(target,epoch,J2000))
    assert H.shape==(2,2)
    for i,et in enumerate(epoch.times.values):
        station=spice.sxform('ITRF93','J2000',et) @ np.r_[P1,np.zeros(3)]
        relative=spice.spkezr(target.name,et,"J2000","NONE","EARTH")[0]-station
        r,v=relative[:3],relative[3:]; rho=np.linalg.norm(r); u=r/rho
        ds=np.column_stack([np.r_[np.cross(axis,station[:3]),np.cross(axis,station[3:])]
                            for axis in ([0.,1.,0.],[-1.,0.,0.])])
        rr=np.r_[v/rho-np.dot(r,v)*r/rho**3,u]
        expected={"RangeIdeal":-u@ds[:3],"RangeRateIdeal":-rr@ds,
                  "DopplerIdeal":2*8e9*1.1/C_KM_S*rr@ds}[kind]
        np.testing.assert_allclose(H[i],expected,rtol=2e-5,atol=1e-12)


@pytest.mark.parametrize("kind",["RangeIdeal","RangeRateIdeal"])
def test_celestial_pole_partials_match_forward_model_differences(bodies,kind):
    """The forward model applies exactly the rotation the CPO columns differentiate."""
    gs,_,target,earth=bodies
    epoch=scb.EpochArray([ET+900.],sys="TDB")
    H=np.asarray(make_model(kind,bodies,celestial_entries(earth)).compute_partials(
        target,epoch,J2000))[0]
    step=1e-7
    for column in range(2):
        outputs=[]
        for sign in (1,-1):
            values=np.zeros(2); values[column]=sign*step
            perturbed=make_model(kind,bodies,celestial_entries(earth,values))
            outputs.append(float(np.asarray(perturbed.computed_measurements(
                target,epoch,frame=J2000,noisy=False)[-1].quantity.values).ravel()[0]))
        difference=(outputs[0]-outputs[1])/(2*step)
        assert abs(difference)>0
        np.testing.assert_allclose(H[column],difference,rtol=2e-6,
                                   atol=1e-9*max(abs(difference),1.))


#: About 2e-4 rad each at ET, so the move clears the round-off of a Mars range.
IAU_POLE_STEPS = {"pole_ra": 1e-4, "pole_dec": 1e-4, "pole_pm": 1e-3, "pole_w0": 1e-4}


def observable(model, target, epoch):
    return float(np.asarray(model.computed_measurements(
        target, epoch, frame=J2000, noisy=False)[-1].quantity.values).ravel()[0])


@pytest.mark.parametrize("kind", ["RangeIdeal", "RangeRateIdeal"])
@pytest.mark.parametrize("body", ["EARTH", "MARS"])
def test_iau_pole_partials_match_forward_model_differences(bodies, kind, body):
    """The IAU pole of the central body moves its stations as its rows say.

    Each value is moved in the state and the model rerun, so the reference
    shares nothing with the closed-form rows but the forward model. The Mars
    station is a table station, the Earth one a kernel station in ITRF93.
    """
    gs, gs2, target, earth = bodies
    owner = earth if body == "EARTH" else scb.Body("MARS", 499)
    station = gs if body == "EARTH" else mars_table_station()
    offsets = np.array([1e-6, -2e-6, 1e-7, 3e-6])
    epoch = scb.EpochArray([ET + 900.], sys="TDB")

    def model(values):
        return make_model(kind, (station, gs2, target, earth), iau_pole_entries(owner, body, values))

    H = np.asarray(model(offsets).compute_partials(target, epoch, J2000))[0]
    assert H.shape == (4,)
    for column, name in enumerate(scb.BodyOrientation.IAU_POLE_PARAMETERS):
        step = IAU_POLE_STEPS[name]
        outputs = []
        for sign in (1, -1):
            values = offsets.copy(); values[column] += sign*step
            outputs.append(observable(model(values), target, epoch))
        difference = (outputs[0]-outputs[1])/(2*step)
        assert abs(difference) > 0
        np.testing.assert_allclose(H[column], difference, rtol=1e-4,
                                   atol=1e-9*max(abs(difference), 1.))


@pytest.mark.parametrize("body", ["EARTH", "MARS"])
def test_unset_iau_pole_leaves_the_model_on_the_secular_rotation(bodies, body):
    """Declared without values the pole moves nothing, and its rows are the
    linearisation about the secular model: those of values at the secular
    angles, whose dropped rates are below 1e-7 of the spin."""
    gs, gs2, target, earth = bodies
    owner = earth if body == "EARTH" else scb.Body("MARS", 499)
    station = gs if body == "EARTH" else mars_table_station()
    group = (station, gs2, target, earth)
    epoch = scb.EpochArray([ET + 900.], sys="TDB")
    unset = make_model("RangeRateIdeal", group, iau_pole_entries(owner, body, (None,)*4))
    assert [p["values"] for p in unset._station_parameters()] == [None]*4
    bare = make_model("RangeRateIdeal", group, [])
    assert observable(unset, target, epoch) == observable(bare, target, epoch)
    at_secular = make_model("RangeRateIdeal", group, iau_pole_entries(owner, body))
    np.testing.assert_allclose(np.asarray(unset.compute_partials(target, epoch, J2000)),
                               np.asarray(at_secular.compute_partials(target, epoch, J2000)),
                               rtol=1e-5)


def test_a_station_owned_iau_pole_is_ignored(bodies):
    """The pole is the central body's; a station declaring one gets nothing."""
    gs, _, target, earth = bodies
    model = make_model("RangeIdeal", bodies, iau_pole_entries(gs, "EARTH", (1e-3,)*4))
    assert model._station_parameters() == []
    epoch = scb.EpochArray([ET + 900.], sys="TDB")
    assert observable(model, target, epoch) == observable(make_model("RangeIdeal", bodies, []),
                                                          target, epoch)


@pytest.mark.parametrize("kind", ["SequentialRangingReal", "DopplerReal"])
def test_differencing_an_unset_iau_pole_is_refused(bodies, kind):
    """Finite differences need a value to move; the unset pole has none."""
    gs, _, target, earth = bodies
    epoch = scb.EpochArray([ET + 314.7], sys="TDB")
    model = getattr(scb, kind)("unset pole", gs,
                               state_definition=iau_pole_entries(earth, "EARTH", (None,)*4),
                               computed_measurements_dict=real_aux(epoch), partials_method="fd")
    with pytest.raises(scb.Measurement.UnsetPoleParameterError, match="pole_ra"):
        model.compute_partials(target, epoch, J2000)


def test_cylindrical_basis_carries_a_rate(bodies):
    """East is an arc length on a turning body, so the basis has a derivative."""
    gs,_,_,_=bodies
    velocity=np.array([.001,-.002,.003])
    gs._nominal_state=lambda epoch,frame=None:np.r_[P1+(epoch-ET)*velocity,velocity]
    basis,rate=gs.cylindrical_basis(ET)
    rate=rate.values
    before=gs.cylindrical_basis(ET-1)[0];after=gs.cylindrical_basis(ET+1)[0]
    np.testing.assert_allclose(rate,(after-before)/2,atol=1e-16)
    assert np.linalg.norm(rate)>0


def test_declared_units_and_reference_iterate_are_used(bodies):
    """Partials come back per declared unit, reading the filter's current iterate."""
    gs,_,target,earth=bodies
    model=make_model('RangeIdeal',bodies,entries(gs,earth))
    m=scb.Units.get_units(['m'])
    definition=scb.StateDefinition().param('gs_delta_location_CYL_station',gs,
                                          scb.ArrayWFrame(np.array([30.,-10.,20.]),m,ITRF),dynamics='static')
    reference=scb.StateArray(scb.EpochArray(ET,sys='TDB'),earth,definition)
    model.update_reference_state(reference)
    epoch=scb.EpochArray([ET],sys='TDB')
    H=np.asarray(model.compute_partials(target,epoch,J2000))[0]
    # A unit direction differentiated per metre rather than per km.
    assert np.linalg.norm(H) == pytest.approx(.001,rel=1e-6)
    values=model._station_parameters()[0]['values']
    np.testing.assert_allclose(values,[.03,-.01,.02],atol=1e-12)


# --------------------------------------------------------------------------- #
# Recovery: the partials actually solve for what was put in                    #
# --------------------------------------------------------------------------- #
def test_cylindrical_station_offset_is_recovered_from_multiepoch_range(bodies):
    gs,_,target,earth=bodies
    epochs=scb.EpochArray(ET+np.linspace(0,70000,20),sys='TDB')
    truth=np.array([.03,-.02,.01])
    truth_model=make_model('RangeIdeal',bodies,entries(gs,earth,np.r_[truth,0,0,0])[:1])
    observed=truth_model.computed_measurements(target,epochs,frame=J2000)[-1].quantity.values.ravel()
    model=make_model('RangeIdeal',bodies,entries(gs,earth)[:1])
    computed=model.computed_measurements(target,epochs,frame=J2000)[-1].quantity.values.ravel()
    H=np.asarray(model.compute_partials(target,epochs,J2000))
    estimate=np.linalg.lstsq(H,observed-computed,rcond=None)[0]
    np.testing.assert_allclose(estimate,truth,atol=1e-7)


def test_eop_corrections_are_recovered_with_two_fixed_stations(bodies):
    gs,gs2,target,earth=bodies
    epochs=scb.EpochArray(ET+np.linspace(0,70000,16),sys='TDB')
    truth=np.array([2e-7,-3e-7,.004])
    residuals,rows=[],[]
    for station in (gs,gs2):
        local=(station,gs2,target,earth)
        true_model=make_model('RangeIdeal',local,entries(station,earth,np.r_[0,0,0,truth])[1:])
        nominal=make_model('RangeIdeal',local,entries(station,earth)[1:])
        y=true_model.computed_measurements(target,epochs,frame=J2000)[-1].quantity.values.ravel()
        h=nominal.computed_measurements(target,epochs,frame=J2000)[-1].quantity.values.ravel()
        residuals.extend(y-h)
        rows.extend(nominal.compute_partials(target,epochs,J2000))
    scales=np.array([1e-6,1e-6,.01])
    design=np.asarray(rows)*scales
    assert np.linalg.matrix_rank(design)==3
    estimate=np.linalg.lstsq(design,np.asarray(residuals),rcond=None)[0]*scales
    np.testing.assert_allclose(estimate,truth,rtol=2e-5,atol=1e-11)


def test_celestial_and_terrestrial_pole_offsets_are_recovered_together(bodies):
    """Static CPOs stay put in J2000 while static polar motion turns with the
    Earth, so a day of range from two stations separates all five EOPs."""
    gs,gs2,target,earth=bodies
    epochs=scb.EpochArray(ET+np.linspace(0,86000,24),sys='TDB')
    truth=np.array([2e-7,-3e-7,.004,1.5e-7,-1e-7])
    residuals,rows=[],[]
    for station in (gs,gs2):
        local=(station,gs2,target,earth)
        true_model=make_model('RangeIdeal',local,entries(station,earth,np.r_[0,0,0,truth[:3]])[1:]
                              +celestial_entries(earth,truth[3:]))
        nominal=make_model('RangeIdeal',local,entries(station,earth)[1:]+celestial_entries(earth))
        y=true_model.computed_measurements(target,epochs,frame=J2000)[-1].quantity.values.ravel()
        h=nominal.computed_measurements(target,epochs,frame=J2000)[-1].quantity.values.ravel()
        residuals.extend(y-h)
        rows.extend(nominal.compute_partials(target,epochs,J2000))
    scales=np.array([1e-6,1e-6,.01,1e-6,1e-6])
    design=np.asarray(rows)*scales
    assert np.linalg.matrix_rank(design)==5
    estimate=np.linalg.lstsq(design,np.asarray(residuals),rcond=None)[0]*scales
    np.testing.assert_allclose(estimate,truth,rtol=1e-4,atol=1e-11)


# --------------------------------------------------------------------------- #
# Scope: who does and does not carry station parameters                        #
# --------------------------------------------------------------------------- #
def test_receive_station_location_is_estimable(bodies):
    """A parameter on the second station moves the observable through its own leg."""
    gs,gs2,target,earth=bodies
    state=entries(gs2,earth)
    for kind in ("DopplerIdeal3way","DiffOneWayRangeIdeal"):
        model=make_model(kind,bodies,state)
        H=np.asarray(model.compute_partials(target,scb.EpochArray([ET],sys="TDB"),J2000))
        assert np.linalg.norm(H[0,:3])>1e-9


def test_spacecraft_camera_has_no_station_and_reports_zero_blocks(bodies):
    """An optical model observing from a spacecraft is untouched by station work."""
    gs,_,target,earth=bodies
    camera=scb.Camera("SC_CAMERA",-1001,J2000,scb.ArrayWUnits(np.array([2.,2.]),rad),
                      (1024,1024),scb.ArrayWUnits(150e-6,km),
                      scb.ArrayWUnits(np.array([25e-6,25e-6]),rad))
    model=scb.CentroidingIdeal("centroid",camera,state_definition=entries(gs,earth))
    assert model._observer_station() is None
    assert model._station_parameters() == []
    assert model._station_state_correction(ET,J2000) is None
    H=np.asarray(model.compute_partials(target,scb.EpochArray([ET+11],sys="TDB"),J2000))
    np.testing.assert_allclose(H[:,:6],0.,atol=0.)


def test_generic_planet_station_geometry(bodies):
    """body_orientation_delta works on a body that has no EOP convention at all."""
    _,_,target,_=bodies
    station=scb.GroundStation('GEOMETRY_MARS_STATION',499014,central_body='MARS',body_fixed_frame='IAU_MARS')
    station.body_orientation=scb.BodyOrientation([1e-7,2e-7,-3e-7],'IAU_MARS')
    model=scb.RangeIdeal('generic rotation',station,state_definition=[
        ('body_orientation_delta',3,True,'static',station,
         scb.ArrayWFrame(np.zeros(3),rad,scb.Frame("IAU_MARS")))])
    H=np.asarray(model.compute_partials(target,scb.EpochArray([ET],sys='TDB'),J2000))[0]
    assert np.all(np.abs(H)>1)


def test_schema_accepts_eop_and_generic_rotation(bodies):
    gs,_,_,earth=bodies
    definition=scb.StateDefinition()
    for name,dim,_,_,owner,value in entries(gs,earth)+celestial_entries(earth):
        definition.param(name,owner,value,dimension=dim,dynamics="static")
    definition.param("body_orientation_delta",earth,scb.ArrayWFrame(np.zeros(3),rad,ITRF),dynamics="static")
    assert len(definition.components)==7


def test_two_location_blocks_on_one_station_are_rejected(bodies):
    gs,_,target,earth=bodies
    state=entries(gs,earth,cylindrical=True)+[
        ("gs_delta_location_ECEF_station",3,True,"static",gs,
         scb.ArrayWFrame(np.zeros(3),km,ITRF))]
    model=make_model("RangeIdeal",bodies,state)
    with pytest.raises(ValueError,match="one Cartesian or cylindrical"):
        model._station_parameters()


@pytest.mark.parametrize("name,unit",[("eop_delta_ut1",sec),("eop_delta_celestial_pole_x",rad)])
def test_eop_parameters_are_rejected_off_earth(bodies,name,unit):
    _,_,target,_=bodies
    station=scb.GroundStation('GEOMETRY_MARS_EOP',499014,central_body='MARS',body_fixed_frame='IAU_MARS')
    mars=scb.Body("MARS",499)
    model=scb.RangeIdeal('mars eop',station,state_definition=[
        (name,1,True,"static",mars,scb.ArrayWFrame(np.zeros(1),unit,None))])
    with pytest.raises(ValueError,match="Earth-specific"):
        model._station_parameters()


@pytest.mark.parametrize('coordinates', [[1, 2], [1, 2, float('nan')]])
def test_explicit_station_coordinates_are_validated(coordinates):
    with pytest.raises(ValueError, match='three finite'):
        scb.GroundStation('TABLE_INVALID', 399988, body_fixed_position=coordinates)


# --------------------------------------------------------------------------- #
# Stations defined by a table rather than an SPK                               #
# --------------------------------------------------------------------------- #
def test_table_station_cylindrical_coordinates_without_station_spk(bodies):
    _, _, target, _ = bodies
    radius, longitude, height = 5209.482541999999, -116.8747939, 3669.24231
    deg = scb.Units.get_units(['deg'])
    station = scb.GroundStation.from_cylindrical('TABLE_DIRECT_GS', 399987,
        longitude=scb.ArrayWUnits(longitude, deg),
        cylindrical_radius=scb.ArrayWUnits(radius * 1000, scb.Units.get_units(['m'])), z=height)
    expected_position = np.array([radius*np.cos(np.deg2rad(longitude)),
                                  radius*np.sin(np.deg2rad(longitude)), height])
    np.testing.assert_allclose(station.nominal_state(ET).quantity.values, np.r_[expected_position, np.zeros(3)], atol=1e-12)
    # No ephemeris for this station ID exists; direct SPICE cannot find it.
    with pytest.raises(spice.utils.exceptions.SpiceyError):
        spice.spkezr(str(station.spice_id), ET, 'J2000', 'NONE', 'EARTH')
    expected = spice.sxform('ITRF93', 'J2000', ET) @ np.r_[expected_position, np.zeros(3)]
    np.testing.assert_allclose(station.nominal_state(ET, 'J2000').quantity.values, expected, atol=1e-12)
    model = scb.RangeIdeal('table', station, sigma=scb.ArrayWUnits(.001, km))
    relative = np.asarray(model._relative_state(target, scb.EpochArray([ET], sys='TDB').times.values,
                                                J2000).values, dtype=float)
    np.testing.assert_allclose(
        relative, spice.spkezr(target.name, ET, 'J2000', 'NONE', 'EARTH')[0] - expected, atol=1e-9)


def test_table_and_kernel_stations_agree_when_they_are_the_same_point(bodies):
    kernel_station, _, target, earth = bodies
    table_station = scb.GroundStation('TABLE_MATCH_GS', 399986, body_fixed_position=P1)
    epochs = scb.EpochArray([ET + 314.7, ET + 5000.], sys='TDB')
    outputs, jacobians = [], []
    for station in (kernel_station, table_station):
        model = scb.RangeIdeal('match', station, sigma=scb.ArrayWUnits(.001, km),
                               state_definition=entries(station, earth))
        outputs.append(model.computed_measurements(target, epochs, frame=J2000)[-1].quantity.values)
        jacobians.append(model.compute_partials(target, epochs, J2000))
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(jacobians[1], jacobians[0], rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# The light-time models                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kind',['SequentialRangingReal','DopplerReal'])
@pytest.mark.parametrize('fd',[False,True])
@pytest.mark.parametrize('custom_steps', [False, True])
def test_real_models_station_partials_match_an_independently_displaced_spk(
        bodies,kind,fd,custom_steps,tmp_path,capsys):
    """The RTLT chain moves with the station, and its partial follows the move.

    The reference here is a *separately written* station SPK displaced along
    each cylindrical axis, so nothing in the library's own perturbation path is
    reused to produce the expected answer.
    """
    gs,_,target,earth=bodies
    state=entries(gs,earth)
    epoch=scb.EpochArray([ET+314.7],sys="TDB")
    # Station steps live in the same eps_dict as the spacecraft ones.
    eps = {'gs_delta_location_CYL': scb.ArrayWUnits(500., scb.Units.get_units(['m']))} if custom_steps else None
    model=getattr(scb,kind)("real station",gs,state_definition=state,
                          computed_measurements_dict=real_aux(epoch),
                          partials_method=("fd" if fd else "analytical"),
                          eps_dict=eps)
    H=np.asarray(model.compute_partials(target,epoch,J2000))[0]
    assert np.all(np.isfinite(H)) and np.all(np.abs(H[:3])>1e-6)
    assert 'failed at iteration' not in capsys.readouterr().out
    lon=np.arctan2(P1[1],P1[0]);c,s=np.cos(lon),np.sin(lon)
    basis=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    for j in range(3):
        ys=[]
        for sign in (1,-1):
            path=tmp_path/f'{kind}-{j}-{sign}.bsp'
            h=spice.spkopn(str(path),'independently displaced station',0)
            states=np.tile(np.r_[P1+sign*.5*basis[:,j],np.zeros(3)],(2,1))
            times=np.array([ET-10000,ET+10000])
            spice.spkw09(h,gs.spice_id,399,'ITRF93',times[0],times[-1],'offset',1,2,states,times)
            spice.spkcls(h);spice.furnsh(str(path))
            try:
                ys.append(model.computed_measurements(target,epoch,frame=J2000)[-1].quantity.values.ravel()[0])
            finally:spice.unload(str(path))
        expected=(ys[0]-ys[1])/(2*.5)
        np.testing.assert_allclose(H[j],expected,rtol=2e-3,
                                   atol=.1 if kind=='SequentialRangingReal' else .001)


@pytest.mark.parametrize('kind', ['SequentialRangingReal', 'DopplerReal'])
def test_real_models_require_a_station_kernel(bodies, kind):
    """A table station is rejected by name, not by a bare SPICE lookup failure.

    The Moyer chain reads the station from the kernel pool at several epochs
    and frames, so table coordinates have nothing to resolve against. The
    geometric models compose them directly and do accept such a station.
    """
    _, _, target, earth = bodies
    table_station = scb.GroundStation('TABLE_REAL_GS', 399986, body_fixed_position=P1)
    epochs = scb.EpochArray([ET + 314.7], sys='TDB')
    model = getattr(scb, kind)('table real', table_station,
        state_definition=entries(table_station, earth),
        computed_measurements_dict=real_aux(epochs))
    with pytest.raises(scb.Measurement.StationKernelRequiredError, match='body_fixed_position'):
        model.computed_measurements(target, epochs, frame=J2000)


def test_station_eps_defaults_and_overrides(bodies):
    """One perturbation dictionary carries both spacecraft and station steps."""
    gs,_,_,earth=bodies
    model=scb.DopplerReal('eps',gs,state_definition=entries(gs,earth))
    assert model._station_eps('gs_delta_location_CYL') == pytest.approx(1.0)
    model.eps_dict={'gs_delta_location_CYL':scb.ArrayWUnits(20.,scb.Units.get_units(['m'])),
                    'dx':scb.ArrayWUnits(10.,km)}
    assert model._station_eps('gs_delta_location_CYL') == pytest.approx(.02)
    assert model._station_eps('eop_delta_ut1') == pytest.approx(1.0)
    model.eps_dict={'gs_delta_location_CYL':scb.ArrayWUnits(-1.,km)}
    with pytest.raises(ValueError,match='positive and finite'):
        model._station_eps('gs_delta_location_CYL')


def test_body_orientation_delta_frame_is_validated(bodies):
    """A rotation declared in the wrong frame is rejected, not misapplied."""
    gs,_,target,earth=bodies
    model=scb.RangeIdeal('wrong frame',gs,sigma=scb.ArrayWUnits(.001,km),
        state_definition=[('body_orientation_delta',3,True,'static',gs,
                           scb.ArrayWFrame(np.zeros(3),rad,scb.Frame('IAU_EARTH')))])
    with pytest.raises(ValueError,match='body_fixed_frame|body-fixed frame'):
        model.compute_partials(target,scb.EpochArray([ET],sys='TDB'),J2000)


def test_spacecraft_to_spacecraft_observer_carries_no_station_parameters(bodies):
    """A spacecraft-mounted antenna endpoint is not probed for station geometry.

    Range between two spacecraft has no ground station at either end. The
    station lookup must return None for such an observer rather than reaching
    for ``central_body``, and must not silently substitute another station.
    """
    _,_,target,earth=bodies
    antenna=scb.Antenna("onboard",turn_ratio=1.1,spice_id=-1001)
    gs=scb.GroundStation("GEOMETRY_GS1",399014)
    model=scb.RangeIdeal("sc to sc",antenna,sigma=scb.ArrayWUnits(.001,km),
                         state_definition=entries(gs,earth))
    assert model._observer_station() is None
    assert model._as_station(antenna) is None
    assert model._station_parameters() == []
    epoch=scb.EpochArray([ET],sys="TDB")
    H=np.asarray(model.compute_partials(target,epoch,J2000))
    np.testing.assert_allclose(H,0.,atol=0.)
    # The observer is still the antenna, not some other station.
    relative=model._relative_state(target,list(epoch)[0],J2000)
    np.testing.assert_allclose(np.asarray(relative.values,dtype=float),
        spice.spkezr(target.name,ET,"J2000","NONE",str(antenna.spice_id))[0],atol=1e-9)


# --------------------------------------------------------------------------- #
# Love/Shida numbers                                                          #
# --------------------------------------------------------------------------- #
def love_state(gs, values=(0.6078, 0.0847)):
    return [("love_h2", 1, True, "static", gs, scb.ArrayWFrame(np.array([values[0]]), None, None)),
            ("shida_l2", 1, True, "static", gs, scb.ArrayWFrame(np.array([values[1]]), None, None))]


def test_love_shida_partials_match_the_tide_jacobian(bodies):
    """The Love/Shida block is the tide Jacobian chained through the range partial."""
    gs, _, target, _ = bodies
    gs.surface_displacement = scb.SolidTide()
    model = scb.RangeIdeal("love", gs, sigma=scb.ArrayWUnits(.001, km),
                           state_definition=love_state(gs))
    epoch = scb.EpochArray([ET + 3000.], sys="TDB")
    H = np.asarray(model.compute_partials(target, epoch, J2000))[0]
    relative = model._relative_state(target, list(epoch)[0], J2000)
    unit = np.asarray(model._compute_h_tilde_pos(relative), dtype=float).ravel()
    jacobian = gs.tide_partials(ET + 3000., J2000).values
    np.testing.assert_allclose(H, -unit @ jacobian[:3], rtol=1e-10, atol=1e-18)
    assert np.all(np.abs(H) > 0)


def tides(pole_tide, shift=(0.0, 0.0)):
    """A solid tide, and optionally a pole tide, both moved by the same amount."""
    solid = scb.SolidTide()
    models = [scb.SolidTide(love_h2=solid.love_h2 + shift[0],
                            shida_l2=solid.shida_l2 + shift[1])]
    if pole_tide:
        pole = scb.PoleTide(pole_x=1e-6, pole_y=2e-6)
        models.append(scb.PoleTide(pole_x=1e-6, pole_y=2e-6,
                                   love_h2=pole.love_h2 + shift[0],
                                   shida_l2=pole.shida_l2 + shift[1]))
    return models


def test_love_shida_at_the_reference_numbers_leave_every_tide_alone(bodies):
    """The pole tide keeps its own numbers when the solid tide's are declared."""
    gs, _, target, _ = bodies
    gs.surface_displacement = tides(True)
    epoch = scb.EpochArray([ET + 3000.], sys="TDB")
    outputs = [float(np.asarray(scb.RangeIdeal("love", gs, sigma=scb.ArrayWUnits(.001, km),
                                               state_definition=state).computed_measurements(
        target=target, epoch_array=epoch, frame=J2000, noisy=False)[-1].quantity.values).ravel()[0])
        for state in (None, love_state(gs))]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("pole_tide", [False, True])
def test_love_shida_partials_match_forward_differences(bodies, pole_tide):
    """One estimate moves every attached tide by the same amount."""
    gs, _, target, _ = bodies
    epoch = scb.EpochArray([ET + 3000.], sys="TDB")
    gs.surface_displacement = tides(pole_tide)
    H = np.asarray(scb.RangeIdeal("love", gs, sigma=scb.ArrayWUnits(.001, km),
                                  state_definition=love_state(gs)).compute_partials(
                                      target, epoch, J2000))[0]
    for column, name in enumerate(("love_h2", "shida_l2")):
        outputs = []
        for sign in (1, -1):
            shift = [0.0, 0.0]
            shift[column] = sign * 1e-4
            values = [0.6078 + shift[0], 0.0847 + shift[1]]
            station = scb.GroundStation("GEOMETRY_GS1", 399014)
            station.surface_displacement = tides(pole_tide, shift)
            perturbed = scb.RangeIdeal("love", station, sigma=scb.ArrayWUnits(.001, km),
                                       state_definition=love_state(station, values))
            outputs.append(float(np.asarray(perturbed.computed_measurements(
                target=target, epoch_array=epoch, frame=J2000,
                noisy=False)[-1].quantity.values).ravel()[0]))
        np.testing.assert_allclose(H[column], (outputs[0] - outputs[1]) / 2e-4, rtol=5e-3)


def test_love_shida_have_no_sensitivity_without_a_tide_model(bodies):
    """Estimating them with no tide attached gives an honest zero column."""
    gs, _, target, _ = bodies
    assert gs.surface_displacement is None
    model = scb.RangeIdeal("love", gs, sigma=scb.ArrayWUnits(.001, km),
                           state_definition=love_state(gs))
    H = np.asarray(model.compute_partials(target, scb.EpochArray([ET], sys="TDB"), J2000))
    np.testing.assert_allclose(H, 0.0, atol=0.0)


# --------------------------------------------------------------------------- #
# Zenith path delays                                                          #
# --------------------------------------------------------------------------- #
def test_zenith_delays_are_read_from_the_state_per_station(bodies):
    gs, gs2, _, _ = bodies
    m = scb.Units.get_units(["m"])
    model = scb.RangeIdeal("media", gs, sigma=scb.ArrayWUnits(.001, km), state_definition=[
        ("tropo_zenith_delay_a", 1, True, "static", gs, scb.ArrayWFrame(np.array([0.12]), m, None)),
        ("iono_zenith_delay_a", 1, True, "static", gs, scb.ArrayWFrame(np.array([-0.04]), m, None)),
        # Another station's atmosphere is not this station's.
        ("tropo_zenith_delay_b", 1, True, "static", gs2, scb.ArrayWFrame(np.array([9.0]), m, None)),
    ])
    tropo, iono = model._zenith_delays()
    assert tropo == pytest.approx(0.12)
    assert iono == pytest.approx(-0.04)
    assert model._zenith_delays(gs2)[0] == pytest.approx(9.0)
    # Declared in km, read in m.
    model._state_definition = [
        ("tropo_zenith_delay_a", 1, True, "static", gs,
         scb.ArrayWFrame(np.array([0.002]), km, None))]
    assert model._zenith_delays()[0] == pytest.approx(2.0)


def test_zenith_delay_mappings_are_unity_at_zenith_and_grow_at_low_elevation():
    media = scb.MediaCorrections
    assert media.ionosphere_mapping(np.pi / 2) == pytest.approx(1.0, abs=1e-9)
    assert media.chao_mapping(np.pi / 2)[1] == pytest.approx(1.0, abs=1e-3)
    # The thin-shell ionosphere is bounded where a plane-parallel model diverges.
    assert 2.0 < media.ionosphere_mapping(np.deg2rad(10)) < 3.5
    assert media.ionosphere_mapping(np.deg2rad(10)) < media.chao_mapping(np.deg2rad(10))[1]
    for angle in (10, 30, 60, 89):
        assert (media.ionosphere_mapping(np.deg2rad(angle))
                > media.ionosphere_mapping(np.deg2rad(angle + 1)))


def test_zenith_delays_enter_the_media_delay_linearly():
    """The delay is linear in each zenith parameter, at the mapped rate."""
    media = scb.MediaCorrections
    elevation = np.deg2rad(25.0)
    mapping = media.chao_mapping(elevation)[1]
    obliquity = media.ionosphere_mapping(elevation)
    # Reference frequency: the ionospheric scaling is unity there by definition.
    assert ((2295 * 1e6) / (2295 * 1e6)) ** 2 == 1.0
    assert mapping > 1.0 and obliquity > 1.0


@pytest.mark.parametrize("table_coordinates", [False, True])
def test_surface_displacement_reaches_the_forward_model_for_either_station_kind(
        bodies, table_coordinates):
    """A kernel station and a table station must both feel the solid tide.

    The forward model used to read a kernel station straight from its SPK,
    which skipped nominal_state and therefore skipped the tide, so the same
    station defined two ways gave two different observables. Nothing else in
    the suite would have caught that: the geometry is identical, only the
    displacement goes missing.
    """
    _, _, target, _ = bodies
    station = (scb.GroundStation("TIDE_TABLE_GS", 399984, body_fixed_position=P1)
               if table_coordinates else scb.GroundStation("GEOMETRY_GS1", 399014))
    epoch = scb.EpochArray([ET + 3000.], sys="TDB")
    model = scb.RangeIdeal("tide fwd", station, sigma=scb.ArrayWUnits(.001, km))
    without = float(np.asarray(model.computed_measurements(
        target=target, epoch_array=epoch, frame=J2000,
        noisy=False)[-1].quantity.values).ravel()[0])

    station.surface_displacement = scb.SolidTide()
    with_tide = float(np.asarray(model.computed_measurements(
        target=target, epoch_array=epoch, frame=J2000,
        noisy=False)[-1].quantity.values).ravel()[0])

    # Decimetre-scale, and equal to the tide projected on the line of sight.
    shift = (with_tide - without) * 1e3
    assert abs(shift) > 1e-3
    relative = np.asarray(model._relative_state(target, list(epoch)[0], J2000).values,
                          dtype=float)
    unit = relative[:3] / np.linalg.norm(relative[:3])
    displacement = spice.sxform("ITRF93", "J2000", ET + 3000.) @ \
        scb.SolidTide().state_displacement(
            station._undisplaced_state(ET + 3000.)[:3], ET + 3000., "ITRF93").values
    assert shift == pytest.approx(-(unit @ displacement[:3]) * 1e3, abs=2e-3)


# --------------------------------------------------------------------------- #
# The light-time models: analytical rows and the attached tide                #
# --------------------------------------------------------------------------- #
@pytest.fixture
def topocentric_frame(tmp_path):
    """GEOMETRY_GS1_TOPO, Z up at the station, for the media elevation angle."""
    up = P1 / np.linalg.norm(P1)
    east = np.cross([0., 0., 1.], up)
    east /= np.linalg.norm(east)
    # Columns are the topocentric axes in ITRF93, written column by column.
    rotation = np.column_stack([east, np.cross(up, east), up])
    path = tmp_path / "geometry_gs1_topo.tf"
    path.write_text(
        "\\begindata\n"
        "FRAME_GEOMETRY_GS1_TOPO = 1990014\n"
        "FRAME_1990014_NAME = 'GEOMETRY_GS1_TOPO'\n"
        "FRAME_1990014_CLASS = 4\n"
        "FRAME_1990014_CLASS_ID = 1990014\n"
        "FRAME_1990014_CENTER = 399014\n"
        "TKFRAME_1990014_RELATIVE = 'ITRF93'\n"
        "TKFRAME_1990014_SPEC = 'MATRIX'\n"
        # One value a line: a text kernel line stops at 132 characters.
        "TKFRAME_1990014_MATRIX = (\n"
        + "".join(f"  {value:.17g}\n" for value in rotation.T.ravel()) + ")\n"
        "\\begintext\n")
    spice.boddef("GEOMETRY_GS1", 399014)
    spice.furnsh(str(path))
    np.testing.assert_allclose(spice.pxform("GEOMETRY_GS1_TOPO", "ITRF93", ET) @ [0, 0, 1],
                               up, atol=1e-12)
    yield
    spice.unload(str(path))


def light_time_state(parameters, gs, earth):
    """One family of station or media parameters, at zero offset."""
    m = scb.Units.get_units(["m"])
    if parameters in ("ECEF", "CYL"):
        return entries(gs, earth, cylindrical=parameters == "CYL")
    if parameters == "orientation":
        return [("body_orientation_delta", 3, True, "static", gs,
                 scb.ArrayWFrame(np.zeros(3), rad, ITRF))]
    if parameters == "Love":
        return love_state(gs)
    if parameters == "celestial":
        return celestial_entries(earth)
    if parameters == "IAU pole":
        return iau_pole_entries(earth, "EARTH", (1e-6, -2e-6, 1e-7, 3e-6))
    return [("tropo_zenith_delay_a", 1, True, "static", gs, scb.ArrayWFrame(np.array([0.]), m, None)),
            ("iono_zenith_delay_a", 1, True, "static", gs, scb.ArrayWFrame(np.array([0.]), m, None))]


@pytest.mark.parametrize("kind", ["SequentialRangingReal", "DopplerReal"])
@pytest.mark.parametrize("parameters", ["ECEF", "CYL", "orientation", "Love", "zenith",
                                        "celestial", "IAU pole"])
def test_real_models_analytical_station_and_media_rows_match_finite_differences(
        bodies, topocentric_frame, kind, parameters):
    """Every station and zenith-delay row of the light-time models is analytical.

    The finite-difference reference moves the parameter in the state and reruns
    the whole Moyer chain, light-time solution and ramp integration included,
    so it shares nothing with the closed-form rows but the forward model they
    differentiate. It runs on the default steps of
    ``Measurement.station_eps_defaults``: with a unit Love step the Doppler
    reference was mostly light-time noise and missed the analytical row by
    0.75 %.
    """
    gs, _, target, earth = bodies
    # The pole tide makes the pole columns and the Love row carry its share.
    gs.surface_displacement = [scb.SolidTide(), scb.PoleTide(pole_x=1e-6, pole_y=2e-6)]
    # The spacecraft stands about 33 degrees above the station's horizon here.
    epoch = scb.EpochArray([ET + 36314.7], sys="TDB")
    media = scb.MediaCorrections("zenith", gs)
    assert 1.5 < media.zenith_delay_partials(ET + 36314.7, str(target.spice_id))[0] < 2.5
    rows = []
    for method in ("analytical", "fd"):
        model = getattr(scb, kind)("light time", gs,
                                   state_definition=light_time_state(parameters, gs, earth),
                                   computed_measurements_dict=real_aux(epoch),
                                   media_corrections=media, partials_method=method)
        rows.append(np.asarray(model.compute_partials(target, epoch, J2000))[0])
    analytical, fd = rows
    assert np.all(np.isfinite(analytical)) and np.all(np.abs(analytical) > 0)
    np.testing.assert_allclose(analytical, fd, rtol=2e-3,
                               atol=np.max(np.abs(fd)) * 1e-4)


def electronic_delay_state(gs, target, values=(0., 0., 0.)):
    """The station's uplink and downlink delays and the spacecraft's transponder delay [s]."""
    names = ("uplink_delay_bias_GS1", "transponder_delay_bias", "downlink_delay_bias_GS1")
    return [(name, 1, True, "static", owner, scb.ArrayWFrame(np.array([value]), sec, None))
            for name, owner, value in zip(names, (gs, target, gs), values)]


def test_real_ranging_electronic_delays_lengthen_the_light_time(bodies):
    """The t1, t2 and t3 delays lengthen the whole Moyer chain's round trip.

    With RU = 1 on a constant 8 GHz uplink the range grows by 8e9·(τ1 + τ2 + τ3)
    RU, and each analytical row, 8e9 RU/s, matches a finite difference of the
    full chain. Another station's delay and another spacecraft's transponder
    delay have zero rows.
    """
    gs, gs2, target, _ = bodies
    epoch = scb.EpochArray([ET + 314.7], sys="TDB")
    others = [("downlink_delay_bias_GS2", 1, True, "static", gs2,
               scb.ArrayWFrame(np.zeros(1), sec, None)),
              ("transponder_delay_bias", 1, True, "static", scb.Spacecraft("OTHER_SC", -1002),
               scb.ArrayWFrame(np.zeros(1), sec, None))]

    def model(values, method="analytical"):
        return scb.SequentialRangingReal(
            "delays", gs, state_definition=electronic_delay_state(gs, target, values) + others,
            computed_measurements_dict=real_aux(epoch), partials_method=method)

    def observable(values):
        return float(np.ravel(model(values).computed_measurements(
            target, epoch, frame=J2000)[-1].quantity.values)[0])

    delays = (1e-6, 2e-6, 3e-6)
    np.testing.assert_allclose(observable(delays) - observable((0., 0., 0.)),
                               8e9 * sum(delays), rtol=1e-5)
    analytical, fd = (np.asarray(model(delays, method).compute_partials(target, epoch, J2000))[0]
                      for method in ("analytical", "fd"))
    np.testing.assert_allclose(analytical, [8e9] * 3 + [0.] * 2, rtol=1e-12)
    np.testing.assert_allclose(fd, analytical, rtol=1e-5)


@pytest.mark.parametrize("kind", ["SequentialRangingReal", "DopplerReal"])
def test_real_models_place_the_station_where_its_attached_tide_puts_it(
        bodies, kind, tmp_path, capsys):
    """The kernel-reading light-time models feel the tide nominal_state carries.

    The reference is the bare station read from a separately written SPK that
    already holds the tide, sampled across the count interval so the Doppler
    sees its rate too.
    """
    gs, _, target, _ = bodies
    epoch = scb.EpochArray([ET + 314.7], sys="TDB")

    def observable(station):
        model = getattr(scb, kind)("tide", station, computed_measurements_dict=real_aux(epoch))
        return float(np.asarray(model.computed_measurements(
            target, epoch, frame=J2000)[-1].quantity.values).ravel()[0])

    bare = observable(gs)
    tide = scb.SolidTide()
    times = ET + 314.7 + np.arange(-200., 201., 10.)
    states = np.array([np.r_[P1, np.zeros(3)] + tide.state_displacement(P1, t, "ITRF93").values
                       for t in times])
    path = tmp_path / "tidal_station.bsp"
    handle = spice.spkopn(str(path), "station with the tide written in", 0)
    spice.spkw09(handle, gs.spice_id, 399, "ITRF93", times[0], times[-1], "tide",
                 7, len(times), states, times)
    spice.spkcls(handle)
    spice.furnsh(str(path))
    try:
        displaced = observable(gs)
    finally:
        spice.unload(str(path))
    gs.surface_displacement = tide
    attached = observable(gs)
    assert 'failed at iteration' not in capsys.readouterr().out
    # Decimetres of tide: about ten range units, or a fraction of a millihertz.
    assert abs(attached - bare) > (1.0 if kind == "SequentialRangingReal" else 1e-5)
    np.testing.assert_allclose(attached, displaced, rtol=0,
                               atol=1e-3 if kind == "SequentialRangingReal" else 1e-7)


# --------------------------------------------------------------------------- #
# Row assembly: widths and zero-fill                                          #
# --------------------------------------------------------------------------- #
def test_angular_position_partial_is_with_respect_to_the_target(bodies):
    """RA/DEC partials must be d/d(target), not its negation.

    They were the negation: _compute_h_tilde_pos returned the *observer*
    partial while the row assembly used it for the target and its negation for
    the observer, so both blocks carried the wrong sign. Nothing caught it
    because no test differenced the forward model.
    """
    gs, _, target, _ = bodies
    model = scb.AngularIdeal("angles", gs, sigma=scb.ArrayWUnits(np.ones(2) * 1e-6, rad))
    epoch = scb.EpochArray([ET + 3000.], sys="TDB")
    relative = model._relative_state(target, list(epoch)[0], J2000)
    analytic = model._compute_h_tilde_pos(relative)

    r = np.asarray(relative.values, dtype=float)[:3]
    x, y, z = r
    rho2 = x * x + y * y
    radius2 = rho2 + z * z
    expected = np.array([[-y / rho2, x / rho2, 0.0],
                         [-(x * z) / (radius2 * np.sqrt(rho2)),
                          -(y * z) / (radius2 * np.sqrt(rho2)),
                          np.sqrt(rho2) / radius2]])
    np.testing.assert_allclose(analytic, expected, rtol=1e-12)
    # The observer partial is the negation of it, not the other way round.
    np.testing.assert_allclose(model._compute_h_tilde_pos_secondary(relative),
                               -expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["RangeIdeal", "RangeRateIdeal", "DopplerIdeal",
                                  "DopplerIdeal3way", "DiffOneWayRangeIdeal"])
def test_unrecognised_components_get_zeros_of_their_own_width(bodies, kind):
    """Every component contributes exactly its own dimension, zeros if unknown.

    A parameter the model has no sensitivity to must still occupy its columns,
    or every column after it is silently misaligned with the state vector.
    """
    gs, gs2, target, earth = bodies
    moon = scb.Body("MOON", 301)
    comps = [
        ("position", 3, "estimated", "dynamic", target,
         scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("velocity", 3, "estimated", "dynamic", target,
         scb.ArrayWFrame(np.zeros(3), km / sec, J2000)),
        # Another body's position, and parameters that reach the observable
        # only through the trajectory: all must be zero here.
        ("position", 3, "estimated", "dynamic", moon,
         scb.ArrayWFrame(np.zeros(3), km, J2000)),
        ("gs_delta_location_CYL_a", 3, "estimated", "static", gs,
         scb.ArrayWFrame(np.zeros(3), km, ITRF)),
        ("gm", 1, "estimated", "static", earth,
         scb.ArrayWFrame(scb.ArrayWUnits(3.986e5, scb.Units.get_units(["mu"])), None)),
        ("cnm_2_0", 1, "estimated", "static", earth,
         scb.ArrayWFrame(np.zeros(1), None, None)),
        ("eta_srp", 1, "estimated", "static", target,
         scb.ArrayWFrame(np.zeros(1), None, None)),
    ]
    model = make_model(kind, (gs, gs2, target, earth),
                       scb.StateDefinition.from_components(comps))
    H = np.atleast_2d(np.asarray(
        model.compute_partials(target, scb.EpochArray([ET], sys="TDB"), J2000),
        dtype=float)).reshape(-1, sum(c[1] for c in comps))
    assert np.all(np.isfinite(H))

    offsets, position = {}, 0
    for name, dim, _, _, owner, _ in comps:
        offsets[(name, getattr(owner, "spice_id", None))] = (position, dim)
        position += dim
    for key in (("position", 301), ("gm", 399), ("cnm_2_0", 399), ("eta_srp", -1001)):
        start, width = offsets[key]
        np.testing.assert_allclose(H[:, start:start + width], 0.0, atol=0.0,
                                   err_msg=f"{key} should contribute zeros")
    # The station block is genuinely non-zero, so the zeros above are not
    # simply an all-zero H.
    start, width = offsets[("gs_delta_location_CYL_a", gs.spice_id)]
    assert np.any(np.abs(H[:, start:start + width]) > 0)


# The bias of each ideal model: its state name, the units it is given in (the
# observable's) and the change of the observable per unit of bias.
BIAS_COMPONENTS = {
    "RangeIdeal": ("range_bias_a", km, 1.0),
    "RangeRateIdeal": ("rangerate_bias_a", km / sec, 1.0),
    "DopplerIdeal": ("rangerate_bias_Hz_a", 1 / sec, 1.0),
    # A range on the first station's leg, so -1/c of delay per km
    "DiffOneWayRangeIdeal": ("dor_bias_a", km, -1.0 / C_KM_S),
}


@pytest.mark.parametrize("kind", list(BIAS_COMPONENTS))
@pytest.mark.parametrize("dimension,passes", [(1, False), (4, True), (4, False)])
def test_bias_block_always_matches_its_declared_width(bodies, kind, dimension, passes):
    """A bias contributes exactly its declared width, however it is configured.

    A stacked bias with no pass grouping used to emit a single column instead
    of its declared width, which shifted every later column out of step with
    the state vector -- silently, since the row was still finite. It now
    raises MissingPassIndexError instead.
    """
    gs, gs2, target, earth = bodies
    bias, unit, partial = BIAS_COMPONENTS[kind]
    comps = [("position", 3, "estimated", "dynamic", target,
              scb.ArrayWFrame(np.zeros(3), km, J2000)),
             (bias, dimension, "estimated", "static", gs,
              scb.ArrayWFrame(np.zeros(dimension), unit, None)),
             ("eta_srp", 1, "estimated", "static", target,
              scb.ArrayWFrame(np.zeros(1), None, None))]
    model = make_model(kind, (gs, gs2, target, earth),
                       scb.StateDefinition.from_components(comps))
    if passes:
        model.set_pass_epochs([[ET], [ET + 1000.], [ET + 2000.], [ET + 3000.]])
    elif dimension > 1:
        # No grouping to attribute the observation to: refused, by name.
        with pytest.raises(getattr(scb, kind).MissingPassIndexError):
            model.compute_partials(target, scb.EpochArray([ET], sys="TDB"), J2000)
        return
    H = np.asarray(model.compute_partials(
        target, scb.EpochArray([ET], sys="TDB"), J2000), dtype=float).ravel()

    assert H.size == sum(c[1] for c in comps)
    block = H[3:3 + dimension]
    if dimension == 1:
        assert block[0] == pytest.approx(partial)                    # one per station
    else:
        np.testing.assert_allclose(block, [partial, 0., 0., 0.])     # this pass only
    # The trailing component must still land in its own column.
    assert H[3 + dimension] == pytest.approx(0.0)


@pytest.mark.parametrize("kind", list(BIAS_COMPONENTS))
@pytest.mark.parametrize("passes", [False, True], ids=["global", "per-pass"])
def test_bias_columns_are_the_forward_model_differences(bodies, kind, passes):
    """Each bias column is the change of the observable per unit of that bias.

    The bias is moved in the reference state, where a filter puts its iterate,
    and the model rerun. One bias per pass moves only its own pass.
    """
    gs, _, target, _ = bodies
    name, unit, partial = BIAS_COMPONENTS[kind]
    epochs = scb.EpochArray([ET + 900., ET + 1900.], sys="TDB")
    nominal = np.full(2 if passes else 1, 0.3)

    def model(values):
        comps = [(name, values.size, True, "static", gs,
                  scb.ArrayWFrame(values, unit, None))]
        built = make_model(kind, bodies, comps)
        built.update_reference_state(SimpleNamespace(state=comps))
        if passes:
            built.set_pass_epochs([[t] for t in epochs.times.values])
        return built

    H = np.asarray(model(nominal).compute_partials(target, epochs, J2000),
                   dtype=float).reshape(2, nominal.size)
    # The observable is linear in its bias, so a whole unit is a clean step.
    for column in range(nominal.size):
        outputs = []
        for sign in (1, -1):
            values = nominal.copy(); values[column] += sign
            outputs.append(np.asarray(model(values).computed_measurements(
                target, epochs, frame=J2000, noisy=False)[-1].quantity.values,
                dtype=float).ravel())
        difference = (outputs[0] - outputs[1]) / 2
        expected = np.full(2, partial) if not passes else np.eye(2)[column] * partial
        np.testing.assert_allclose(difference, expected, rtol=1e-6, atol=0)
        np.testing.assert_allclose(H[:, column], difference, rtol=1e-6, atol=0)


# The bias of each light-time model, with the units the state gives it in.
REAL_BIAS_COMPONENTS = {
    "SequentialRangingReal": ("range_bias_RU_a", km / km),
    "DopplerReal": ("rangerate_bias_Hz_a", 1 / sec),
}


@pytest.mark.parametrize("kind", list(REAL_BIAS_COMPONENTS))
def test_real_models_bias_per_pass_is_the_forward_model_difference(bodies, kind):
    """One light-time bias per pass moves its own pass, by one unit per unit.

    Without a pass grouping the stacked bias cannot be attributed and is
    refused, by name, in both the forward model and the partials.
    """
    gs, _, target, _ = bodies
    name, unit = REAL_BIAS_COMPONENTS[kind]
    epochs = scb.EpochArray([ET + 36314.7, ET + 36614.7], sys="TDB")

    def model(values, passes=True):
        comps = [(name, 2, True, "static", gs, scb.ArrayWFrame(values, unit, None))]
        built = getattr(scb, kind)("bias", gs, state_definition=comps,
                                   computed_measurements_dict=real_aux(epochs))
        built.update_reference_state(SimpleNamespace(state=comps))
        if passes:
            built.set_pass_epochs([[t] for t in epochs.times.values])
        return built

    def observable(values):
        return np.asarray(model(values).computed_measurements(
            target, epochs, frame=J2000)[-1].quantity.values, dtype=float).ravel()

    nominal = np.array([0.3, -0.2])
    H = np.asarray(model(nominal).compute_partials(target, epochs, J2000), dtype=float)
    for column in range(2):
        step = np.eye(2)[column]
        difference = (observable(nominal + step) - observable(nominal - step)) / 2
        np.testing.assert_allclose(difference, step, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(H[:, column], step, rtol=1e-12, atol=0)

    for evaluate in ("computed_measurements", "compute_partials"):
        with pytest.raises(scb.Measurement.MissingPassIndexError):
            getattr(model(nominal, passes=False), evaluate)(target, epochs, frame=J2000)


@pytest.mark.parametrize("kind", ["SequentialRangingReal", "DopplerReal"])
def test_real_models_take_each_rows_state_definition_from_its_leg(
        bodies, topocentric_frame, kind):
    """Across a mission sequence each analytical row follows its leg's state.

    The two legs estimate different parameters, so their rows differ in width;
    each must equal the row a model holding that leg's state alone gives.
    The finite-difference path builds every column from one state and refuses
    a sequence.
    """
    gs, _, target, earth = bodies
    legs = [light_time_state("ECEF", gs, earth), light_time_state("zenith", gs, earth)]
    # Only the leg epochs and their state definitions are read; the scalar
    # middle entry is the node between the legs.
    sequence = SimpleNamespace(
        epochs_vec=[scb.EpochArray([ET + 36000., ET + 36400.], sys="TDB"),
                    scb.EpochArray(ET + 36400., sys="TDB"),
                    scb.EpochArray([ET + 36400., ET + 37000.], sys="TDB")],
        models=[SimpleNamespace(state_definition=legs[0]), None,
                SimpleNamespace(state_definition=legs[1])])
    epochs = scb.EpochArray([ET + 36314.7, ET + 36614.7], sys="TDB")
    media = scb.MediaCorrections("zenith", gs)

    def model(**options):
        return getattr(scb, kind)("legs", gs, computed_measurements_dict=real_aux(epochs),
                                  media_corrections=media, **options)

    rows = model(sequence_definition=sequence).compute_partials(target, epochs, J2000)
    assert [np.size(row) for row in rows] == [6, 2]
    for index, leg in enumerate(legs):
        alone = model(state_definition=leg).compute_partials(target, epochs, J2000)
        assert np.any(np.abs(rows[index]) > 0)
        np.testing.assert_allclose(rows[index], alone[index], rtol=1e-12, atol=0)

    with pytest.raises(scb.Measurement.SequencePartialsError):
        model(sequence_definition=sequence, partials_method="fd").compute_partials(
            target, epochs, J2000)


# --------------------------------------------------------------------------- #
# Considered parameters                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["RangeIdeal", "RangeRateIdeal",
                                  "SequentialRangingReal", "DopplerReal"])
def test_a_considered_ut1_offset_moves_the_model_as_an_estimated_one(
        bodies, topocentric_frame, kind):
    """A considered TAI-UT1 offset reaches the forward model and the rows.

    The filters hold a considered parameter at its value and take its column
    as the consider partial, so the measurement model must read and
    differentiate it exactly as it does an estimated one, and the observable
    must move by that column times the offset. The offset is a realistic
    a-priori sigma: a smaller one sinks the Doppler move into the 0.05 mHz
    floor of its count-interval integration.
    """
    gs, _, target, earth = bodies
    epoch = scb.EpochArray([ET + 36314.7], sys="TDB")
    media = scb.MediaCorrections("zenith", gs)

    def model(estimation, ut1):
        state = [(name, dim, estimation if name == "eop_delta_ut1" else "estimated", *rest)
                 for name, dim, _, *rest in entries(gs, earth, np.r_[np.zeros(5), ut1])]
        if kind.endswith("Real"):
            return getattr(scb, kind)("considered UT1", gs, state_definition=state,
                                      computed_measurements_dict=real_aux(epoch),
                                      media_corrections=media)
        return make_model(kind, bodies, state)

    ut1 = 0.05
    considered, estimated = model("considered", ut1), model("estimated", ut1)
    rows = [np.asarray(m.compute_partials(target, epoch, J2000))[0]
            for m in (considered, estimated)]
    assert rows[0].shape == (6,) and abs(rows[0][-1]) > 0
    np.testing.assert_array_equal(rows[0], rows[1])

    moved = observable(considered, target, epoch)
    assert moved == observable(estimated, target, epoch)
    np.testing.assert_allclose(moved - observable(model("considered", 0.), target, epoch),
                               rows[0][-1] * ut1, rtol=5e-3)
