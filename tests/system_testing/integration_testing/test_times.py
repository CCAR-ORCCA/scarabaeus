# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import pytest
import scarabaeus as scb
import spiceypy as spice
import numpy as np
import os

# load kernels
furnshKernelFilename = os.getcwd() + "/data/kernels/locked/locked_generic.tm"
scb.SpiceManager.load_kernel_from_mkfile(furnshKernelFilename)
# scb.SpiceManager.load_kernel('tutorials/supplementary/supp_data/kernels/OREX/ORX_SCLKSCET_00075.tsc')

sec, hr, min = scb.Units.get_units(['sec', 'hr', 'min'])


# all systems, representations, and combos that don't exist
@pytest.fixture(scope = 'module')
def systems():
    return ['TDB', 'ET', 'UTC', 'TT', 'TDT', 'TAI', 'GPS']

@pytest.fixture(scope = 'module')
def representations():
    return ['NUM', 'AWU', 'CAL', 'DOY', 'ISOC', 'ISOD',
            'JDTDB', 'JED', 'JDTDT', 'JDUTC']

@pytest.fixture(scope = 'module')
def invalid_combos():
    return {('TDB', 'JDTDT'), ('ET',  'JDTDT'),
            ('UTC', 'JDTDT'), ('TAI', 'JDTDT'),
            ('GPS', 'JDTDT'), ('UTC', 'NUM'),
            ('UTC', 'AWU'),   ('UTC', 'JDTDB'),
            ('UTC', 'JED'),   ('TT',  'JDTDB'),
            ('TDT', 'JDTDB'), ('TT',  'JED'),
            ('TDT', 'JED'),   ('TAI', 'JDTDB'),
            ('TAI', 'JED'),   ('GPS', 'JDTDB'),
            ('GPS', 'JED')}

@pytest.fixture(scope = 'module')
def zero_epochs():
    # get zero epochs in each system from SPICE
    return {'TDB': scb.EpochArray(0.0, sys = 'TDB'),
            'ET' : scb.EpochArray(0.0, sys = 'ET'),
            'UTC': scb.EpochArray('2000 JAN 01 12:00:00.000', sys = 'UTC', rep = 'CAL'),
            'TT' : scb.EpochArray(0.0, sys = 'TT'),
            'TDT': scb.EpochArray(0.0, sys = 'TDT'),
            'TAI': scb.EpochArray(0.0, sys = 'TAI'),
            'GPS': scb.EpochArray(0.0, sys = 'GPS')}

@pytest.fixture(scope = 'module')
def ten_epochs(zero_epochs):
    # and at 10 seconds after 0
    return {sys: scb.EpochArray._from_et(ep._times + 10.0, sys, 'NUM',
                                         ep._prec, ep._disp_len, ep._sc_id)
            for sys, ep in zero_epochs.items()}

@pytest.fixture(scope = 'module')
def vec_epochs(zero_epochs):
    # create vectors of epochs in each system
    vec = {sys: scb.EpochArray._from_et(np.array([ep._times, ep._times + 10.0, ep._times + 20.0]),
                                        sys, 'NUM', ep._prec, ep._disp_len, ep._sc_id)
           for sys, ep in zero_epochs.items() if sys != 'UTC'}

    zero_utc = zero_epochs['UTC']
    utc_cal_epochs = np.array([zero_utc._times, zero_utc._times + 10.0, zero_utc._times + 20.0])
    vec['UTC'] = scb.EpochArray._from_et(utc_cal_epochs, 'UTC', 'CAL',
                                         zero_utc._prec, zero_utc._disp_len, 
                                         zero_utc._sc_id)
    
    return vec

@pytest.fixture(scope = 'module')
def ten_vec_epochs(vec_epochs):
    # and vectors at +10
    return {sys: scb.EpochArray._from_et(ep._times + 10.0, sys, 'NUM',
                                         ep._prec, ep._disp_len, ep._sc_id)
            for sys, ep in vec_epochs.items()}

# easier check to see if system is synonym to another
check_synonym = lambda s: {'ET': 'TDB', 'TDT': 'TT', 'UTC': 'UTC'}.get(s, s)

def test_scalar_roundtrip(zero_epochs, invalid_combos, representations):
    """ All valid combos round-trip back to original TDB. """
    failures = []
    for sys, base in zero_epochs.items():
        base_et = float(base._times)
        for rep in representations:
            combo = (sys, rep)
            if combo in invalid_combos:
                # invalid combo should raise
                try:
                    ep_rep = base.to(rep=rep)
                    _      = ep_rep.times
                    failures.append(f'{sys}/{rep}: expected error but got none')
                except Exception:
                    pass  # correct
            else:
                try:
                    ep_rep  = base.to(rep=rep)
                    et_back = float(ep_rep.to(sys='TDB', rep='NUM')._times)
                    err     = abs(et_back - base_et)
                    if err >= 1e-9:
                        failures.append(f'{sys}/{rep}: round-trip err={err:.2e}')
                except Exception as e:
                    failures.append(f'{sys}/{rep}: unexpected ERR {type(e).__name__}: {e}')

    assert not failures, '\n' + '\n'.join(failures)

def test_internal_et(zero_epochs):
    """ Ensure zero epochs are using TDB internally. """
    failures = []

    # TDB and ET should be exactly 0
    for sys in ['TDB', 'ET']:
        et = float(zero_epochs[sys]._times)
        if et != 0.0:
            failures.append(f'{sys}: expected ET = 0.0, got {et}')

    # TAI and GPS should store positive ET (their zero is ahead of J2000)
    for sys in ['TAI', 'GPS']:
        et = float(zero_epochs[sys]._times)
        if et <= 0.0:
            failures.append(f'{sys}: expected positive ET for system zero, got {et}')

    # TAI/GPS display should be 0.0
    for sys in ['TAI', 'GPS']:
        disp = float(zero_epochs[sys].times.values)
        if abs(disp) >= 1e-9:
            failures.append(f'{sys}: expected display=0.0, got {disp}')

    # UTC display should be a string
    utc_disp = zero_epochs['UTC'].times
    if not isinstance(utc_disp, str):
        failures.append(f'UTC: expected string display, got {type(utc_disp)}')

    assert not failures, '\n' + '\n'.join(failures)

def test_vector_roundtrip(vec_epochs, invalid_combos, representations):
    """ All valid comboc round-trip correctly for vector epochs. """
    failures = []
    for sys, base in vec_epochs.items():
        for rep in representations:
            if (sys, rep) in invalid_combos:
                continue
            try:
                ep_rep  = base.to(rep=rep)
                et_back = ep_rep.to(sys='TDB', rep='NUM')._times
                max_err = float(np.max(np.abs(et_back - base._times)))
                if max_err >= 1e-9:
                    failures.append(f'{sys}/{rep}: max round-trip err={max_err:.2e}')
            except Exception as e:
                failures.append(f'{sys}/{rep}: unexpected ERR {type(e).__name__}: {e}')

    assert not failures, '\n' + '\n'.join(failures)

def test_interval():
    """ Ensure interval() works. """
    t0 = scb.EpochArray('2000 JAN 01 12:00:00.000', rep = 'NUM')
    tf = scb.EpochArray('2000 JAN 01 13:00:00.000', rep = 'NUM')

    ## step definition 
    # 15 min steps over 1 hour = 4 epochs
    ep_step = scb.EpochArray.interval(start = t0, end = tf, dt = scb.ArrayWUnits(15, min))
    assert len(ep_step) == 4, f'step interval: expected len = 4, got {len(ep_step)}'

    ## count definition
    ep_count = scb.EpochArray.interval(start=t0, end=tf, n_epochs=5)
    assert len(ep_count) == 5, f'count interval: expected len=5, got {len(ep_count)}'

    # verify spacing
    diffs = np.diff(ep_step._times)
    assert np.all(np.abs(diffs - 900.0) < 1e-9), f'step interval: uneven spacing {diffs}'

def test_len_and_duration(vec_epochs):
    """ Ensure that len() and duration() return correct values for all systems. """
    failures = []
    for sys, base in vec_epochs.items():
        l = len(base)
        if l != 3:
            failures.append(f'{sys}: expected len=3, got {l}')
        try:
            dur = base.duration()
            err = abs(float(dur.values) - 20.0)
            if err >= 1e-9:
                failures.append(f'{sys}: expected dur = 20.0, got {float(dur.values):.6f}')
        except Exception as e:
            failures.append(f'{sys}: duration ERR {type(e).__name__}: {e}')

    assert not failures, '\n' + '\n'.join(failures)

def test_getitem(vec_epochs):
    """ Ensure indexing works correcly. """
    failures = []
    for sys, base in vec_epochs.items():
        if sys == 'UTC':
            continue
        try:
            # scalar index
            ep0 = base[0]
            if ep0.size != 1:
                failures.append(f'{sys}[0]: expected size=1, got {ep0.size}')
            if ep0.system != sys:
                failures.append(f'{sys}[0]: expected system={sys}, got {ep0.system}')
            if abs(float(ep0._times) - float(base._times[0])) >= 1e-9:
                failures.append(f'{sys}[0]: ET mismatch')

            # slice index
            ep_slice = base[1:]
            if ep_slice.size != 2:
                failures.append(f'{sys}[1:]: expected size=2, got {ep_slice.size}')
            if ep_slice.system != sys:
                failures.append(f'{sys}[1:]: expected system={sys}, got {ep_slice.system}')
        except Exception as e:
            failures.append(f'{sys}: getitem ERR {type(e).__name__}: {e}')

    assert not failures, '\n' + '\n'.join(failures)


@pytest.mark.parametrize('awu', [pytest.param(scb.ArrayWUnits(25, sec), id='25 sec'),
                                 pytest.param(scb.ArrayWUnits(1,  hr),  id='1 hr')])
def test_addition(awu, zero_epochs, vec_epochs):
    """ EpochArray + AWU produces correct TDB and preserves system. """
    failures = []

    for label_prefix, epochs in [('scalar', zero_epochs), ('vector', vec_epochs)]:
        for sys, base in epochs.items():
            try:
                result    = base + awu
                result_et = result._times
                expected  = base._times + float(awu.convert_to(sec).values)
                err       = float(np.max(np.abs(result_et - expected)))
                if err >= 1e-9:
                    failures.append(f'{label_prefix} {sys}: err={err:.2e}')
                if result.system != sys:
                    failures.append(f'{label_prefix} {sys}: system={result.system}')
            except Exception as e:
                failures.append(f'{label_prefix} {sys}: ERR {type(e).__name__}: {e}')

    assert not failures, '\n' + '\n'.join(failures)

@pytest.mark.parametrize('awu', [pytest.param(scb.ArrayWUnits(25, sec), id='25 sec'),
                                 pytest.param(scb.ArrayWUnits(1,  hr),  id='1 hr')])
def test_subtraction_awu(awu, zero_epochs, vec_epochs):
    """ EpochArray - AWU produces correct TDB and preserves system. """
    failures = []
    for label_prefix, epochs in [('scalar', zero_epochs), ('vector', vec_epochs)]:
        for sys, base in epochs.items():
            try:
                result    = base - awu
                result_et = result._times
                expected  = base._times - float(awu.convert_to(sec).values)
                err       = float(np.max(np.abs(result_et - expected)))
                if err >= 1e-9:
                    failures.append(f'{label_prefix} {sys}: err={err:.2e}')
                if result.system != sys:
                    failures.append(f'{label_prefix} {sys}: system={result.system}')
            except Exception as e:
                failures.append(f'{label_prefix} {sys}: ERR {type(e).__name__}: {e}')

    assert not failures, '\n' + '\n'.join(failures)

@pytest.mark.parametrize('ten_key,zero_key', [pytest.param('ten_epochs',     'zero_epochs', id='scalar'),
                                              pytest.param('ten_vec_epochs', 'vec_epochs',  id='vector')])
def test_subtraction_epoch(request, ten_key, zero_key, systems):
    """ EpochArray - EpochArray gives 10 sec for same system, raises for different. """
    ten      = request.getfixturevalue(ten_key)
    zero     = request.getfixturevalue(zero_key)
    failures = []

    for sys_a in systems:
        for sys_b in systems:
            same = check_synonym(sys_a) == check_synonym(sys_b)
            try:
                result = ten[sys_a] - zero[sys_b]
                vals   = result.values
                arr    = vals if isinstance(vals, np.ndarray) else np.array([float(vals)])
                err    = float(np.max(np.abs(arr - 10.0)))
                if not same:
                    failures.append(f'{sys_a}-{sys_b}: expected error, got {err:.2e}')
                elif err >= 1e-9:
                    failures.append(f'{sys_a}-{sys_b}: err={err:.2e}')
            except Exception:
                if same:
                    failures.append(f'{sys_a}-{sys_b}: unexpected error for same system')

    assert not failures, '\n' + '\n'.join(failures)

def test_jdtdb_vector_value(vec_epochs):
    """ JDTDB rep for TDB vector should return Julian Dates ~2451545. """
    ep     = vec_epochs['TDB'].to(rep='JDTDB')
    values = ep.times
    # J2000 is JD 2451545.0 in TDB, vector is [0, 10, 20] ET seconds
    expected = np.array([2451545.0,
                         2451545.0 + 10.0  / 86400.0,
                         2451545.0 + 20.0  / 86400.0])
    assert np.all(np.abs(values - expected) < 1e-9)

def test_jed_vector_value(vec_epochs):
    """ JED rep should match JDTDB since they are the same. """
    jdtdb = vec_epochs['TDB'].to(rep='JDTDB').times
    jed   = vec_epochs['TDB'].to(rep='JED').times
    assert np.all(np.abs(jdtdb - jed) < 1e-9), f'different JED and JDTDB: {jdtdb} vs {jed}'

def test_jdtdt_vector_value(vec_epochs):
    """ JDTDT rep for TT vector should return Julian Dates near 2451545. """
    ep     = vec_epochs['TT'].to(rep='JDTDT')
    values = ep.times
    # TT zero is very close to J2000 — JD should be near 2451545.0
    assert np.all(np.abs(values - 2451545.0) < 1.0)

def test_jdutc_vector_value(vec_epochs):
    """ JDUTC rep should return strings starting with JD for all systems. """
    for sys in ['TDB', 'ET', 'TAI', 'GPS', 'TT', 'TDT']:
        ep     = vec_epochs[sys].to(rep='JDUTC')
        values = ep.times
        for v in values:
            assert str(v).startswith('JD ')

def test_cal_vector_value(vec_epochs):
    """ CAL rep for TDB zero vector should contain J2000 date. """
    ep     = vec_epochs['TDB'].to(rep='CAL')
    values = ep.times
    # TDB zero in UTC is ~11:58:55 on 2000 JAN 01
    assert all('2000' in v for v in values), f'CAL values do not contain year 2000: {values}'
    assert all('JAN' in v for v in values), f'CAL values do not contain JAN: {values}'