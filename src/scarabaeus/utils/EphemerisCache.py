# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
"""Shared builder for the Rust :class:`EphemerisCache`.

Reads Chebyshev records for the requested bodies (and any center bodies
needed for frame chaining) directly out of the loaded SPK kernels, restricted
to the propagation interval. Originally part of ``ThreeBodyGravity``; now
shared so other models (e.g. the SRP models' Sun lookup) can build a cache
too.
"""
import scarabaeus.utils.NumpyWrapper as np


class NoLoadedSPKError(Exception):
    pass


def build_ephemeris_cache(body_ids, t0: float, tf: float):
    """
    Build a Rust :class:`EphemerisCache` for ``body_ids`` over ``[t0, tf]``.

    Parameters
    ----------
    body_ids : list of int
        SPICE IDs to cache. The FIRST entry is the primary/observer body.
    t0, tf : float
        Interval to cache, in ephemeris seconds past J2000 (TDB). Callers
        should include any epoch buffer they need.

    Returns
    -------
    EphemerisCache
        Cache covering ``body_ids`` plus any center bodies required for
        chaining, appended after them (see ``EphemerisCache.ids``).

    Raises
    ------
    NoLoadedSPKError
        If no SPK kernel is loaded.
    ValueError
        If a requested body has no segment in any loaded SPK.
    RuntimeError
        If a segment has an unsupported SPK type (only types 2 and 3 are
        readable as Chebyshev position records).
    """
    import spiceypy as spice
    from scarabaeus.rust import EphemerisCache

    SPK_TYPES = (2, 3)

    loaded_spks = [spice.kdata(i, 'SPK')[0] for i in range(spice.ktotal('SPK'))]
    if not loaded_spks:
        raise NoLoadedSPKError('Could not locate any SPK file for caching.')

    ## read through all loaded SPKs. duplicates are overridden if a newer loaded SPK also includes it
    segment_index = {}
    for spk in loaded_spks:
        # get handle and open DAF
        handle = spice.dafopr(spk)
        spice.dafbfs(handle)
        found = spice.daffna()
        while found:
            # get segment info
            descr  = spice.dafgs()
            _, ic = spice.dafus(descr, 2, 6)
            body_id  = int(ic[0])
            # save spk_file, center_id, spk_type, addr_start, addr_end
            segment_index[body_id] = (spk, int(ic[1]), int(ic[3]), int(ic[4]), int(ic[5]))
            found = spice.daffna()
        spice.dafcls(handle)

    all_ids = list(body_ids)

    ## get Chebyshevs for all requested bodies and their primaries necessary for frame chaining in the cache
    # intiialize
    seen, to_process = set(all_ids), list(all_ids)
    center_ids, coeffs_by_id = {}, {}

    # look through each body
    while to_process:
        body_id = to_process.pop(0)
        info = segment_index.get(body_id)
        if info is None:
            raise ValueError(f'body ID {body_id} not found as a segment in any loaded SPK')

        spk, center_id, spk_type, addr_start, addr_end = info
        if spk_type not in SPK_TYPES:
            raise RuntimeError(f'Cannot read SPK segment for body {body_id}: type {spk_type} '
                               f'not supported. Accepted types are: {SPK_TYPES}.')

        # open DAF and check its type
        handle = spice.dafopr(spk)
        meta = spice.dafgda(handle, addr_end - 3, addr_end)
        init, intlen, rsize, n = meta
        rsize, nrec = int(rsize), int(n)
        n_coeffs = (rsize - 2) // (3 if spk_type == 2 else 6)   # type 2 is just pos, type 3 also vel

        # only grab chebys for the prop interval + given epoch buffer
        first_idx = max(0, int((t0 - init) // intlen) - 1)
        last_idx  = min(nrec - 1, int((tf - init) // intlen) + 1)
        n_needed  = last_idx - first_idx + 1

        data_start = addr_start + first_idx * rsize
        data_end   = data_start + n_needed * rsize - 1
        all_data = np.asarray(spice.dafgda(handle, data_start, data_end)).reshape(n_needed, rsize)
        spice.dafcls(handle)

        t_mids = all_data[:, 0].tolist()
        t_rads = all_data[:, 1].tolist()
        xs = all_data[:, 2:2+n_coeffs].tolist()
        ys = all_data[:, 2+n_coeffs:2+2*n_coeffs].tolist()
        zs = all_data[:, 2+2*n_coeffs:2+3*n_coeffs].tolist()
        body_coeffs = sorted(zip(t_mids, t_rads, xs, ys, zs), key=lambda r: r[0] - r[1])

        # save if don't already have it
        coeffs_by_id[body_id] = body_coeffs
        center_ids[body_id] = center_id
        if center_id != 0 and center_id not in seen:
            all_ids.append(center_id)
            seen.add(center_id)
            to_process.append(center_id)

    ## finish formatting and build the cache
    all_coeffs     = [coeffs_by_id[body_id] for body_id in all_ids]
    all_center_ids = [center_ids[body_id]   for body_id in all_ids]
    return EphemerisCache(all_coeffs, all_ids, all_center_ids)
