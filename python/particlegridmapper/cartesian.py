"""
Cartesian grid mapping functions.
"""

import numpy as np

from ._bridge import (
    get_pgm,
    julia_array_to_numpy,
    positions_to_svec,
    to_float64_vec,
    to_ntuple_bool,
    to_ntuple_float,
    to_ntuple_int,
)


def map_to_2d(
    positions,
    field,
    volume,
    hsml,
    xmin,
    xmax,
    *,
    ngrids=(100, 100, 100),
    xaxis=1,
    yaxis=2,
    column=True,
    periodic=(True, True, True),
    threaded=True,
):
    """
    SPH interpolation of particle data onto a 2D grid (projection along LOS).

    Parameters
    ----------
    positions : (N, 3) float64 array — particle positions
    field : (N,) float64 array — field values
    volume : (N,) float64 array — particle volumes (mass/density)
    hsml : (N,) float64 array — smoothing lengths
    xmin, xmax : tuple of 3 floats — domain bounds
    ngrids : tuple of 3 ints — grid resolution per dimension
    xaxis, yaxis : int — 1-indexed projection axes
    column : bool — if True, return column density; else average
    periodic : tuple of 3 bools — periodic boundary conditions
    threaded : bool — use multithreaded version

    Returns
    -------
    2D numpy array of shape (ngrids[xaxis-1], ngrids[yaxis-1])
    """
    pgm = get_pgm()
    X = positions_to_svec(positions)
    f = to_float64_vec(field)
    v = to_float64_vec(volume)
    h = to_float64_vec(hsml)

    if threaded:
        result = pgm.map_particle_to_2Dgrid_loopP_thread(
            f, v, X, h,
            to_ntuple_float(xmin), to_ntuple_float(xmax),
            xaxis=xaxis, yaxis=yaxis, column=column,
            ngrids=to_ntuple_int(ngrids), pbc=to_ntuple_bool(periodic),
        )
    else:
        result = pgm.map_particle_to_2Dgrid_loopP(
            f, v, X, h,
            to_ntuple_float(xmin), to_ntuple_float(xmax),
            xaxis=xaxis, yaxis=yaxis, column=column,
            ngrids=to_ntuple_int(ngrids), pbc=to_ntuple_bool(periodic),
        )

    return julia_array_to_numpy(result)


def map_to_3d(
    positions,
    field,
    volume,
    hsml,
    xmin,
    xmax,
    *,
    ngrids=(100, 100, 100),
    periodic=(True, True, True),
    threaded=True,
):
    """
    SPH interpolation of particle data onto a 3D grid (threaded).

    Parameters
    ----------
    positions : (N, 3) float64 array — particle positions
    field : (N,) float64 array — field values
    volume : (N,) float64 array — particle volumes (mass/density)
    hsml : (N,) float64 array — smoothing lengths
    xmin, xmax : tuple of 3 floats — domain bounds
    ngrids : tuple of 3 ints — grid resolution per dimension
    periodic : tuple of 3 bools — periodic boundary conditions
    threaded : bool — use multithreaded version (only True supported for 3D)

    Returns
    -------
    3D numpy array of shape ngrids
    """
    if not threaded:
        raise ValueError(
            "Non-threaded 3D mapping is not available. Use threaded=True."
        )
    pgm = get_pgm()
    X = positions_to_svec(positions)
    f = to_float64_vec(field)
    v = to_float64_vec(volume)
    h = to_float64_vec(hsml)

    result = pgm.map_particle_to_3Dgrid_loopP_thread(
        f, v, X, h,
        to_ntuple_float(xmin), to_ntuple_float(xmax),
        ngrids=to_ntuple_int(ngrids), pbc=to_ntuple_bool(periodic),
    )
    return julia_array_to_numpy(result)


def map_to_3d_ngp(
    positions,
    field,
    mass,
    ngrids,
    xmin,
    xmax,
    *,
    threaded=True,
):
    """
    Nearest Grid Point (NGP) interpolation onto a 3D grid.

    Parameters
    ----------
    positions : (N, 3) float64 array — particle positions
    field : (N,) float64 array — field values
    mass : (N,) float64 array — particle masses
    ngrids : tuple of 3 ints — grid resolution
    xmin, xmax : tuple of 3 floats — domain bounds
    threaded : bool — use multithreaded version

    Returns
    -------
    3D numpy array of shape ngrids
    """
    pgm = get_pgm()
    X = positions_to_svec(positions)
    f = to_float64_vec(field)
    m = to_float64_vec(mass)

    if threaded:
        result = pgm.map_particle_to_3Dgrid_NGP_thread(
            f, m, X, to_ntuple_int(ngrids),
            to_ntuple_float(xmin), to_ntuple_float(xmax),
        )
    else:
        result = pgm.map_particle_to_3Dgrid_NGP(
            f, m, X, to_ntuple_int(ngrids),
            to_ntuple_float(xmin), to_ntuple_float(xmax),
        )
    return julia_array_to_numpy(result)
