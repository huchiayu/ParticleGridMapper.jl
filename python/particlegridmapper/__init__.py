"""
ParticleGridMapper — Python wrapper for ParticleGridMapper.jl

Map SPH/GIZMO particle data to Cartesian and AMR grids.

Usage
-----
    import particlegridmapper as pgm
    pgm.init(num_threads=4)  # optional, before first call

    image = pgm.map_to_2d(positions, field, volume, hsml, xmin, xmax)
"""

import os

from .cartesian import map_to_2d, map_to_3d, map_to_3d_ngp
from .amr import AMRTree

__all__ = [
    "init",
    "map_to_2d",
    "map_to_3d",
    "map_to_3d_ngp",
    "AMRTree",
]


def init(num_threads=None):
    """
    Initialize Julia runtime and load ParticleGridMapper.jl.

    Parameters
    ----------
    num_threads : int or None
        Number of Julia threads. Must be called before any other function
        (Julia thread count is fixed at startup). If None, uses the
        JULIA_NUM_THREADS environment variable or Julia's default.
    """
    # Prevent segfaults when Julia uses multiple threads via juliacall
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

    if num_threads is not None:
        os.environ["JULIA_NUM_THREADS"] = str(int(num_threads))

    from ._bridge import _ensure_initialized
    _ensure_initialized()
