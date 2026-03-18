"""
AMR grid mapping via OctreeBH, wrapped in a stateful AMRTree class.
"""

import numpy as np

from ._bridge import (
    get_bang,
    get_jl,
    get_pgm,
    julia_array_to_numpy,
    make_particles,
    positions_to_svec,
    to_float64_vec,
    to_svector,
)


class AMRTree:
    """
    Wraps the OctreeBH tree and ParticleGridMapper AMR functions.

    Parameters
    ----------
    positions : (N, 3) float64 array — particle positions
    hsml : (N,) float64 array — smoothing lengths
    mass : (N,) float64 array — particle masses
    center : tuple of 3 floats — tree center (default origin)
    box_length : tuple of 3 floats — tree box side lengths (default 1.0)
    """

    def __init__(self, positions, hsml, mass, center=(0.0, 0.0, 0.0),
                 box_length=(1.0, 1.0, 1.0)):
        pgm = get_pgm()

        particles = make_particles(positions, hsml, mass)
        center_sv = to_svector(center)
        box_sv = to_svector(box_length)

        self._tree = pgm.buildtree(particles, center_sv, box_sv)
        self._center = center_sv
        self._box_length = box_sv

    @property
    def max_depth(self):
        """Maximum depth of the tree."""
        pgm = get_pgm()
        return int(pgm.get_max_tree_depth(self._tree))

    def set_max_depth(self, depth):
        """Limit AMR refinement to the given depth."""
        get_bang("set_max_depth_AMR!")(self._tree, int(depth))

    def balance(self):
        """Apply 2:1 refinement balance constraint."""
        get_bang("balance_all_level!")(self._tree)

    def map_ngp(self, field, serial=False):
        """Map field to AMR cells using Nearest Grid Point."""
        f = to_float64_vec(field)
        get_bang("map_particle_to_AMRgrid_NGP!")(self._tree, f, serial=serial)

    def map_sph(self, field, volume, positions, hsml, boxsizes,
                tree_ngb=None, serial=False):
        """
        Map field to AMR cells using SPH kernel interpolation.

        Parameters
        ----------
        field : (N,) float64 — field values
        volume : (N,) float64 — particle volumes
        positions : (N, 3) float64 — particle positions
        hsml : (N,) float64 — smoothing lengths
        boxsizes : tuple of 3 floats — box sizes for periodic BC
        tree_ngb : AMRTree or None — neighbor tree (defaults to self)
        serial : bool — run single-threaded
        """
        f = to_float64_vec(field)
        v = to_float64_vec(volume)
        X = positions_to_svec(positions)
        h = to_float64_vec(hsml)
        bs = to_svector(boxsizes)
        ngb = tree_ngb._tree if tree_ngb is not None else self._tree
        get_bang("map_particle_to_AMRgrid_SPH!")(
            self._tree, f, v, X, h, bs, treeNgb=ngb, serial=serial
        )

    def map_mfm(self, field, positions, hsml, boxsizes,
                tree_ngb=None, serial=False):
        """
        Map field to AMR cells using MFM kernel interpolation.

        Parameters
        ----------
        field : (N,) float64 — field values
        positions : (N, 3) float64 — particle positions
        hsml : (N,) float64 — smoothing lengths
        boxsizes : tuple of 3 floats — box sizes for periodic BC
        tree_ngb : AMRTree or None — neighbor tree (defaults to self)
        serial : bool — run single-threaded
        """
        f = to_float64_vec(field)
        X = positions_to_svec(positions)
        h = to_float64_vec(hsml)
        bs = to_svector(boxsizes)
        ngb = tree_ngb._tree if tree_ngb is not None else self._tree
        get_bang("map_particle_to_AMRgrid_MFM!")(
            self._tree, f, X, h, bs, treeNgb=ngb, serial=serial
        )

    def get_grid(self):
        """Return AMR grid structure as int8 array (0=leaf, 1=internal)."""
        pgm = get_pgm()
        return julia_array_to_numpy(pgm.get_AMRgrid(self._tree))

    def get_field(self):
        """Return AMR field values as float64 array."""
        pgm = get_pgm()
        return julia_array_to_numpy(pgm.get_AMRfield(self._tree))

    def get_volumes(self):
        """Return AMR cell volumes as float64 array."""
        pgm = get_pgm()
        return julia_array_to_numpy(pgm.get_AMRgrid_volumes(self._tree))

    def set_field(self, field):
        """Set field values on AMR cells."""
        pgm = get_pgm()
        f = to_float64_vec(field)
        pgm.set_AMRfield(f, self._tree)

    def project_to_image(self, nx=128, ny=128, dimx=1, dimy=2,
                         center=None, boxsizes=None, serial=False):
        """
        Project 3D AMR field to a 2D image.

        Parameters
        ----------
        nx, ny : int — image resolution
        dimx, dimy : int — 1-indexed projection axes
        center : tuple of 3 floats or None (uses tree center)
        boxsizes : tuple of 3 floats or None (uses tree box_length)
        serial : bool — run single-threaded

        Returns
        -------
        2D numpy array of shape (nx, ny)
        """
        pgm = get_pgm()
        c = to_svector(center) if center is not None else self._center
        bs = to_svector(boxsizes) if boxsizes is not None else self._box_length
        result = pgm.project_AMRgrid_to_image(
            nx, ny, dimx, dimy, self._tree, c, bs, serial=serial
        )
        return julia_array_to_numpy(result)

    def copy(self):
        """Return a deep copy of this AMRTree."""
        jl = get_jl()
        new = object.__new__(AMRTree)
        new._tree = jl.deepcopy(self._tree)
        new._center = self._center
        new._box_length = self._box_length
        return new
