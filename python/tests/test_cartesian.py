"""Tests for Cartesian grid mapping functions."""

import numpy as np
import pytest

import particlegridmapper as pgm


class TestNGP3D:
    """Tests for map_to_3d_ngp (Nearest Grid Point)."""

    def test_constant_field(self, spherical_cloud):
        """Mapping a constant field should produce 1.0 everywhere particles exist."""
        d = spherical_cloud
        ngrids = (16, 16, 16)
        bs = d["boxsize"]
        xmin = (-bs / 2, -bs / 2, -bs / 2)
        xmax = (bs / 2, bs / 2, bs / 2)

        grid = pgm.map_to_3d_ngp(
            d["positions"], d["field_ones"], d["mass"],
            ngrids, xmin, xmax, threaded=True,
        )

        assert grid.shape == ngrids
        # All occupied cells should have value ~1.0 (constant field)
        occupied = grid[grid > 0]
        np.testing.assert_allclose(occupied, 1.0)

    def test_threaded_vs_serial(self, spherical_cloud):
        """Threaded and serial NGP should give the same result."""
        d = spherical_cloud
        ngrids = (16, 16, 16)
        bs = d["boxsize"]
        xmin = (-bs / 2, -bs / 2, -bs / 2)
        xmax = (bs / 2, bs / 2, bs / 2)

        grid_t = pgm.map_to_3d_ngp(
            d["positions"], d["field_ones"], d["mass"],
            ngrids, xmin, xmax, threaded=True,
        )
        grid_s = pgm.map_to_3d_ngp(
            d["positions"], d["field_ones"], d["mass"],
            ngrids, xmin, xmax, threaded=False,
        )
        np.testing.assert_allclose(grid_t, grid_s)

    def test_output_dtype(self, spherical_cloud):
        """Output should be float64."""
        d = spherical_cloud
        ngrids = (8, 8, 8)
        bs = d["boxsize"]
        xmin = (-bs / 2, -bs / 2, -bs / 2)
        xmax = (bs / 2, bs / 2, bs / 2)

        grid = pgm.map_to_3d_ngp(
            d["positions"], d["field_ones"], d["mass"],
            ngrids, xmin, xmax,
        )
        assert grid.dtype == np.float64


class TestSPH2D:
    """Tests for map_to_2d (SPH projection)."""

    def test_runs_threaded(self, spherical_cloud):
        """Basic smoke test for threaded 2D SPH mapping."""
        d = spherical_cloud
        ngrids = (16, 16, 16)
        bs = d["boxsize"]
        xmin = (-bs / 2, -bs / 2, -bs / 2)
        xmax = (bs / 2, bs / 2, bs / 2)

        image = pgm.map_to_2d(
            d["positions"], d["field_ones"], d["volume"], d["hsml"],
            xmin, xmax, ngrids=ngrids, threaded=True,
        )
        assert image.ndim == 2
        assert image.shape == (ngrids[0], ngrids[1])  # default xaxis=1, yaxis=2

    def test_runs_serial(self, spherical_cloud):
        """Basic smoke test for serial 2D SPH mapping."""
        d = spherical_cloud
        ngrids = (16, 16, 16)
        bs = d["boxsize"]
        xmin = (-bs / 2, -bs / 2, -bs / 2)
        xmax = (bs / 2, bs / 2, bs / 2)

        image = pgm.map_to_2d(
            d["positions"], d["field_ones"], d["volume"], d["hsml"],
            xmin, xmax, ngrids=ngrids, threaded=False,
        )
        assert image.ndim == 2


class TestSPH3D:
    """Tests for map_to_3d (SPH 3D grid)."""

    def test_runs(self, spherical_cloud):
        """Basic smoke test for threaded 3D SPH mapping."""
        d = spherical_cloud
        ngrids = (16, 16, 16)
        bs = d["boxsize"]
        xmin = (-bs / 2, -bs / 2, -bs / 2)
        xmax = (bs / 2, bs / 2, bs / 2)

        grid = pgm.map_to_3d(
            d["positions"], d["field_ones"], d["volume"], d["hsml"],
            xmin, xmax, ngrids=ngrids,
        )
        assert grid.shape == ngrids
        assert grid.dtype == np.float64

    def test_non_threaded_raises(self, spherical_cloud):
        """3D SPH only supports threaded mode."""
        d = spherical_cloud
        bs = d["boxsize"]
        xmin = (-bs / 2, -bs / 2, -bs / 2)
        xmax = (bs / 2, bs / 2, bs / 2)

        with pytest.raises(ValueError):
            pgm.map_to_3d(
                d["positions"], d["field_ones"], d["volume"], d["hsml"],
                xmin, xmax, threaded=False,
            )
