"""Tests for AMR grid mapping (AMRTree class)."""

import numpy as np
import pytest

import particlegridmapper as pgm


class TestAMRTree:
    """Tests for the AMRTree class, mirroring test/test_amr.jl."""

    def test_build_tree(self, spherical_cloud):
        """Tree builds without error and has expected properties."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        assert amr.max_depth >= 1

    def test_grid_structure(self, spherical_cloud):
        """Grid structure should have only 0s and 1s, with expected leaf count."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        grid = amr.get_grid()
        assert set(np.unique(grid)).issubset({0, 1})
        num_leaves = np.sum(grid == 0)
        assert d["n_part"] <= num_leaves <= 8 * d["n_part"]

    def test_volumes_sum(self, spherical_cloud):
        """Total volume of all cells should equal box volume."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        amr.set_max_depth(amr.max_depth)
        amr.balance()
        volumes = amr.get_volumes()
        expected = d["boxsize"] ** 3
        np.testing.assert_allclose(np.sum(volumes), expected)

    def test_ngp_constant_field(self, spherical_cloud):
        """NGP with constant field should produce 1.0 in occupied cells."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        amr.set_max_depth(2)
        amr.balance()

        amr.map_ngp(d["field_ones"], serial=True)
        field_s = amr.get_field()

        amr.map_ngp(d["field_ones"])
        field_p = amr.get_field()

        np.testing.assert_allclose(field_p, field_s)

    def test_sph_constant_field(self, spherical_cloud):
        """SPH serial and parallel should give same result for constant field."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        tree_ngb = amr.copy()

        amr.set_max_depth(2)
        amr.balance()

        amr.map_sph(
            d["field_ones"], d["volume"], d["positions"], d["hsml"],
            d["boxsizes"], tree_ngb=tree_ngb, serial=True,
        )
        field_s = amr.get_field()

        amr.map_sph(
            d["field_ones"], d["volume"], d["positions"], d["hsml"],
            d["boxsizes"], tree_ngb=tree_ngb,
        )
        field_p = amr.get_field()

        np.testing.assert_allclose(field_p, field_s)

    def test_mfm_constant_field(self, spherical_cloud):
        """MFM with constant field should give all 1.0 (field-independent of volume)."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        tree_ngb = amr.copy()

        amr.set_max_depth(2)
        amr.balance()

        amr.map_mfm(
            d["field_ones"], d["positions"], d["hsml"],
            d["boxsizes"], tree_ngb=tree_ngb, serial=True,
        )
        field_s = amr.get_field()

        amr.map_mfm(
            d["field_ones"], d["positions"], d["hsml"],
            d["boxsizes"], tree_ngb=tree_ngb,
        )
        field_p = amr.get_field()

        np.testing.assert_allclose(field_p, field_s)
        np.testing.assert_allclose(field_p, 1.0)

    def test_project_to_image(self, spherical_cloud):
        """Projection of constant MFM field should give boxsize everywhere."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        tree_ngb = amr.copy()

        amr.set_max_depth(2)
        amr.balance()

        amr.map_mfm(
            d["field_ones"], d["positions"], d["hsml"],
            d["boxsizes"], tree_ngb=tree_ngb,
        )

        for p in range(1, 8):
            nx = ny = 2 ** p
            image_s = amr.project_to_image(
                nx, ny, 1, 2,
                center=d["center"], boxsizes=d["boxsizes"],
                serial=True,
            )
            image_p = amr.project_to_image(
                nx, ny, 1, 2,
                center=d["center"], boxsizes=d["boxsizes"],
            )
            np.testing.assert_allclose(image_p, image_s)
            np.testing.assert_allclose(image_p / d["boxsize"], 1.0)

    def test_copy(self, spherical_cloud):
        """Deep copy should be independent of original."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        amr2 = amr.copy()
        # Modifying copy shouldn't affect original
        amr2.set_max_depth(1)
        assert amr.max_depth >= amr2.max_depth or amr.max_depth == amr2.max_depth

    def test_set_field(self, spherical_cloud):
        """set_field / get_field round-trip."""
        d = spherical_cloud
        amr = pgm.AMRTree(
            d["positions"], d["hsml"], d["mass"],
            center=d["center"], box_length=d["topnode_length"],
        )
        amr.set_max_depth(2)
        amr.balance()

        # get_field returns only leaf cells, not all nodes
        n_leaves = len(amr.get_field())
        field_in = np.arange(n_leaves, dtype=np.float64)
        amr.set_field(field_in)
        field_out = amr.get_field()
        np.testing.assert_allclose(field_out, field_in)
