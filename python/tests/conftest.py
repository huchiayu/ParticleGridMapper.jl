"""Shared fixtures for ParticleGridMapper Python tests."""

import numpy as np
import pytest

import particlegridmapper as pgm


@pytest.fixture(scope="session", autouse=True)
def init_julia():
    """Initialize Julia with 4 threads before any test runs."""
    pgm.init(num_threads=4)


@pytest.fixture()
def spherical_cloud():
    """
    Generate a spherical particle cloud, mirroring the Julia test setup.

    Returns dict with: positions, hsml, mass, volume, field_ones,
    boxsize, center, topnode_length, boxsizes.
    """
    rng = np.random.default_rng(1114)
    n_part = 1000
    boxsize = 0.807

    # Generate spherical distribution
    positions = np.empty((n_part, 3), dtype=np.float64)
    rmax = 0.5 * boxsize
    for i in range(n_part):
        r3d = rmax * 2
        while r3d > rmax:
            m_frac = rng.random() * 0.99
            r3d = rmax * m_frac
        phi = (2.0 * rng.random() - 1.0) * np.pi
        cos_theta = 2.0 * rng.random() - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        positions[i, 0] = r3d * sin_theta * np.cos(phi)
        positions[i, 1] = r3d * sin_theta * np.sin(phi)
        positions[i, 2] = r3d * cos_theta

    radius = np.sqrt(np.sum((positions / boxsize) ** 2, axis=1))
    m_tot = 1.0
    mass = np.full(n_part, m_tot / n_part)
    rho = 2 * m_tot / (4 * np.pi) * radius ** (-2)
    volume = mass / rho
    hsml = 0.7 * boxsize * radius ** (2.0 / 3.0)

    return {
        "positions": positions,
        "hsml": hsml,
        "mass": mass,
        "volume": volume,
        "field_ones": np.ones(n_part),
        "n_part": n_part,
        "boxsize": boxsize,
        "center": (0.0, 0.0, 0.0),
        "topnode_length": (boxsize, boxsize, boxsize),
        "boxsizes": (boxsize, boxsize, boxsize),
    }
