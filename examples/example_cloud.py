"""
Python equivalent of examples/example_cloud.jl — 2x3 comparison figure of
particle distribution, AMR MFM, Cartesian NGP, AMR NGP, Cartesian SPH, AMR SPH.
"""

import numpy as np
import matplotlib.pyplot as plt
import particlegridmapper as pgm

pgm.init(num_threads=4)

# --- Parameters (same as example_cloud.jl) ---
n = 1000
boxsize = 1.0
rmax = 0.5 * boxsize
Mtot = 1.0
nx = ny = 128
ix, iy = 1, 2

# --- Generate spherical cloud with uniform r in [0, rmax) ---
rng = np.random.default_rng(42)
r = rmax * rng.uniform(0, 0.99, size=n)
phi = (2 * rng.uniform(size=n) - 1) * np.pi
cos_theta = 2 * rng.uniform(size=n) - 1
sin_theta = np.sqrt(1 - cos_theta**2)

positions = np.column_stack([
    r * sin_theta * np.cos(phi),
    r * sin_theta * np.sin(phi),
    r * cos_theta,
])

# --- Equal-mass particles, density ~ r^{-2} ---
mass = np.ones(n) * (Mtot / n)
radius = np.sqrt(np.sum(positions**2, axis=1)) / boxsize
rho = 2 * Mtot / (4 * np.pi) * radius**(-2)
volume = mass / rho

# --- Smoothing length: each particle encloses ~32 neighbors ---
Nngb = 32
hsml = ((Nngb * (Mtot / n) / (4 * np.pi / 3)) / rho) ** (1 / 3)

# --- Domain bounds ---
xmin = (-0.5 * boxsize,) * 3
xmax = (0.5 * boxsize,) * 3
center = (0.0, 0.0, 0.0)
boxsizes = (boxsize, boxsize, boxsize)

# --- Cartesian NGP (3D grid, then sum along LOS) ---
print("Cartesian NGP...")
grid_ngp = pgm.map_to_3d_ngp(positions, rho, mass, (nx, ny, nx), xmin, xmax)
dimlos = [d for d in range(3) if d != (ix - 1) and d != (iy - 1)][0]
image_cart_NGP = grid_ngp.sum(axis=dimlos)

# --- Cartesian SPH (2D projection) ---
print("Cartesian SPH...")
image_cart_SPH = pgm.map_to_2d(
    positions, rho, volume, hsml, xmin, xmax,
    ngrids=(nx, ny, nx), xaxis=ix, yaxis=iy,
    column=True, periodic=(True, True, True),
)

# --- AMR workflow ---
amr = pgm.AMRTree(positions, hsml, mass,
                   center=center, box_length=boxsizes)
max_depth = 10
print(f"max_depth = {max_depth}, true max_depth = {amr.max_depth}")

amr.set_max_depth(max_depth)
amr.balance()

# AMR SPH (run twice like the Julia example)
print("AMR SPH...")
tree_ngb = amr.copy()
amr.map_sph(rho, volume, positions, hsml, boxsizes, tree_ngb=tree_ngb)
amr.map_sph(rho, volume, positions, hsml, boxsizes)
image_AMR_SPH = amr.project_to_image(nx=nx, ny=ny, dimx=ix, dimy=iy,
                                      center=center, boxsizes=boxsizes)

# AMR MFM
print("AMR MFM...")
amr.map_mfm(rho, positions, hsml, boxsizes)
image_AMR_MFM = amr.project_to_image(nx=nx, ny=ny, dimx=ix, dimy=iy,
                                      center=center, boxsizes=boxsizes)

# AMR NGP
print("AMR NGP...")
amr.map_ngp(rho)
image_AMR_NGP = amr.project_to_image(nx=nx, ny=ny, dimx=ix, dimy=iy,
                                      center=center, boxsizes=boxsizes)

# --- 2x3 figure (same layout as Julia version) ---
ext = (-0.5 * boxsize, 0.5 * boxsize, -0.5 * boxsize, 0.5 * boxsize)
vmin, vmax = -1, 1.5
imshow_kw = dict(origin="lower", vmin=vmin, vmax=vmax,
                 interpolation="none", extent=ext)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].set_title("particle distribution")
axes[0, 0].plot(positions[:, 0], positions[:, 1], ".", ms=3, color="red")
axes[0, 0].axis(ext)

axes[0, 1].set_title("AMR MFM")
axes[0, 1].imshow(np.log10(image_AMR_MFM.T), **imshow_kw)

axes[0, 2].set_title("Cartesian NGP")
axes[0, 2].imshow(np.log10(image_cart_NGP.T), **imshow_kw)

axes[1, 0].set_title("AMR NGP")
axes[1, 0].imshow(np.log10(image_AMR_NGP.T), **imshow_kw)

axes[1, 1].set_title("Cartesian SPH")
axes[1, 1].imshow(np.log10(image_cart_SPH.T), **imshow_kw)

axes[1, 2].set_title("AMR SPH")
axes[1, 2].imshow(np.log10(image_AMR_SPH.T), **imshow_kw)

fig.tight_layout()
plt.savefig("compare_MFM_SPH_NGP_cloud.png", dpi=150)
print("Saved compare_MFM_SPH_NGP_cloud.png")
plt.show()
