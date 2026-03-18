"""
Julia initialization and type conversion utilities for ParticleGridMapper.jl.
"""

import os
import threading

import numpy as np

_jl = None
_pgm = None
_bridge = None
_lock = threading.Lock()
_initialized = False

# Cache for Julia functions with ! in their names (not valid Python identifiers)
_bang_funcs = {}


def _ensure_initialized():
    """Initialize Julia and load ParticleGridMapper if not already done."""
    global _jl, _pgm, _bridge, _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        from juliacall import Main as jl

        _jl = jl

        # Activate the Julia project
        pkg_dir = os.environ.get(
            "PARTICLEGRIDMAPPER_JL_PATH",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )
        _jl.seval("using Pkg")
        if os.path.isfile(os.path.join(pkg_dir, "Project.toml")):
            _jl.seval(f'Pkg.activate("{pkg_dir}")')
        else:
            # Fall back to registry install
            _jl.seval('Pkg.add("ParticleGridMapper")')

        _jl.seval("using ParticleGridMapper")
        _jl.seval("using StaticArrays")
        _jl.seval("using OctreeBH")

        # Load helper bridge module
        _jl.seval("""
        module _PGMBridge
            using StaticArrays
            using Main.ParticleGridMapper: DataP2G
            function to_svec3(flat_in, n::Int)
                flat = Vector{Float64}(flat_in)
                [SVector{3,Float64}(flat[3*(i-1)+1], flat[3*(i-1)+2], flat[3*i]) for i in 1:n]
            end
            to_jl_vec(x) = Vector{Float64}(x)
            function make_particles(flat_pos_in, hsml_in, mass_in, n::Int)
                flat_pos = Vector{Float64}(flat_pos_in)
                hsml = Vector{Float64}(hsml_in)
                mass = Vector{Float64}(mass_in)
                [DataP2G{3,Float64}(
                    SVector{3,Float64}(flat_pos[3*(i-1)+1], flat_pos[3*(i-1)+2], flat_pos[3*i]),
                    i, hsml[i], mass[i], 0.0, Float64[]
                ) for i in 1:n]
            end
        end
        """)

        _pgm = _jl.ParticleGridMapper
        _bridge = _jl._PGMBridge

        # Pre-cache Julia functions with ! in their names
        _bang_funcs["set_max_depth_AMR!"] = _jl.seval("ParticleGridMapper.set_max_depth_AMR!")
        _bang_funcs["balance_all_level!"] = _jl.seval("ParticleGridMapper.balance_all_level!")
        _bang_funcs["map_particle_to_AMRgrid_NGP!"] = _jl.seval("ParticleGridMapper.map_particle_to_AMRgrid_NGP!")
        _bang_funcs["map_particle_to_AMRgrid_SPH!"] = _jl.seval("ParticleGridMapper.map_particle_to_AMRgrid_SPH!")
        _bang_funcs["map_particle_to_AMRgrid_MFM!"] = _jl.seval("ParticleGridMapper.map_particle_to_AMRgrid_MFM!")

        _initialized = True


def get_jl():
    """Return the Julia Main module, initializing if needed."""
    _ensure_initialized()
    return _jl


def get_pgm():
    """Return the ParticleGridMapper module."""
    _ensure_initialized()
    return _pgm


def get_bridge():
    """Return the _PGMBridge helper module."""
    _ensure_initialized()
    return _bridge


def get_bang(name):
    """Return a cached Julia function with ! in its name."""
    _ensure_initialized()
    return _bang_funcs[name]


# ---------- Type conversion helpers ----------

def positions_to_svec(positions):
    """Convert (N, 3) numpy array to Vector{SVector{3,Float64}}."""
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    n = positions.shape[0]
    flat = positions.ravel()
    return get_bridge().to_svec3(flat, n)


def to_float64_vec(arr):
    """Convert 1D numpy array to Julia Vector{Float64}."""
    np_arr = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    return get_bridge().to_jl_vec(np_arr)


def to_ntuple_float(tup):
    """Convert Python tuple of floats to Julia NTuple."""
    return tuple(float(x) for x in tup)


def to_ntuple_int(tup):
    """Convert Python tuple of ints to Julia NTuple."""
    return tuple(int(x) for x in tup)


def to_ntuple_bool(tup):
    """Convert Python tuple of bools to Julia NTuple."""
    return tuple(bool(x) for x in tup)


def to_svector(tup):
    """Convert Python tuple to Julia SVector{3,Float64}."""
    jl = get_jl()
    return jl.seval(f"SVector{{3,Float64}}({float(tup[0])}, {float(tup[1])}, {float(tup[2])})")


def julia_array_to_numpy(jl_arr):
    """Convert a Julia Array to a numpy array."""
    return np.asarray(jl_arr)


def make_particles(positions, hsml, mass):
    """Create Vector{DataP2G{3,Float64}} from numpy arrays."""
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    n = positions.shape[0]
    flat_pos = positions.ravel()
    hsml_vec = np.ascontiguousarray(hsml, dtype=np.float64).ravel()
    mass_vec = np.ascontiguousarray(mass, dtype=np.float64).ravel()
    return get_bridge().make_particles(flat_pos, hsml_vec, mass_vec, n)
