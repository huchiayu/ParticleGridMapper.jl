using ParticleGridMapper
using StaticArrays
using PyPlot
using HDF5

const N = 3 #spatial dimension
const T = Float64

const BOXSIZE = 1.0  #kpc
const max_depth = 10 #maximum refinement level (RADMC requires < 20)

const nx = ny = 128  #for the column density maps of H2 and CO (npix in RADMC)
const ix, iy = 1, 2  #axes of the column density maps

function read_snap(filename)

    header::Dict = h5readattr(filename, "/Header")
    boxsize::T = header["BoxSize"]
    time::T    = header["Time"]

    N_gas::Int64 = header["NumPart_ThisFile"][1]

    pos_gas::Matrix{T} = h5read(filename, "PartType0/Coordinates");
    vel_gas::Matrix{T} = h5read(filename, "PartType0/Velocities");
    rho::Vector{T}     = h5read(filename, "PartType0/Density");
    u::Vector{T}       = h5read(filename, "PartType0/InternalEnergy");
    m_gas::Vector{T}   = h5read(filename, "PartType0/Masses");
    hsml::Vector{T}    = h5read(filename, "PartType0/SmoothingLength");

    id_gas::Vector{Int64} = h5read(filename, "PartType0/ParticleIDs");

    return N_gas, pos_gas, vel_gas, rho, u, m_gas, hsml, id_gas, boxsize, time
end

function read_chemistry(fname)
    header = h5readattr(fname, "/Header")
    all_species = header["all_species"]
    abund_all   = h5read(fname, "Chemistry/Abundances");
    N_spec = size(abund_all)[1]
    d = Dict(all_species .=> collect(1:N_spec) );
    d[""] = d["PHOTON"] = d["CRP"] = d["CRPHOT"] = 0
    xH2 = abund_all[d["H2"], :]
    xHI = abund_all[d["H"], :]
    xHp = abund_all[d["H+"], :]
    xCO = abund_all[d["CO"], :]
    xCI = abund_all[d["C"], :]
    xCp = abund_all[d["C+"], :]
    xelec = abund_all[d["e-"], :]
    return xH2, xHI, xHp, xCO, xCI, xCp, xelec
end


function vec2svec(vec::Matrix{T}) where {T}
    svec = [SVector{size(vec,1),T}(vec[:,i]) for i in 1:size(vec,2)]
end
function mat2smat(mat::Array{T,3}) where {T}
    smat = [SMatrix{3,3,T}(mat[:,:,i]) for i in 1:size(mat,3)]
end



const XH = 0.71
const BOLTZMANN=1.3806e-16
const PROTONMASS=1.6726e-24
const GRAVCON=6.67e-8
const UnitMass_in_g = 1.989e43
const UnitLength_in_cm    = 3.085678e21
const UnitTime_in_s = 3.08568e+16
const UnitDensity_in_cgs = UnitMass_in_g / UnitLength_in_cm^3
const UnitDensity_in_pccm = UnitDensity_in_cgs/PROTONMASS
const Year_in_s = 31556926.
const fac_col = (UnitMass_in_g/UnitLength_in_cm^2)*(XH/PROTONMASS)


#let

i=500
@show i
snap = ""
if i < 10
    snap = "00" * string(i)
elseif i < 100
    snap = "0" * string(i)
else
    snap = string(i)
end

file_path = "/Users/chu/simulations/tallbox/SFSNPI_N1e5_gS10H250dS40H250_soft0p92_from200Myr_SFcut_Z1"
#file_path = "/Users/chu/simulations/tallbox/SFSNPI_N1e6_gS10H250dS40H250_soft0p43_from200Myr_SFcut_Z1/"
#file_path = "/Users/chu/simulations/tallbox/SFSNPI_N1e7_gS10H250dS40H250_soft0p2_from100Myr_Z1"
#file_path = "/Users/chu/simulations/turbbox/N32"
#file_path = "./"
filename = file_path * "/snap_" * snap * ".hdf5"

Npart, pos, vel, rho, u, mass, hsml, id_gas, boxsize, time = read_snap(filename);


X = vec2svec(pos);

vx = vel[1,:]
vy = vel[2,:]
vz = vel[3,:]

topnode_length = @SVector(ones(N)) * BOXSIZE  #actual length of tree

#center = topnode_length .* 0.5
center = @SVector [0.5,0.5,0.]

part = [DataP2G{N,T}(SVector(X[i]), i, hsml[i], mass[i], 0.0, Int64[]) for i in eachindex(X)]

#build the tree
println("building the tree...")
@time treeNgb = buildtree(part, center, topnode_length);
grid_volumes = get_AMRgrid_volumes(treeNgb)
@assert sum(grid_volumes) ≈ prod(topnode_length) #partition unity check
println( "true maximum depth of tree = ", log2( BOXSIZE / minimum(grid_volumes)^(1/3) ) )


max_depth_true = get_max_tree_depth(treeNgb)
@show max_depth, max_depth_true

treeAMR = deepcopy(treeNgb)
set_max_depth_AMR!(treeAMR, max_depth)
balance_all_level!(treeAMR)


println("getting AMR grid structure...")
@time gridAMR = get_AMRgrid(treeAMR)
num_nodes  = length(gridAMR[gridAMR.==1])
num_leaves = length(gridAMR[gridAMR.==0])
@show num_nodes, num_leaves


boxsizes = @SVector(ones(N)) * BOXSIZE  #for periodic B.C.

PARSEC   = 3.08572e18 #[cm] important to use the exact same value as in radmc to avoid unnecessary sub-pixeling
KILOPARSEC = 1e3 * PARSEC


#xmin, ymin, zmin = (center .- 0.5.*boxsizes) .* KILOPARSEC
#xmax, ymax, zmax = (center .+ 0.5.*boxsizes) .* KILOPARSEC
#!!!!!! RADMC default camera is at (0,0), so we let (xmin,xmax) = (-0.5,0.5) instead of (0,1)
#otherwise we will miss 3/4 of the image!!!
xmin, ymin, zmin = -0.5.*boxsizes .* KILOPARSEC
xmax, ymax, zmax = +0.5.*boxsizes .* KILOPARSEC


#the ascii version of amr_grid.inp isn't much bigger from the binary one so we don't bother to go for binary
println("writing to amr_grid.inp...")
@time open("amr_grid.inp", "w") do f
    write(f, "1\n")                       # iformat
    write(f, "1\n")                       # AMR grid style  (0=regular grid, no AMR)
    write(f, "0\n")                       # Coordinate system (Cartesian)
    write(f, "0\n")                       # gridinfo =0 to save disc space
    write(f, "1 1 1\n")                   # included dimension (1 1 1 for 3D)
    write(f, "1 1 1\n")                   # nx ny nz
    write(f, "$(max_depth) $(num_leaves) $(num_leaves+num_nodes)\n")  #depth, # of leaves, # of nodes
    write(f, "$(xmin) $(xmax)\n")                       #base grid coordinate in x
    write(f, "$(ymin) $(ymax)\n")                       #base grid coordinate in y
    write(f, "$(zmin) $(zmax)\n")                       #base grid coordinate in z
    for i in eachindex(gridAMR)
        write(f, "$(gridAMR[i])\n")                       #0 = leaf, 1 = nonleaf
    end
end


println("mapping vx...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, vx, X, hsml, boxsizes, treeNgb=treeNgb)
vxAMR_MFM = get_AMRfield(treeAMR)
println("mapping vy...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, vy, X, hsml, boxsizes)
vyAMR_MFM = get_AMRfield(treeAMR)
println("mapping vz...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, vz, X, hsml, boxsizes)
vzAMR_MFM = get_AMRfield(treeAMR)

open("gas_velocity.binp", "w") do f
    write(f, 1) # iformat
    write(f, 8) # double precision (only for binary input)
    write(f, num_leaves)
    for i in eachindex(vxAMR_MFM)
        write(f, vxAMR_MFM[i]*1e5, vyAMR_MFM[i]*1e5, vzAMR_MFM[i]*1e5) #in cm/sec (instead of km/sec)!!!
        #write(f, vxAMR_MFM[i]*0, vyAMR_MFM[i]*0, vzAMR_MFM[i]*0)
    end
end


println("reading chemistry data...")
fname_base = file_path * "/chem-neqH2Hp-noCOic-TF3-GrRec-" * snap * "-3"
fname = fname_base * ".hdf5"
xH2, xHI, xHp, xCO, xCI, xCp, xelec = read_chemistry(fname);


nHtot = rho.*(UnitDensity_in_pccm*XH)
println("mapping nHtot...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, nHtot, X, hsml, boxsizes)
nHtot_AMR_MFM = get_AMRfield(treeAMR)

println("mapping nH2...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, xH2.*nHtot, X, hsml, boxsizes)
nH2_AMR_MFM = get_AMRfield(treeAMR)

NH2 = project_AMRgrid_to_image(nx, ny, ix, iy, treeAMR, center, boxsizes);

println("mapping nCO...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, xCO.*nHtot, X, hsml, boxsizes)
nCO_AMR_MFM = get_AMRfield(treeAMR)

NCO = project_AMRgrid_to_image(nx, ny, ix, iy, treeAMR, center, boxsizes);

XHe = 0.1
mu = @. 1 / (XH * (xHI + xH2 + xHp + XHe + xelec));
temp = u.*mu .* (1e10*PROTONMASS/(1.5*BOLTZMANN));

println("mapping temp...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, temp, X, hsml, boxsizes)
tempAMR_MFM = get_AMRfield(treeAMR)

#ortho-to-para ratio = 3
fac_para = 0.25
fac_ortho = 1 - fac_para
open("numberdens_p-h2.binp", "w") do f
    write(f, 1) # iformat
    write(f, 8) # double precision (only for binary input)
    write(f, num_leaves)
    for i in eachindex(nH2_AMR_MFM)
        write(f, fac_para.*nH2_AMR_MFM[i])
    end
end
open("numberdens_o-h2.binp", "w") do f
    write(f, 1) # iformat
    write(f, 8) # double precision (only for binary input)
    write(f, num_leaves)
    for i in eachindex(nH2_AMR_MFM)
        write(f, fac_ortho.*nH2_AMR_MFM[i])
    end
end

open("numberdens_co.binp", "w") do f
    write(f, 1) # iformat
    write(f, 8) # double precision (only for binary input)
    write(f, num_leaves)
    for i in eachindex(nCO_AMR_MFM)
        write(f, nCO_AMR_MFM[i])
    end
end


open("gas_temperature.binp", "w") do f
    write(f, 1) # iformat
    write(f, 8) # double precision (only for binary input)
    write(f, num_leaves)
    for i in eachindex(tempAMR_MFM)
        write(f, tempAMR_MFM[i] < 1.99e3 ? tempAMR_MFM[i] : 1.99e3) #T_max = 2000 K in RADMC3D
    end
end

L_esc0   = 1e30
L_esc = ones(num_leaves).*L_esc0
open("escprob_lengthscale.binp", "w") do f
    write(f, 1) # iformat
    write(f, 8) # double precision (only for binary input)
    write(f, num_leaves)
    for i in eachindex(L_esc)
        write(f, L_esc[i])
    end
end

#vturb0 = 3.0 #km/s
#vturb = ones(num_leaves).*vturb0

v2 = @.(vx^2 + vy^2 + vz^2)
println("mapping v2...")
@time map_particle_to_AMRgrid_MFM!(treeAMR, v2, X, hsml, boxsizes)
v2AMR_MFM = get_AMRfield(treeAMR)
vturbAMR_MFM = @. (v2AMR_MFM - vxAMR_MFM^2 - vyAMR_MFM^2 - vzAMR_MFM^2)
vturbAMR_MFM[vturbAMR_MFM.<0] .= 0 #happens when it's very close to zero due to round-off error (usually at low-density regions in a coarse grid)
vturbAMR_MFM = sqrt.(vturbAMR_MFM)

open("microturbulence.binp", "w") do f
    write(f, 1) # iformat
    write(f, 8) # double precision (only for binary input)
    write(f, num_leaves)
    for i in eachindex(vturbAMR_MFM)
        write(f, vturbAMR_MFM[i]*1e5) #in cm/sec (instead of km/sec)!!!
    end
end

open("wavelength_micron.inp", "w") do f
    write(f, "2\n")                       # iformat
    write(f, "1e-1\n")
    write(f, "1e4\n")
end

# Write the lines.inp control file
open("lines.inp", "w") do f
    write(f, "2\n")
    write(f, "1\n")
    write(f, "co    leiden    0    0    2\n")
    write(f, "p-h2\n")
    write(f, "o-h2\n")
end

open("radmc3d.inp", "w") do f
    write(f, "lines_mode = 3\n")
    write(f, "lines_slowlvg_as_alternative = 1\n") #more stable LVG (but slower)
    #write(f, "lines_tbg = 0\n")
    write(f, "rto_style = 3\n") #binary output
    write(f, "writeimage_unformatted = 1\n") #binary image output
end


#save NH2 & NCO
fname = file_path * "/data_AMRmaps_H2CO_N" * string(nx) * "_" * snap * ".hdf5"
fid=h5open(fname,"w")
grp_part = create_group(fid,"AMRmaps");
h5write(fname, "AMRmaps/NH2", NH2)
h5write(fname, "AMRmaps/NCO", NCO)
close(fid)


#end #let
