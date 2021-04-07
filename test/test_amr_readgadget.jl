using ParticleGridMapper
using StaticArrays
using PyPlot
using HDF5

function read_snap(filename)

    T=Float64
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


const BOXSIZE = 1.0

const N = 3 #spatial dimension
const T = Float64


#function main()
i=500
#i=600
#i=112
@show i
snap = ""
if i < 10
    snap = "00" * string(i)
elseif i < 100
    snap = "0" * string(i)
else
    snap = string(i)
end

file_path = "/Users/chu/simulations/tallbox/SFSNPI_N1e6_gS10H250dS40H250_soft0p43_from200Myr_SFcut_Z1"
#file_path = "/Users/chu/simulations/tallbox/SFSNPI_N1e5_gS10H250dS40H250_soft0p92_from200Myr_SFcut_Z1"
#file_path = "/Users/chu/simulations/turbbox/N32"
#file_path = "./"
filename = file_path * "/snap_" * snap * ".hdf5"

N_gas, pos, vel, rho, u, mass, hsml, id_gas, boxsize, time = read_snap(filename);

#pos = pos[1:2,:]

Npart = N_gas #number of particles
X = vec2svec(pos);
#Vel = vec2svec(vel);
vx = vel[1,:]
vy = vel[2,:]
vz = vel[3,:]


topnode_length = @SVector(ones(N)) * BOXSIZE  #actual length of tree

#center = topnode_length .* 0.5
center = @SVector [0.5,0.5,0.]


part = [DataP2G{N,T}(SVector(X[i]), i, hsml[i], mass[i], 0.0, Int64[]) for i in eachindex(X)]

#build the tree
@time tree = buildtree(part, center, topnode_length);

gridAMR = Int8[]
@time get_AMRgrid!(gridAMR, tree)
num_nodes  = length(gridAMR[gridAMR.==1])
num_leaves = length(gridAMR[gridAMR.==0])
@show num_nodes, num_leaves

grid_volumes = T[]
get_AMRgrid_volumes!(grid_volumes,tree)
@show sum(grid_volumes), prod(topnode_length)


#@time open("amr_grid.inp", "w") do f
#    write(f, "1\n")                       # iformat
#    write(f, "0\n")                       # AMR grid style  (0=regular grid, no AMR)
#    write(f, "0\n")                       # Coordinate system
#    write(f, "0\n")                       # gridinfo
#    write(f, "1 1 1\n")                   # Include x,y,z coordinate
#    write_radmcAMRinfo!(f, tree)
#    write(f, "0\n")                       #0 = leaf
#end


boxsizes = @SVector(ones(N)) * BOXSIZE  #for periodic B.C.

tree2 = deepcopy(tree)

volume = mass ./ rho

println("serial version...")
@time map_particle_to_AMRgrid!(rho, volume, X, hsml, tree, tree, boxsizes)
rhoAMR = T[]
get_AMRfield!(rhoAMR, tree)
@show sum(grid_volumes.*rhoAMR)

println("parallel version...")
@time map_particle_to_AMRgrid_thread!(rho, volume, X, hsml, tree, tree, boxsizes)
rhoAMR = T[]
get_AMRfield!(rhoAMR, tree)
@show sum(grid_volumes.*rhoAMR)

println("serial version, precomputed ngbs...")
#@time map_particle_to_AMRgrid_knownNgb!(rho, volume, X, hsml, tree, tree, boxsizes)
@time map_particle_to_AMRgrid!(rho, volume, X, hsml, tree, tree, boxsizes, knownNgb=true)
rhoAMR = T[]
get_AMRfield!(rhoAMR, tree)
@show sum(grid_volumes.*rhoAMR)

println("parallel version, precomputed ngbs...")
#@time map_particle_to_AMRgrid_knownNgb_thread!(rho, volume, X, hsml, tree, tree, boxsizes)
@time map_particle_to_AMRgrid_thread!(rho, volume, X, hsml, tree, tree, boxsizes, knownNgb=true)
rhoAMR = T[]
get_AMRfield!(rhoAMR, tree)
@show sum(grid_volumes.*rhoAMR)
#rhoAMR3 = T[]
#@time get_AMRfield!(rhoAMR3, tree3)

nx = ny = 512
image = zeros(T, nx, ny);
@time project_AMRgrid_to_image!(image, nx, ny, 1, 2, tree, center, boxsizes)
image2 = zeros(T, nx, ny);
@time project_AMRgrid_to_image_thread!(image2, nx, ny, 1, 2, tree, center, boxsizes);
@show all(image .≈ image2)

#image, X, tree
clf()
fig, ax = subplots(1, 1, figsize=(10, 10))
imshow(log10.(image'), origin="lower", interpolation="none")
#end

#=
clf()
fig, ax = subplots(1, 1, figsize=(10, 10))
ax.plot(getindex.(X,1), getindex.(X,2), ".", c="tab:blue", ms=10)
@time ParticleGridMapper.plot_quadtree(tree,1,2,ax)
#for i in 1:Npart
#draw_circle_periodic(X[i], hsml[i], 1, 2, boxsizes, ax)
#end
ax.axis([0,boxsizes[1],0,boxsizes[2]])
ax.set_aspect(1)
#image, X, tree = main()
=#
