using ParticleGridMapper
using StaticArrays
using PyPlot

const BOXSIZE = 1.0

const N = 3 #spatial dimension
const T = Float64

Npart = 100000 #number of particles

#desired number of neighbor particles (ngbs) within a search radius (hsml)
const Nngb0 = 3

#searching radius that should contain roughly Nngb0 neighboring particles
const hsml0 = BOXSIZE * (Nngb0/(4*pi/3*Npart))^(1/N)

using Random
Random.seed!(1114)

#randomly distributing particles
X = [@SVector rand(N) for _ in 1:Npart]

topnode_length = @SVector(ones(N)) * BOXSIZE  #actual length of tree

center = topnode_length .* 0.5

hsml = ones(Npart) .* hsml0
mass = ones(Npart) #particle mass

part = [DataP2G{N,T}(SVector(X[i]), i, hsml[i], mass[i], 0.0, T[]) for i in eachindex(X)]

#build the tree
tree = buildtree(part, center, topnode_length);
gridAMR = Int8[]
@time get_AMRgrid!(gridAMR, tree)
num_nodes  = length(gridAMR[gridAMR.==1])
num_leaves = length(gridAMR[gridAMR.==0])
@show num_nodes, num_leaves

grid_volumes = T[]
get_AMRgrid_volumes!(grid_volumes,tree)
@show sum(grid_volumes), prod(topnode_length)


#search center
x = @SVector rand(N)

boxsizes = @SVector(ones(N)) * BOXSIZE  #for periodic B.C.


rho = ones(Npart)
volume = ones(Npart)

tree2 = deepcopy(tree)

println("serial version...")
@time map_particle_to_AMRgrid!(rho, volume, X, hsml, tree, tree, boxsizes)
println("parallel version...")
@time map_particle_to_AMRgrid_thread!(rho, volume, X, hsml, tree, tree, boxsizes)
rhoAMR = T[]
@time get_AMRfield!(rhoAMR, tree)
println("serial version, precomputed ngbs...")
@time map_particle_to_AMRgrid_knownNgb!(rho, volume, X, hsml, tree, tree, boxsizes)
println("parallel version, precomputed ngbs...")
@time map_particle_to_AMRgrid_knownNgb_thread!(rho, volume, X, hsml, tree, tree, boxsizes)
rhoAMR2 = T[]
@time get_AMRfield!(rhoAMR2, tree)

@show sum(grid_volumes.*rhoAMR), sum(grid_volumes.*rhoAMR2)

nx = ny = 2
image = zeros(T, nx, ny)
@time project_AMRgrid_to_image!(image, nx, ny, 1, 2, tree, center, boxsizes)

clf()
fig, ax = subplots(1, 1, figsize=(10, 10))
#ax.plot(getindex.(X,1), getindex.(X,2), ".", c="tab:blue", ms=10)
#@time ParticleGridMapper.plot_quadtree(tree,1,2,ax)
#for i in 1:Npart
#	draw_circle_periodic(X[i], hsml[i], 1, 2, boxsizes, ax)
#end
ax.axis([0,boxsizes[1],0,boxsizes[2]])
ax.set_aspect(1)
