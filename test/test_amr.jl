using ParticleGridMapper
using StaticArrays
using PyPlot

const N = 3 #spatial dimension
const T = Float64
Npart = 1000 #number of particles

using Random
Random.seed!(1114)

function get_spherical_cloud(Ngas, rmax)
    X = Vector{SVector{N,T}}(undef, Ngas)
    for i in 1:Ngas
        r3d = rmax * 2 #s.t. the while-loop will be evaluated at least once
        while r3d > rmax
            m_frac = rand() * 0.99;      #m_frac = 0 ~ 0.99
            r3d = rmax * m_frac;
        end
        phi       = ( 2. * rand() - 1. ) * pi
        cos_theta = ( 2. * rand() - 1. )
        x = r3d * sqrt( 1. - cos_theta^2 ) * cos(phi)
        y = r3d * sqrt( 1. - cos_theta^2 ) * sin(phi)
        z = r3d * cos_theta
        X[i] = @SVector [x,y,z]
    end
    return X
end

const BOXSIZE = 1.0
const hsml0 = 0.2


#randomly distributing particles
#X = [@SVector rand(N) for _ in 1:Npart]
X = get_spherical_cloud(Npart, 0.5*BOXSIZE)
topnode_length = @SVector(ones(N)) * BOXSIZE  #actual length of tree

center = @SVector [0.,0.,0.]

hsml = 0.5 .* ParticleGridMapper.norm.(X) .^ 0.5 #
#hsml = ones(Npart) .* hsml0 #
mass = ones(Npart) #particle mass

part = [DataP2G{N,T}(SVector(X[i]), i, hsml[i], mass[i], 0.0, T[]) for i in eachindex(X)]

#build the tree
tree = buildtree(part, center, topnode_length);
gridAMR = get_AMRgrid(tree)
num_nodes  = length(gridAMR[gridAMR.==1])
num_leaves = length(gridAMR[gridAMR.==0])
@show num_nodes, num_leaves

grid_volumes = get_AMRgrid_volumes(tree)
@show sum(grid_volumes) ≈ prod(topnode_length)


boxsizes = @SVector(ones(N)) * BOXSIZE  #for periodic B.C.

rho = ones(Npart)
volume = ones(Npart)

println("serial version...")
@time map_particle_to_AMRgrid!(tree, rho, volume, X, hsml, boxsizes)
rhoAMR = get_AMRfield(tree)
@show sum(grid_volumes.*rhoAMR)

println("parallel version...")
@time map_particle_to_AMRgrid_SPH_thread!(tree, rho, volume, X, hsml, boxsizes)
rhoAMR = get_AMRfield(tree)
@show sum(grid_volumes.*rhoAMR)

println("serial version, precomputed ngbs...")
#@time map_particle_to_AMRgrid_knownNgb!(tree, rho, volume, X, hsml, boxsizes)
@time map_particle_to_AMRgrid!(tree, rho, volume, X, hsml, boxsizes, knownNgb=true)
rhoAMR = get_AMRfield(tree)
@show sum(grid_volumes.*rhoAMR)

println("parallel version, precomputed ngbs...")
#@time map_particle_to_AMRgrid_knownNgb_thread!(tree, rho, volume, X, hsml, boxsizes)
@time map_particle_to_AMRgrid_SPH_thread!(tree, rho, volume, X, hsml, boxsizes, knownNgb=true)
rhoAMR = get_AMRfield(tree)
@show sum(grid_volumes.*rhoAMR)


nx = ny = 256
image = zeros(T, nx, ny)
@time image = project_AMRgrid_to_image(nx, ny, 1, 2, tree, center, boxsizes)
image2 = zeros(T, nx, ny);
@time image2 = project_AMRgrid_to_image_thread(nx, ny, 1, 2, tree, center, boxsizes);
@show all(image .≈ image2)


clf()
fig, ax = subplots(1, 2, figsize=(20, 10))
ax[1].plot(getindex.(X,1), getindex.(X,2), ".", ms=3, color="red")
ax[1].axis([-0.5*BOXSIZE, 0.5*BOXSIZE, -0.5*BOXSIZE, 0.5*BOXSIZE])
ax[2].imshow(log10.(image'), origin="lower", interpolation="none", extent=(-0.5*BOXSIZE, 0.5*BOXSIZE, -0.5*BOXSIZE, 0.5*BOXSIZE))

fig.tight_layout()
