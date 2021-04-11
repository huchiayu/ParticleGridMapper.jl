using ParticleGridMapper
using StaticArrays
using PyPlot

const N = 3 #spatial dimension
const T = Float64
const Npart = 1000 #number of particles
const BOXSIZE = 1.0

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



X = get_spherical_cloud(Npart, 0.5*BOXSIZE)
topnode_length = @SVector(ones(N)) * BOXSIZE  #actual length of tree

center = @SVector [0.,0.,0.]

Mtot = 1.0
mass = ones(Npart) .* (Mtot / Npart) #particle mass

radius = ParticleGridMapper.norm.(X./BOXSIZE)
rho = 2*Mtot/(4*pi) .* radius.^(-2)
volume = mass ./ rho

Nngb = 32
hsml = ( ( Nngb * (Mtot / Npart) / (4*pi/3) ) ./ rho).^(1/3)

part = [DataP2G{N,T}(SVector(X[i]), i, hsml[i], mass[i], 0.0, T[]) for i in eachindex(X)]

max_depth = 10

#build the tree
tree = buildtree(part, center, topnode_length);

boxsizes = @SVector(ones(N)) * BOXSIZE  #for periodic B.C.

nx = ny = 256

#println("MFM...")
map_particle_to_AMRgrid_MFM_thread!(tree, rho, X, hsml, boxsizes, knownNgb=false, max_depth=max_depth)
rhoAMR_MFM = get_AMRfield(tree, max_depth=max_depth)
image_MFM = project_AMRgrid_to_image_thread(nx, ny, 1, 2, tree, center, boxsizes, max_depth=max_depth);

#println("SPH...")
map_particle_to_AMRgrid_SPH_thread!(tree, rho, volume, X, hsml, boxsizes, knownNgb=true, max_depth=max_depth)
rhoAMR_SPH = get_AMRfield(tree, max_depth=max_depth)
image_SPH = project_AMRgrid_to_image_thread(nx, ny, 1, 2, tree, center, boxsizes, max_depth=max_depth);

#println("NGP...")
map_particle_to_AMRgrid_NGP_thread!(tree, rho, max_depth=max_depth)
rhoAMR_NGP = get_AMRfield(tree, max_depth=max_depth)
image_NGP = project_AMRgrid_to_image_thread(nx, ny, 1, 2, tree, center, boxsizes, max_depth=max_depth);

clf()
fig, ax = subplots(2, 2, figsize=(10, 10))
ax[1].set_title("particle distribution")
ax[1].plot(getindex.(X,1), getindex.(X,2), ".", ms=3, color="red")
ax[1].axis([-0.5*BOXSIZE, 0.5*BOXSIZE, -0.5*BOXSIZE, 0.5*BOXSIZE])
ax[2].set_title("MFM")
ax[2].imshow(log10.(image_MFM'), origin="lower", vmin=-1, vmax=1.5, interpolation="none", extent=(-0.5*BOXSIZE, 0.5*BOXSIZE, -0.5*BOXSIZE, 0.5*BOXSIZE))
ax[3].set_title("SPH")
ax[3].imshow(log10.(image_SPH'), origin="lower", vmin=-1, vmax=1.5, interpolation="none", extent=(-0.5*BOXSIZE, 0.5*BOXSIZE, -0.5*BOXSIZE, 0.5*BOXSIZE))
ax[4].set_title("NGP")
ax[4].imshow(log10.(image_NGP'), origin="lower", vmin=-1, vmax=1.5, interpolation="none", extent=(-0.5*BOXSIZE, 0.5*BOXSIZE, -0.5*BOXSIZE, 0.5*BOXSIZE))

fig.tight_layout()
savefig("compare_MFM_SPH_NGP_cloud.png")
