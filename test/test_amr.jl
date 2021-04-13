using ParticleGridMapper
using StaticArrays
#using PyPlot
using Test

const N = 3 #spatial dimension
const T = Float64
const Npart = 1000 #number of particles

using Random
Random.seed!(1114) #happy birthday! <3

const BOXSIZE = 0.807 #use something != 1.0 to test round-off error

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


let
@testset "spherical cloud with AMR" begin

#randomly distributing particles
#X = [@SVector rand(N) for _ in 1:Npart]
X = get_spherical_cloud(Npart, 0.5*BOXSIZE)
topnode_length = @SVector(ones(N)) * BOXSIZE  #actual length of tree

center = @SVector zeros(N)

radius = ParticleGridMapper.norm.(X./BOXSIZE)
Mtot = 1.0
mass = ones(Npart) .* (Mtot / Npart) #particle mass
rho = 2*Mtot/(4*pi) .* radius.^(-2)

volume = mass ./ rho #only needed for SPH

#large enough s.t. there are no grids with 0 ngb particles and we can test all(image./BOXSIZE .≈  1.) (see below)
hsml = 0.7*BOXSIZE .* radius.^(2/3)

part = [DataP2G{N,T}(SVector(X[i]), i, hsml[i], mass[i], 0.0, T[]) for i in eachindex(X)]

#build the tree
tree = buildtree(part, center, topnode_length);
grid_volumes = get_AMRgrid_volumes(tree)
tree_max_depth = Int( log2( round( BOXSIZE / minimum(grid_volumes)^(1/3))) )
#println( "true maximum depth of tree = ", log2( BOXSIZE / minimum(grid_volumes)^(1/3) ) )

gridAMR = get_AMRgrid(tree)
num_nodes  = length(gridAMR[gridAMR.==1])
num_leaves = length(gridAMR[gridAMR.==0])
@test num_nodes + num_leaves == length(gridAMR) #gridAMR is either 0 or 1
@test Npart <= num_leaves <=  (2^N)*Npart

boxsizes = @SVector(ones(N)) * BOXSIZE  #for periodic B.C.

#map a const. field to AMR gird with NGP, SPH and MFM with different max_depth
for idepth in 1:tree_max_depth
    max_depth = idepth

    gridAMR = get_AMRgrid(tree, max_depth=max_depth)
    num_nodes  = length(gridAMR[gridAMR.==1])
    num_leaves = length(gridAMR[gridAMR.==0])
    @test num_nodes + num_leaves == length(gridAMR) #gridAMR is either 0 or 1

    grid_volumes = get_AMRgrid_volumes(tree, max_depth=max_depth)
    @test sum(grid_volumes) ≈ prod(topnode_length) #particle of unity

    map_particle_to_AMRgrid_NGP!(tree, ones(Npart), max_depth=max_depth)
    onesAMR_NGP_s = get_AMRfield(tree, max_depth=max_depth)
    map_particle_to_AMRgrid_NGP_thread!(tree, ones(Npart), max_depth=max_depth)
    onesAMR_NGP = get_AMRfield(tree, max_depth=max_depth)
    @test all(onesAMR_NGP .≈ onesAMR_NGP_s)

    map_particle_to_AMRgrid_SPH!(tree, ones(Npart), volume, X, hsml, boxsizes, knownNgb=false, max_depth=max_depth)
    onesAMR_SPH_s = get_AMRfield(tree, max_depth=max_depth)
    map_particle_to_AMRgrid_SPH_thread!(tree, ones(Npart), volume, X, hsml, boxsizes, knownNgb=false, max_depth=max_depth)
    onesAMR_SPH = get_AMRfield(tree, max_depth=max_depth)
    @test all(onesAMR_SPH .≈ onesAMR_SPH_s)

    map_particle_to_AMRgrid_MFM!(tree, ones(Npart), X, hsml, boxsizes, knownNgb=false, max_depth=max_depth)
    onesAMR_MFM_s = get_AMRfield(tree, max_depth=max_depth)
    map_particle_to_AMRgrid_MFM_thread!(tree, ones(Npart), X, hsml, boxsizes, knownNgb=false, max_depth=max_depth)
    onesAMR_MFM = get_AMRfield(tree, max_depth=max_depth)
    @test all(onesAMR_MFM .≈ onesAMR_MFM_s)
    @test all(onesAMR_MFM .≈ 1.) #const. field should give all ones for MFM as it is indep. of volume estimates (unless there are grids with no ngb particles)

    #this has to be right after map_particle_to_AMRgrid_MFM_thread! s.t. all(image./BOXSIZE .≈  1.)
    for p in 1:10
        nx = ny = 2^p
        image = zeros(T, nx, ny)
        image = project_AMRgrid_to_image(nx, ny, 1, 2, tree, center, boxsizes, max_depth=max_depth)
        image2 = zeros(T, nx, ny);
        image2 = project_AMRgrid_to_image_thread(nx, ny, 1, 2, tree, center, boxsizes, max_depth=max_depth);
        @test all(image .≈ image2) #parallel == serial
        @test all(image./BOXSIZE .≈  1.) #projection working properly
    end

end #idepth
end #@testset
end #let
