module ParticleGridMapper
using StaticArrays
using OctreeBH
#using LinearAlgebra  #norm()
using .Threads

export map_particle_to_2Dgrid_loopP,
map_particle_to_2Dgrid_loopP_thread,
map_particle_to_3Dgrid_loopP_thread,
map_particle_to_2Dgrid_loopP_noCar,
map_particle_to_3Dgrid_NGP,
map_particle_to_3Dgrid_NGP_thread,
buildtree, DataP2G,
get_AMRgrid_volumes, get_AMRgrid, get_AMRfield, set_AMRfield,
map_particle_to_AMRgrid_SPH!,
map_particle_to_AMRgrid_MFM!,
map_particle_to_AMRgrid_NGP!,
map_particle_to_AMRgrid_loopP_recursive!,
project_AMRgrid_to_image,
get_max_tree_depth,
set_max_depth_AMR!,
balance_all_level!

import OctreeBH: AbstractData, assign_additional_node_data!

include("kernel.jl")
include("cartesian_grid.jl")
include("mapAMRgrid.jl")

end
