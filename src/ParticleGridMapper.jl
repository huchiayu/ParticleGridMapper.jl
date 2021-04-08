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
buildtree, DataP2G, get_AMRgrid_volumes, get_AMRgrid, get_AMRfield,
map_particle_to_AMRgrid!,
map_particle_to_AMRgrid_thread!,
project_AMRgrid_to_image,
project_AMRgrid_to_image_thread

import OctreeBH: AbstractData, assign_additional_node_data!

include("kernel.jl")
include("cartesian_grid.jl")
include("mapAMRgrid.jl")

end
