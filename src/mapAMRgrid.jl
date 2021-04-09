mutable struct DataP2G{N,T<:Real} <: AbstractData{N,T}
	pos::SVector{N,T}
	idx::Int64
	hsml::T
    mass::T
    field::T
	idx_ngbs::Vector{Int64} #ngb list for leaf node
end

DataP2G{N,T}() where {N,T} = DataP2G{N,T}(zero(SVector{N,T}),0,0,0,0,[])

function assign_additional_node_data!(n::DataP2G, old::DataP2G, new::DataP2G)
end

#driver function
function map_particle_to_AMRgrid!(tree::Node{N,T,D}, field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
	map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, tree, boxsizes, knownNgb=knownNgb)
end

function map_particle_to_AMRgrid!(tree::Node{N,T,D}, field::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
	volume = ones(T,length(field))
	map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, tree, boxsizes, knownNgb=knownNgb)
end

function kernel_average_SPH(field::Vector{T}, X0::SVector{N,T}, idx_ngbs::Vector{Int64}, X::Vector{SVector{N,T}},
							hsml::Vector{T}, volume::Vector{T}, boxsizes::SVector{N,T}) where {N,T}
	res = zero(T)
	for k in eachindex(idx_ngbs)
		j = idx_ngbs[k]
		dx = nearest.(X[j] - X0, boxsizes)
		dr = norm(dx)
		Wij = kernel_cubic(dr/hsml[j]) / hsml[j]^N
		res += field[j] * (volume[j] * Wij)
		#res += 1.0  #debug
	end
	return res
end

function kernel_average_MFM(field::Vector{T}, X0::SVector{N,T}, idx_ngbs::Vector{Int64}, X::Vector{SVector{N,T}},
							hsml::Vector{T}, boxsizes::SVector{N,T}) where {N,T}
	res = zero(T)
	sigma = zero(T)
	for k in eachindex(idx_ngbs)
		j = idx_ngbs[k]
		dx = nearest.(X[j] - X0, boxsizes)
		dr = norm(dx)
		Wij = kernel_cubic(dr/hsml[j]) / hsml[j]^N
		res += field[j] * Wij
		sigma += Wij
	end
	return length(idx_ngbs) > 0 ? (res / sigma) : zero(T)
end


#recursively walk the tree
function map_particle_to_AMRgrid_recursive!(tree::Node{N,T,D}, field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	node::Node{N,T,D}, boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
    if isLeaf(node)
		#calculate field value regardless of there is a particle or not
		if hsml==T[]
			if node.n == nothing
				node.n = D()
			end
			if node.p !== nothing
				node.n.field = field[node.p.idx]
			else
				node.n.field = zero(T)
			end
			return
		end

		if knownNgb == true
			#already have a ngb list so we don't have to search for ngbs again
			node.n.field = zero(T)
		else
			if node.n == nothing
				node.n = D()
			end
        	node.n.idx_ngbs = get_scatter_ngb_tree(node.center, tree, boxsizes)
		end

		if volume == T[]
			node.n.field = kernel_average_MFM(field, node.center, node.n.idx_ngbs, X, hsml, boxsizes)
		else
			node.n.field = kernel_average_SPH(field, node.center, node.n.idx_ngbs, X, hsml, volume, boxsizes)
		end
    else
        #always open the node until we find a leaf
	    @inbounds for i in 1:2^N
            map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, node.child[i], boxsizes, knownNgb=knownNgb)
	    end
    end
end


#function map_particle_to_AMRgrid_thread_2nd!(tree::Node{N,T,D}, field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T}, node::Node{N,T,D}, boxsizes::SVector{N,T}) where {N,T,D}
#	@sync for i in 1:2^N, j in 1:2^N
#		Threads.@spawn map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, node.child[i].child[j], boxsizes)
#	end
#end

#1-layer unrolled
function map_particle_to_AMRgrid_thread!(tree::Node{N,T,D}, field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
	@sync for i in 1:2^N
		Threads.@spawn map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, tree.child[i], boxsizes, knownNgb=knownNgb)
	end
end

function map_particle_to_AMRgrid_SPH_thread!(tree, field, volume, X, hsml, boxsizes; knownNgb=false)
	map_particle_to_AMRgrid_thread!(tree, field, volume, X, hsml, boxsizes, knownNgb=knownNgb)
end
function map_particle_to_AMRgrid_MFM_thread!(tree, field, X, hsml, boxsizes; knownNgb=false)
	volume = eltype(field)[]
	map_particle_to_AMRgrid_thread!(tree, field, volume, X, hsml, boxsizes, knownNgb=knownNgb)
end
function map_particle_to_AMRgrid_NGP_thread!(tree::Node{N,T,D}, field) where {N,T,D}
	map_particle_to_AMRgrid_thread!(tree, field, T[], SVector{N,T}[], T[], SVector{N}(zeros(T,N)))
end

#=
function map_particle_to_AMRgrid_thread!(tree::Node{N,T,D}, field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	node::Node{N,T,D}, boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
	for i in 1:2^N
	#@sync for i in 1:2^N
		if isLeaf(node.child[i])
			#@show "it's a leaf!", i
			map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, node.child[i], boxsizes, knownNgb=knownNgb)
		else
			for j in 1:2^N
				if isLeaf(node.child[i].child[j])
					#@show "it's a leaf!", i,j
					map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, node.child[i].child[j], boxsizes, knownNgb=knownNgb)
				else
					@threads for k in 1:2^N
						#@show i,j,k
						#println("open this node")
						#Threads.@spawn map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, node.child[i].child[j].child[k], boxsizes, knownNgb=knownNgb)
						map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, node.child[i].child[j].child[k], boxsizes, knownNgb=knownNgb)
					end
				end
			end
		end
	end
end
=#

function get_AMRgrid(Tint::DataType, tree::Node{N,T,D}) where {N,T,D}
	gridAMR = Tint[]
	get_AMRgrid_recursive!(gridAMR, tree)
	return gridAMR
end

#default type is Int8 (to be compatible with RADMC3D)
function get_AMRgrid(tree::Node{N,T,D}) where {N,T,D}
	get_AMRgrid(Int8, tree)
end

function get_AMRgrid_recursive!(gridAMR::Vector{Tint}, node::Node{N,T,D}) where {N,T,D, Tint<:Integer}
    if isLeaf(node)
        #println("in a leaf node")
		push!(gridAMR, 0)  #0 = leaf
    else
        #println("This is a node... ")
		push!(gridAMR, 1)  #0 = leaf
        #always open the node until we find a leaf
	    @inbounds for i in 1:2^N
	        #println("open this node")
			get_AMRgrid_recursive!(gridAMR, node.child[i])
	    end
    end
end

function get_AMRfield(tree::Node{N,T,D}) where {N,T,D}
	fieldAMR = T[]
	get_AMRfield_recursive!(fieldAMR, tree)
	return fieldAMR
end

function get_AMRfield_recursive!(fieldAMR::Vector{T}, node::Node{N,T,D}) where {N,T,D, Tint<:Integer}
    if isLeaf(node)
		push!(fieldAMR, node.n.field)
    else
	    @inbounds for i in 1:2^N
			get_AMRfield_recursive!(fieldAMR, node.child[i])
	    end
    end
end

function get_AMRgrid_volumes(tree::Node{N,T,D}) where {N,T,D}
	grid_volumes = T[]
	get_AMRgrid_volumes_recursive!(grid_volumes,tree)
	return grid_volumes
end

function get_AMRgrid_volumes_recursive!(volumearray, node::Node{N,T,D}) where {N,T,D}
    if isLeaf(node)
		push!(volumearray, prod(node.length))
    else
	    @inbounds for i in 1:2^N
			get_AMRgrid_volumes_recursive!(volumearray, node.child[i])
	    end
    end
end

function project_AMRgrid_to_image(nx, ny, dimx, dimy, tree::Node{N,T,D}, toptreecenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
	image = zeros(T, nx, ny)
	project_AMRgrid_to_image_recursive!(image, nx, ny, dimx, dimy, tree, toptreecenter, boxsizes);
	return image
end

function project_AMRgrid_to_image_recursive!(image, nx, ny, dimx, dimy, node::Node{N,T,D}, toptreecenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
	@assert ispow2(nx) && ispow2(ny)
    if isLeaf(node)
		#the lower left corner of the tree
		xmin = toptreecenter[dimx] - 0.5 * boxsizes[dimx]
		ymin = toptreecenter[dimy] - 0.5 * boxsizes[dimy]
		x1 = node.center[dimx] - xmin - 0.5 * node.length[dimx]
		x2 = node.center[dimx] - xmin + 0.5 * node.length[dimx]
		y1 = node.center[dimy] - ymin - 0.5 * node.length[dimy]
		y2 = node.center[dimy] - ymin + 0.5 * node.length[dimy]
		#fraction of area when pixel is larger than node
		f_area = (node.length[dimx]*node.length[dimy]) / (boxsizes[dimx]*boxsizes[dimy]/(nx*ny))
		fac = min(f_area, 1.0)
		if N==3
			dimlos = filter!(x->(x!=dimx)&&(x!=dimy), [1,2,3])[1]
			fac *= node.length[dimlos]
		end
		dx = boxsizes[dimx] / nx
		dy = boxsizes[dimy] / ny
		ix1 = Int64(fld(x1 , dx) + 1)  #+1 becuase julia is one-based
		ix2 = Int64(fld(x2 , dx) + 1)
		iy1 = Int64(fld(y1 , dy) + 1)
		iy2 = Int64(fld(y2 , dy) + 1)
		if x2%dx==0 ix2 -= 1 end
		if y2%dy==0 iy2 -= 1 end
		@assert ix2>=ix1 && iy2>=iy1
		if ix1>ix2 ix2=ix1 end #happens when node length < pixel size
		if iy1>iy2 iy2=iy1 end
		#image[ix1:ix2,iy1:iy2] .+= fac #debug
		#image[ix1:ix2,iy1:iy2] .+= node.n.field * fac
		#avoid using sub-array to reduce allocation
		for j in iy1:iy2, i in ix1:ix2
			image[i,j] += node.n.field * fac
		end
    else
	    @inbounds for i in 1:2^N
            project_AMRgrid_to_image_recursive!(image, nx, ny, dimx, dimy, node.child[i], toptreecenter, boxsizes)
	    end
    end
end

function project_AMRgrid_to_image_thread(nx, ny, dimx, dimy, node::Node{N,T,D}, toptreecenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
	image = zeros(T, nx, ny)
	image_thread = [zeros(T, nx, ny) for i in 1:nthreads()]; #each thread has its own image
	@sync for i in 1:2^N
		Threads.@spawn project_AMRgrid_to_image_recursive!(image_thread[threadid()], nx, ny, dimx, dimy, node.child[i], toptreecenter, boxsizes)
	end
	image .= sum(image_thread)
	return image
end
