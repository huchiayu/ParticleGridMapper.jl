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

const MAX_DEPTH = Int(1000)

#return AMR structure (0 = leaf, 1=non-leaf node)
function get_AMRgrid(Tint::DataType, tree::Node{N,T,D}; max_depth::Int=MAX_DEPTH) where {N,T,D}
	gridAMR = Tint[]
	root_node_length = tree.length[1]
	get_AMRgrid_recursive!(gridAMR, tree, max_depth, root_node_length)
	return gridAMR
end

#default type is Int8 (to be compatible with RADMC3D)
function get_AMRgrid(tree::Node{N,T,D}; max_depth::Int=MAX_DEPTH) where {N,T,D}
	get_AMRgrid(Int8, tree, max_depth=max_depth)
end

#root_node_length is only used to figure out the node's depth
function get_AMRgrid_recursive!(gridAMR::Vector{Tint}, node::Node{N,T,D}, max_depth::Int, root_node_length::T) where {N,T,D, Tint<:Integer}
    if isLeaf(node)
        #println("in a leaf node")
		push!(gridAMR, 0)  #0 = leaf
    else
        #println("This is a node... ")
        #always open the node until we find a leaf
		depth = log2(round(root_node_length / node.length[1]))
		if depth < max_depth
			push!(gridAMR, 1)  #0 = leaf
	    	@inbounds for i in 1:2^N
	        	#println("open this node")
				get_AMRgrid_recursive!(gridAMR, node.child[i], max_depth, root_node_length)
	    	end
		else
			#don't open this node
			push!(gridAMR, 0)
		end
    end
end

#return grid/node volume following the order of the tree
function get_AMRgrid_volumes(tree::Node{N,T,D}; max_depth::Int=MAX_DEPTH) where {N,T,D}
	grid_volumes = T[]
	root_node_length = tree.length[1]
	get_AMRgrid_volumes_recursive!(grid_volumes, tree, max_depth, root_node_length)
	return grid_volumes
end

function get_AMRgrid_volumes_recursive!(volumearray, node::Node{N,T,D}, max_depth, root_node_length::T) where {N,T,D}
    if isLeaf(node)
		push!(volumearray, prod(node.length))
    else
		depth = log2(round(root_node_length / node.length[1]))
		if depth < max_depth
	    	@inbounds for i in 1:2^N
				get_AMRgrid_volumes_recursive!(volumearray, node.child[i], max_depth, root_node_length)
	    	end
		else
			push!(volumearray, prod(node.length))
		end
    end
end


#main function to map the particle information to an AMR grid


function map_particle_to_AMRgrid!(treeAMR::Node{N,T,D}, field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	boxsizes::SVector{N,T}, treeNgb::Union{Node{N,T,D}, Nothing}, serial::Bool) where {N,T,D}
	if serial
		map_particle_to_AMRgrid_recursive!(treeAMR, field, volume, X, hsml, boxsizes, treeNgb)
	else
		#4-layer unrolled (4^4 tasks)
		if isLeaf(treeAMR)
			map_particle_to_AMRgrid_recursive!(treeAMR, field, volume, X, hsml, boxsizes, treeNgb)
			return
		end
		@sync for i in 1:2^N
			if isLeaf(treeAMR.child[i])
				map_particle_to_AMRgrid_recursive!(treeAMR.child[i], field, volume, X, hsml, boxsizes, treeNgb)
			else
				for j in 1:2^N
					if isLeaf(treeAMR.child[i].child[j])
						map_particle_to_AMRgrid_recursive!(treeAMR.child[i].child[j], field, volume, X, hsml, boxsizes, treeNgb)
					else
						for k in 1:2^N
							if isLeaf(treeAMR.child[i].child[j].child[k])
								map_particle_to_AMRgrid_recursive!(treeAMR.child[i].child[j].child[k], field, volume, X, hsml, boxsizes, treeNgb)
							else
								for l in 1:2^N
									#@show i,j,k
									Threads.@spawn map_particle_to_AMRgrid_recursive!(treeAMR.child[i].child[j].child[k].child[l], field, volume, X, hsml, boxsizes, treeNgb)
								end
							end
						end
					end
				end
			end
		end

		#parallel version (1-layer unrolled)
		#@sync for i in 1:2^N
		#	Threads.@spawn map_particle_to_AMRgrid_recursive!(treeAMR.child[i], field, volume, X, hsml, boxsizes, treeNgb)
		#end
	end
end

function map_particle_to_AMRgrid_SPH!(treeAMR, field, volume, X, hsml, boxsizes;
	treeNgb::Union{Node{N,T,D}, Nothing}=nothing, serial::Bool=false) where {N,T,D}
	map_particle_to_AMRgrid!(treeAMR, field, volume, X, hsml, boxsizes, treeNgb, serial)
end
function map_particle_to_AMRgrid_MFM!(treeAMR, field, X, hsml, boxsizes;
	treeNgb::Union{Node{N,T,D}, Nothing}=nothing, serial::Bool=false) where {N,T,D}
	volume = eltype(field)[]
	map_particle_to_AMRgrid!(treeAMR, field, volume, X, hsml, boxsizes, treeNgb, serial)
end
function map_particle_to_AMRgrid_NGP!(treeAMR::Node{N,T,D}, field; serial::Bool=false) where {N,T,D}
	map_particle_to_AMRgrid!(treeAMR, field, T[], SVector{N,T}[], T[], SVector{N}(zeros(T,N)), nothing, serial)
end


#function map_particle_to_AMRgrid_thread_2nd!(tree::Node{N,T,D}, field::Vector{T}, volume::Vector{T},
#X::Vector{SVector{N,T}}, hsml::Vector{T}, node::Node{N,T,D}, boxsizes::SVector{N,T}) where {N,T,D}
#	@sync for i in 1:2^N, j in 1:2^N
#		Threads.@spawn map_particle_to_AMRgrid_recursive!(tree, field, volume, X, hsml, node.child[i].child[j], boxsizes)
#	end
#end

#recursively walk the tree
function map_particle_to_AMRgrid_loopP_recursive!(node::Node{N,T,D}, f_j::T, V_j::T, h_j::T, X_j::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}

	dx = nearest.(X_j - node.center, boxsizes)

	if !isLeaf(node)
		#@show "node"
		if all(abs.(dx) .< 0.5*node.length .+ h_j) #skip the node if no overlap
			@inbounds for i in 1:2^N
	        	map_particle_to_AMRgrid_loopP_recursive!(node.child[i], f_j, V_j, h_j, X_j, boxsizes)
			end
		end
	else
		#node can be used (either a leaf or a deep enough node)
		#println("leaf")
		dr = norm(dx)
		h_inv = 1 / h_j
		if dr < h_j
			Wij = kernel_cubic(dr*h_inv) * h_inv^N
			if node.n == nothing
				node.n = D()
			end
			node.n.field += f_j * (V_j * Wij)
		end
    end
end


#recursively walk the tree
function map_particle_to_AMRgrid_recursive!(node::Node{N,T,D}, field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	boxsizes::SVector{N,T}, treeNgb::Union{Node{N,T,D}, Nothing} ) where {N,T,D}

	if !isLeaf(node)
		@inbounds for i in 1:2^N
	        map_particle_to_AMRgrid_recursive!(node.child[i], field, volume, X, hsml, boxsizes, treeNgb)
		end
	else
	#if isLeaf(node)
		if hsml==T[] #NGP
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

		if treeNgb == nothing
			#already have a ngb list so we don't have to search for ngbs again
			node.n.field = zero(T)
		else
			if node.n == nothing
				node.n = D()
			end
        	node.n.idx_ngbs = get_scatter_ngb_tree(node.center, treeNgb, boxsizes)
		end

		if volume == T[]
			node.n.field = kernel_average_MFM(field, node.center, node.n.idx_ngbs, X, hsml, boxsizes)
		else
			node.n.field = kernel_average_SPH(field, node.center, node.n.idx_ngbs, X, hsml, volume, boxsizes)
		end
    end
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


########## return the AMR field stored in the tree as an array (following the order of the tree)
function get_AMRfield(tree::Node{N,T,D}; max_depth::Int=MAX_DEPTH) where {N,T,D}
	fieldAMR = T[]
	root_node_length = tree.length[1]
	get_AMRfield_recursive!(fieldAMR, tree, max_depth, root_node_length)
	return fieldAMR
end

function get_AMRfield_recursive!(fieldAMR::Vector{T}, node::Node{N,T,D}, max_depth::Int, root_node_length::T) where {N,T,D, Tint<:Integer}
    if isLeaf(node)
		if node.n == nothing
			node.n = D()
		end
		push!(fieldAMR, node.n.field)
    else
		depth = log2(round(root_node_length / node.length[1]))
		if depth < max_depth
	    	@inbounds for i in 1:2^N
				get_AMRfield_recursive!(fieldAMR, node.child[i], max_depth, root_node_length)
	    	end
		else
			push!(fieldAMR, node.n.field)
		end
    end
end

########## project the 3D AMR field to a 2D image
#function project_AMRgrid_to_image(nx, ny, dimx, dimy, tree::Node{N,T,D}, rootcenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
#	image = zeros(T, nx, ny)
#	project_AMRgrid_to_image_recursive!(image, nx, ny, dimx, dimy, tree, rootcenter, boxsizes);
#	return image
#end

function project_AMRgrid_to_image(nx, ny, dimx, dimy, tree::Node{N,T,D}, rootcenter::SVector{N,T}, boxsizes::SVector{N,T}; serial::Bool=false) where {N,T,D}
	image = zeros(T, nx, ny)
	if serial
		project_AMRgrid_to_image_recursive!(image, nx, ny, dimx, dimy, tree, rootcenter, boxsizes);
		return image
	end

	image_thread = [zeros(T, nx, ny) for i in 1:nthreads()]; #each thread has its own image
	if isLeaf(tree)
		project_AMRgrid_to_image_recursive!(image_thread[threadid()], nx, ny, dimx, dimy, tree, rootcenter, boxsizes)
	else
		@sync for i in 1:2^N
			if isLeaf(tree.child[i])
				project_AMRgrid_to_image_recursive!(image_thread[threadid()], nx, ny, dimx, dimy, tree.child[i], rootcenter, boxsizes)
			else
				for j in 1:2^N
					if isLeaf(tree.child[i].child[j])
						project_AMRgrid_to_image_recursive!(image_thread[threadid()], nx, ny, dimx, dimy, tree.child[i].child[j], rootcenter, boxsizes)
					else
						for k in 1:2^N
							Threads.@spawn project_AMRgrid_to_image_recursive!(image_thread[threadid()], nx, ny, dimx, dimy, tree.child[i].child[j].child[k], rootcenter, boxsizes)
						end
					end
				end
			end
		end
	end
	image .= sum(image_thread)
	return image
end

function project_AMRgrid_to_image_recursive!(image, nx, ny, dimx, dimy, node::Node{N,T,D}, rootcenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
	@assert ispow2(nx) && ispow2(ny)

    if !isLeaf(node)
		@inbounds for i in 1:2^N
			project_AMRgrid_to_image_recursive!(image, nx, ny, dimx, dimy, node.child[i], rootcenter, boxsizes)
		end
	else
		#the lower left corner of the tree
		xmin = rootcenter[dimx] - 0.5 * boxsizes[dimx]
		ymin = rootcenter[dimy] - 0.5 * boxsizes[dimy]
		#tree cell boundaries
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
		dtol = 10
		#get rid of the round-off error with round()
		ix1r = round(x1 / dx, digits=dtol)
		ix2r = round(x2 / dx, digits=dtol)
		iy1r = round(y1 / dy, digits=dtol)
		iy2r = round(y2 / dy, digits=dtol)
		#convert into integer index
		ix1 = Int64(floor(ix1r) + 1)  #+1 becuase julia is one-based
		ix2 = Int64(floor(ix2r) + 1)
		iy1 = Int64(floor(iy1r) + 1)
		iy2 = Int64(floor(iy2r) + 1)
		tol = 10.0^(-dtol)
		@assert ix2 >= ix2r && iy2 >= iy2r
		if isinteger(ix2r) ix2 -= 1 end
		if isinteger(iy2r) iy2 -= 1 end
		if ix1>ix2 ix2=ix1 end #happens when node length < pixel size
		if iy1>iy2 iy2=iy1 end
		@assert ix2>=ix1 && iy2>=iy1


		if ix2 > nx || ix1 < 1
			@show ix1, ix2, ix1r, ix2r, x1, x2, dx, boxsizes[dimx], depth
		end
		if iy2 > ny || iy1 < 1
			@show iy1, iy2, y1, y2, dy, boxsizes[dimy], depth
		end

		#image[ix1:ix2,iy1:iy2] .+= node.n.field * fac
		#avoid using sub-array to reduce allocation
		for j in iy1:iy2, i in ix1:ix2
			image[i,j] += node.n.field * fac
		end
    end
end
