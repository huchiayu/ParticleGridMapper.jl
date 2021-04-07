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


function map_particle_to_AMRgrid!(field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	node::Node{N,T,D}, tree::Node{N,T,D}, boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
    if isLeaf(node)
        #println("in a leaf node")
		#calculate field value regardless of there is a particle or not
		if knownNgb == true
			node.n.field = 0.
		else
			node.n = D()
			#@show node.n
        	node.n.idx_ngbs = get_scatter_ngb_tree(node.center, tree, boxsizes)
		end

        #@show idx_ngbs
		for k in eachindex(node.n.idx_ngbs)
			j = node.n.idx_ngbs[k]
			dx = nearest.(X[j] - node.center, boxsizes)
			dr = norm(dx)
			Wij = kernel_cubic(dr/hsml[j]) / hsml[j]^N
			#if i == 1 @show typeof.((mass[j] , rho[j] , Wij, dr, dx)) end
			node.n.field += field[j] * (volume[j] * Wij)
			#node.n.field += 1.0
			#aaa = @SMatrix zeros(T,N,N)
			#aaa += 1.0
		end
		#@show node.n
    else
        #println("This is a node... ")
        #always open the node until we find a leaf
	    @inbounds for i in 1:2^N
	        #println("open this node")
            map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i], tree, boxsizes, knownNgb=knownNgb)
	    end
    end
end


#function map_particle_to_AMRgrid_thread_1st!(field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T}, node::Node{N,T,D}, tree::Node{N,T,D}, boxsizes::SVector{N,T}) where {N,T,D}
#	@sync for i in 1:2^N
#		#println("open this node")
#		Threads.@spawn map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i], tree, boxsizes)
#	end
#end
#function map_particle_to_AMRgrid_thread_2nd!(field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T}, node::Node{N,T,D}, tree::Node{N,T,D}, boxsizes::SVector{N,T}) where {N,T,D}
#	@sync for i in 1:2^N, j in 1:2^N
#		#println("open this node")
#		Threads.@spawn map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i].child[j], tree, boxsizes)
#	end
#end

#1-layer unrolled
function map_particle_to_AMRgrid_thread!(field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	node::Node{N,T,D}, tree::Node{N,T,D}, boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
	@sync for i in 1:2^N
		#println("open this node")
		Threads.@spawn map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i], tree, boxsizes, knownNgb=knownNgb)
	end
end

#=
function map_particle_to_AMRgrid_thread!(field::Vector{T}, volume::Vector{T}, X::Vector{SVector{N,T}}, hsml::Vector{T},
	node::Node{N,T,D}, tree::Node{N,T,D}, boxsizes::SVector{N,T}; knownNgb::Bool=false) where {N,T,D}
	for i in 1:2^N
	#@sync for i in 1:2^N
		if isLeaf(node.child[i])
			#@show "it's a leaf!", i
			map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i], tree, boxsizes, knownNgb=knownNgb)
		else
			for j in 1:2^N
				if isLeaf(node.child[i].child[j])
					#@show "it's a leaf!", i,j
					map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i].child[j], tree, boxsizes, knownNgb=knownNgb)
				else
					@threads for k in 1:2^N
						#@show i,j,k
						#println("open this node")
						#Threads.@spawn map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i].child[j].child[k], tree, boxsizes, knownNgb=knownNgb)
						map_particle_to_AMRgrid!(field, volume, X, hsml, node.child[i].child[j].child[k], tree, boxsizes, knownNgb=knownNgb)
					end
				end
			end
		end
	end
end
=#

function get_AMRgrid!(gridAMR::Vector{Tint}, node::Node{N,T,D}) where {N,T,D, Tint<:Integer}
    if isLeaf(node)
        #println("in a leaf node")
		push!(gridAMR, 0)  #0 = leaf
    else
        #println("This is a node... ")
		push!(gridAMR, 1)  #0 = leaf
        #always open the node until we find a leaf
	    @inbounds for i in 1:2^N
	        #println("open this node")
			get_AMRgrid!(gridAMR, node.child[i])
	    end
    end
end

function get_AMRfield!(fieldAMR::Vector{T}, node::Node{N,T,D}) where {N,T,D, Tint<:Integer}
    if isLeaf(node)
        #println("in a leaf node")
		push!(fieldAMR, node.n.field)
    else
        #println("This is a node... ")
        #always open the node until we find a leaf
	    @inbounds for i in 1:2^N
	        #println("open this node")
			get_AMRfield!(fieldAMR, node.child[i])
	    end
    end
end


function get_AMRgrid_volumes!(volumearray, node::Node{N,T,D}) where {N,T,D}
    if isLeaf(node)
        #println("in a leaf node")
		#@show node.n
		push!(volumearray, prod(node.length))
    else
        #println("This is a node... ")
        #always open the node until we find a leaf
	    @inbounds for i in 1:2^N
	        #println("open this node")
			get_AMRgrid_volumes!(volumearray, node.child[i])
            #loop_over_all_leaves!(node.child[i])
	    end
    end
end

function project_AMRgrid_to_image!(image, nx, ny, dimx, dimy, node::Node{N,T,D}, toptreecenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
	@assert ispow2(nx) && ispow2(ny)
    if isLeaf(node)
        #println("in a leaf node")
		#@show node.n
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
		#image[ix1:ix2,iy1:iy2] .+= node.n.field * fac
		#image[ix1:ix2,iy1:iy2] .+= fac #debug
		#avoid using sub-array to reduce allocation
		for j in iy1:iy2, i in ix1:ix2
			image[i,j] += node.n.field * fac
		end
    else
        #println("This is a node... ")
        #always open the node until we find a leaf
	    @inbounds for i in 1:2^N
	        #println("open this node")
            project_AMRgrid_to_image!(image, nx, ny, dimx, dimy, node.child[i], toptreecenter, boxsizes)
	    end
    end
end

function project_AMRgrid_to_image_thread!(image, nx, ny, dimx, dimy, node::Node{N,T,D}, toptreecenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
	#image_thread = zeros(T, nx, ny, nthreads()); #each thread has its own image
	image_thread = [zeros(T, nx, ny) for i in 1:nthreads()]; #each thread has its own image
	@sync for i in 1:2^N
		#println("open this node")
		Threads.@spawn project_AMRgrid_to_image!(image_thread[threadid()], nx, ny, dimx, dimy, node.child[i], toptreecenter, boxsizes)
	end
	image .= sum(image_thread)
end

#=
#3-layer unrolled
function project_AMRgrid_to_image_thread!(image, nx, ny, dimx, dimy, node::Node{N,T,D}, toptreecenter::SVector{N,T}, boxsizes::SVector{N,T}) where {N,T,D}
	#image_thread = zeros(T, nx, ny, nthreads()); #each thread has its own image
	image_thread = [zeros(T, nx, ny) for i in 1:nthreads()]; #each thread has its own image

	for i in 1:2^N
		if isLeaf(node.child[i])
			#@show "it's a leaf!", i
			project_AMRgrid_to_image!(image_thread[1], nx, ny, dimx, dimy, node.child[i], toptreecenter, boxsizes)
		else
			for j in 1:2^N
				if isLeaf(node.child[i].child[j])
					#@show "it's a leaf!", i,j
					project_AMRgrid_to_image!(image_thread[1], nx, ny, dimx, dimy, node.child[i].child[j], toptreecenter, boxsizes)
				else
					@threads for k in 1:2^N
						#@show i,j,k
						#println("open this node")
						project_AMRgrid_to_image!(image_thread[threadid()], nx, ny, dimx, dimy, node.child[i].child[j].child[k], toptreecenter, boxsizes)
					end
				end
			end
		end
	end
 	#image = dropdims(sum(image_thread, dims=3),dims=3) #sum over threads
	image .= sum(image_thread)
end
=#
