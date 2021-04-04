module ParticleGridMapper
using StaticArrays
using LinearAlgebra  #norm()
using .Threads

export map_particle_to_2Dgrid_loopP,
map_particle_to_2Dgrid_loopP_thread,
map_particle_to_3Dgrid_loopP_thread,
map_particle_to_2Dgrid_loopP_noCar,
map_particle_to_3Dgrid_NGP,
map_particle_to_3Dgrid_NGP_thread

const GEO_FAC = 4.0 / 3.0 * pi
const KERNELCONST = 16.0 / pi
@inline ramp(x) = max(0, x);
@inline function kernel_cubic(x::T) where {T}
    return T(KERNELCONST * (ramp(1.0 - x)^3 - 4.0 * ramp(0.5 - x)^3))
end

#=
#loop over pixels, very slow and need a tree
function map_particle_to_grid(field::Vector{T}, volume::Vector{T},
        X::Array{SVector{N,T},1}, hsml::Vector{T}, boxsizes::SVector{N,T}, tree::Node{N,T}) where {N,T}
    map = zeros(T, Ngrid_x, Ngrid_y, Ngrid_z);
    Δx = BOXSIZE_X / Ngrid_x
    Δy = BOXSIZE_Y / Ngrid_y
    Δz = BOXSIZE_Z / Ngrid_z

    @time for i in CartesianIndices(map)
        #@show i.I
        Xgrid = SVector(i.I .* (Δx,Δy,Δz))
        #@show Xgrid
        idx_ngbs = get_scatter_ngb_tree(Xgrid, tree, boxsizes)
        for k in eachindex(idx_ngbs)
            j = idx_ngbs[k]
            dx = nearest.(X[j] - Xgrid, boxsizes)
            dr = norm(dx)
            Wij = kernel_cubic(dr/hsml[j]) / hsml[j]^N #divide by hsml[i]^N after the loop
            #if i == 1 @show typeof.((mass[j] , rho[j] , Wij, dr, dx)) end
            map[i] += field[j] * (volume[j] * Wij)
            #aaa = @SMatrix zeros(T,N,N)
            #aaa += 1.0
        end
    end
    return map
end
=#

#use NTuple because of simpler syntax with CartesianIndex
function idx2pos(i::NTuple{N,T}, dx::NTuple{N,T2}, xmin) where {N,T<:Int,T2<:Real}
    x = SVector(@. (i-0.5) * dx + xmin)
end

function pos2idx_right(x::SVector{N,T}, dx::NTuple{N,T}, xmin) where {N,T<:Real}
    i = convert(NTuple{N,Int}, (@. floor((x-xmin)/dx+0.5)).data )
end
function pos2idx_left(x::SVector{N,T}, dx::NTuple{N,T}, xmin) where {N,T<:Real}
    i = convert(NTuple{N,Int}, (@. floor((x-xmin)/dx+1.5)).data ) #an extra +1 to include the left-most cell
end

function pos2idxNGP(x::SVector{N,T}, dx::NTuple{N,T}, xmin) where {N,T<:Real}
    i = convert(NTuple{N,Int}, (@. ceil((x-xmin)/dx)).data )
end

#we only need to do periodic wrapping for idx (when they fall out of bound of the array)
#no periodic wrapping needed for pos
function periodic_idx(i::T, Ngrid::T, pbc::Bool) where {T<:Int}
    if pbc == false
        return i
    end
    i = i <= 0 ? i + Ngrid : i
    i = i > Ngrid ? i - Ngrid : i
    return i
end

#loop over particles; much faster and doesn't require a tree
function map_particle_to_2Dgrid_loopP(
    field::Vector{T}, volume::Vector{T}, X::Array{SVector{N,T},1}, hsml::Vector{T},
    Xmin::NTuple{N,T}, Xmax::NTuple{N,T}; xaxis::Int=1,yaxis::Int=2, column::Bool=true,
    ngrids::NTuple{N,Int}=(100,100,100), pbc::NTuple{N,Bool}=(true,true,true)) where {N,T}

    map = zeros(T, ngrids[xaxis], ngrids[yaxis]); #each thread has its own slice
    ΔX = @. (Xmax - Xmin) / ngrids
    @show ΔX

    #in the special case of a slice (Nx/y/z = 1), the following hack can speed up a lot without touching the code
    xmin,xmax,Δx=trick(Xmin,Xmax,ΔX,ngrids)

    @time @inbounds for p in eachindex(X)
        Istart = CartesianIndex(pos2idx_left( X[p] .- hsml[p], Δx, xmin))
        Iend   = CartesianIndex(pos2idx_right(X[p] .+ hsml[p], Δx, xmin))

        hsml_inv = 1.0 / hsml[p]
        f_v = field[p] * volume[p] * hsml_inv^3

        for Idx in Istart:Iend
            #@show Idx
            #only for the periodic dimensions
            iw = periodic_idx.(Idx.I, ngrids, pbc)

            #drop the out-of-bound idx (can only happen along the non-periodic dimensions after wrapping)
            if any(iw .<= 0) || any(iw .> ngrids) continue end

            #calculate distance using unwrapped idx
            r = norm(idx2pos(Idx.I,Δx, xmin) - X[p]) * hsml_inv

            map[iw[xaxis],iw[yaxis]] += kernel_cubic(r) * f_v
        end
    end
    res = map
    losdim = findfirst(((1,2,3).!=xaxis) .& ((1,2,3).!=yaxis)) #line-of-sight dimension

    #If this is a slice, we can't do column density, so use column=false instead.
    column = ngrids[losdim] == 1 ? false : column

    fac = column ? Δx[losdim] : 1.0 / ngrids[losdim]
    return res.*=fac
end

#loop over particles; much faster and doesn't require a tree
function map_particle_to_2Dgrid_loopP_thread(
    field::Vector{T}, volume::Vector{T}, X::Array{SVector{N,T},1}, hsml::Vector{T},
    Xmin::NTuple{N,T}, Xmax::NTuple{N,T}; xaxis::Int=1,yaxis::Int=2, column::Bool=true,
    ngrids::NTuple{N,Int}=(100,100,100), pbc::NTuple{N,Bool}=(true,true,true)) where {N,T}

    map = zeros(T, ngrids[xaxis], ngrids[yaxis], nthreads()); #each thread has its own slice
    ΔX = @. (Xmax - Xmin) / ngrids

    #in the special case of a slice (Nx/y/z = 1), the following hack can speed up a lot without touching the code
    xmin,xmax,Δx=trick(Xmin,Xmax,ΔX,ngrids)

    @time @inbounds @threads for p in eachindex(X)
        Istart = CartesianIndex(pos2idx_left( X[p] .- hsml[p], Δx, xmin))
        Iend   = CartesianIndex(pos2idx_right(X[p] .+ hsml[p], Δx, xmin))

        hsml_inv = 1.0 / hsml[p]
        f_v = field[p] * volume[p] * hsml_inv^3

        for Idx in Istart:Iend
            #only for the periodic dimensions
            iw = periodic_idx.(Idx.I, ngrids, pbc)

            #drop the out-of-bound idx (can only happen along the non-periodic dimensions after wrapping)
            if any(iw .<= 0) || any(iw .> ngrids) continue end

            #calculate distance using unwrapped idx
            r = norm(idx2pos(Idx.I,Δx, xmin) - X[p]) * hsml_inv

            map[iw[xaxis],iw[yaxis],threadid()] += kernel_cubic(r) * f_v
        end
    end
    res = dropdims(sum(map, dims=3),dims=3) #sum over threads
    losdim = findfirst(((1,2,3).!=xaxis) .& ((1,2,3).!=yaxis)) #line-of-sight dimension

    #If this is a slice, we can't do column density, so use column=false instead.
    column = ngrids[losdim] == 1 ? false : column

    fac = column ? Δx[losdim] : 1.0 / ngrids[losdim]
    return res.*=fac
end

#loop over particles; much faster and doesn't require a tree
function map_particle_to_3Dgrid_loopP_thread(
    field::Vector{T}, volume::Vector{T}, X::Array{SVector{N,T},1}, hsml::Vector{T},
    xmin::NTuple{N,T}, xmax::NTuple{N,T};
    ngrids::NTuple{N,Int}=(100,100,100),
    pbc::NTuple{N,Bool}=(true,true,true)) where {N,T}

    map = zeros(T, ngrids[1], ngrids[2], ngrids[3], nthreads()); #each thread has its own slice
    Δx = @. (xmax - xmin) / ngrids

    if findfirst(ngrids.==1) != nothing
        println("ngrids=", ngrids)
        error("this is a slice! use map_particle_to_2Dgrid_loopP_thread() instead...")
    end

    @time @inbounds @threads for p in eachindex(X)
        Istart = CartesianIndex(pos2idx_left( X[p] .- hsml[p], Δx, xmin))
        Iend   = CartesianIndex(pos2idx_right(X[p] .+ hsml[p], Δx, xmin))

        hsml_inv = 1.0 / hsml[p]
        f_v = field[p] * volume[p] * hsml_inv^3

        for Idx in Istart:Iend
            #only for the periodic dimensions
            iw = periodic_idx.(Idx.I, ngrids, pbc)

            #drop the out-of-bound idx (can only happen along the non-periodic dimensions after wrapping)
            if any(iw .<= 0) || any(iw .> ngrids) continue end

            #calculate distance using unwrapped idx
            r = norm(idx2pos(Idx.I,Δx, xmin) - X[p]) * hsml_inv

            idx = CartesianIndex(iw)
            map[idx,threadid()] += kernel_cubic(r) * f_v
        end
    end
    return dropdims(sum(map, dims=4),dims=4) #sum over threads
end

function trick(xmin::NTuple{N,T}, xmax::NTuple{N,T}, Δx::NTuple{N,T}, ngrids::NTuple{N,Int}) where {N,T}
    islice = findfirst(ngrids.==1)  #assume there is only one element = 1, which is the slice dimension
    if islice != nothing #we're doing a slice
        mask = ((1,0,0),(0,1,0),(0,0,1))[islice]
        xmin = xmin .- mask.*maximum(Δx).*1e3
        xmax = xmax .+ mask.*maximum(Δx).*1e3
        Δx = @. (xmax - xmin) / ngrids
        @show Δx, (xmax .+ xmin) ./ 2
    end
    #strangely this doesn't work in the thread version (which got even slower)
    #okay, in order to be fast, xmin,xmax,Δx need to have explicit type declaration (but why?)
    #turns out that we just need a diferent name Xmin,Xmax,ΔX (but why?)
    return xmin,xmax,Δx
end

#loop over particles; much faster and doesn't require a tree
function map_particle_to_2Dgrid_loopP_noCar(
    field::Vector{T}, volume::Vector{T}, X::Array{SVector{N,T},1}, hsml::Vector{T},
    Xmin::NTuple{N,T}, Xmax::NTuple{N,T}; xaxis::Int=1,yaxis::Int=2, column::Bool=true,
    ngrids::NTuple{N,Int}=(100,100,100), pbc::NTuple{N,Bool}=(true,true,true)) where {N,T}

    map = zeros(T, ngrids[xaxis], ngrids[yaxis]); #each thread has its own slice
    ΔX = @. (Xmax - Xmin) / ngrids
    @show ΔX

    #in the special case of a slice (Nx/y/z = 1), the following hack can speed up a lot without touching the code
    xmin,xmax,Δx=trick(Xmin,Xmax,ΔX,ngrids)

    @time @inbounds for p in eachindex(X)
    #@time @inbounds for i in eachindex(X)
        istart = pos2idx_left( X[p] .- hsml[p], Δx, xmin)
        iend   = pos2idx_right(X[p] .+ hsml[p], Δx, xmin)
        hsml_inv = 1.0 / hsml[p]
        f_v = field[p] * volume[p] * hsml_inv^3
        for k in istart[3]:iend[3], j in istart[2]:iend[2], i in istart[1]:iend[1]
            #@show (i,j,k)
            #only for the periodic dimensions
            iw = periodic_idx.((i,j,k), ngrids, pbc)

            #drop the out-of-bound idx (can only happen along the non-periodic dimensions after wrapping)
            if any(iw .<= 0) || any(iw .> ngrids) continue end

            #calculate distance using unwrapped idx
            r = norm(idx2pos((i,j,k),Δx, xmin) - X[p]) * hsml_inv

            map[iw[xaxis],iw[yaxis]] += kernel_cubic(r) * f_v
        end
    end
    res = map
    losdim = findfirst(((1,2,3).!=xaxis) .& ((1,2,3).!=yaxis)) #line-of-sight dimension

    #If this is a slice, we can't do column density, so use column=false instead.
    column = ngrids[losdim] == 1 ? false : column

    fac = column ? Δx[losdim] : 1.0 / ngrids[losdim]
    return res.*=fac
end

#nearest-grid-point (NGP): super fast but super noisy too
function map_particle_to_3Dgrid_NGP(field::Vector{T}, mass::Vector{T},
        X::Array{SVector{N,T},1}, ngrids::NTuple{N,Int}, xmin::NTuple{N,T}, xmax::NTuple{N,T}) where {N,T}

    map = zeros(T, ngrids[1], ngrids[2], ngrids[3]); #each thread has its own slice
    counts = zeros(T, ngrids[1], ngrids[2], ngrids[3]); #each thread has its own slice
    Δx = @. (xmax - xmin) / ngrids

    @time @inbounds for i in eachindex(X)
        #idx = periodic_idx(pos2idx(X[i], Δx))
        idx = (pos2idxNGP(X[i], Δx, xmin))
        if any(idx .<= 0) || any(idx .> ngrids) continue end
        #Idx = CartesianIndex(idx.data)
        Idx = CartesianIndex(idx)
        #map[Idx] += mass[i] / Δx^3
        map[Idx] += field[i]
        counts[Idx] += 1.0
    end
    map[counts.>0] ./= counts[counts.>0]
    return map
end

#nearest-grid-point (NGP): super fast but super noisy too
function map_particle_to_3Dgrid_NGP_thread(field::Vector{T}, mass::Vector{T},
        X::Array{SVector{N,T},1}, ngrids::NTuple{N,Int}, xmin::NTuple{N,T}, xmax::NTuple{N,T}) where {N,T}

    map = zeros(T, ngrids[1], ngrids[2], ngrids[3], nthreads()); #each thread has its own slice
    counts = zeros(T, ngrids[1], ngrids[2], ngrids[3], nthreads()); #each thread has its own slice
    Δx = @. (xmax - xmin) / ngrids

    @time @inbounds @threads for i in eachindex(X)
        #idx = periodic_idx(pos2idx(X[i], Δx))
        idx = (pos2idxNGP(X[i], Δx, xmin))
        if any(idx .<= 0) || any(idx .> ngrids) continue end
        #Idx = CartesianIndex(idx.data)
        Idx = CartesianIndex(idx)
        #map[Idx] += mass[i] / Δx^3
        map[Idx,threadid()] += field[i]
        counts[Idx,threadid()] += 1.0
    end
    #map[counts.>0] ./= counts[counts.>0]

    map_all = dropdims(sum(map, dims=4),dims=4) #sum over threads
    counts_all = dropdims(sum(counts, dims=4),dims=4) #sum over threads
    map_all[counts_all.>0] ./= counts_all[counts_all.>0]
    return map_all
end

end
