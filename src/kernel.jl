norm(a::SVector{N,T}) where{N,T} = sqrt(sum(a.^2))

const GEO_FAC = 4.0 / 3.0 * pi
const KERNELCONST = 16.0 / pi
@inline ramp(x) = max(0, x);
@inline function kernel_cubic(x::T) where {T}
    return T(KERNELCONST * (ramp(1.0 - x)^3 - 4.0 * ramp(0.5 - x)^3))
end
