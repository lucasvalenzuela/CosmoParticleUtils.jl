module CosmoParticleUtils

using CosmoParticles
using Distances
using FastUnitfulOperations
using FillArrays
using FillArrays: AbstractFillVector
using NearestNeighbors
using Statistics
using StatsBase

export half_mass_radius, center_of_mass_iterative, center_of_velocity, get_smoothing_lengths


"""
    half_mass_radius(p::AbstractParticles; posprop=:pos, massprop=:mass, r2prop=nothing)
    half_mass_radius(pos::AbstractMatrix{<:Number}, mass::Union{AbstractVector{<:Number},Number,Nothing})
    half_mass_radius(r²::AbstractVector{<:Number}, mass::Union{AbstractVector{<:Number},Number,Nothing})

Returns the half-mass radius of the particles within radius `rmax`, or of all particles if `rmax`
is `nothing`.

This works for any dimensions of the position matrix (2D, 3D, etc.).

# Keyword arguments
- `rmax::Union{Number,Nothing}=nothing`
- `sortalg=QuickSort`: sorting algorithm
- `sorted=false`: whether r² can be assumed to be sorted
"""
function half_mass_radius(pos::AbstractMatrix{<:Number}, mass=nothing; kwargs...)
    r² = colnorm2(pos)
    return half_mass_radius(r², mass; kwargs...)
end

function half_mass_radius(
    r²::AbstractVector{<:Number},
    mass::AbstractVector{T};
    rmax::Union{Number,Nothing}=nothing,
    sortalg=QuickSort,
    sorted=false,
) where {T<:Number}
    # check dimensions
    @assert length(mass) == length(r²)

    if !isnothing(rmax)
        mask = r² .≤ rmax^2
        mass = mass[mask]
        r² = r²[mask]
    end

    mass_half = 1 // 2 * sum(mass)

    # mass sorted by distance to center
    if sorted
        mass_sorted = mass
    else
        ind = usortperm(r²; alg=sortalg)
        mass_sorted = mass[ind]
    end

    # traverse masses outwardly until half of the total mass is reached
    mass_cumul = Float64(zero(T)) # set Float64 to avoid losing precision for Float32
    i = 1
    @inbounds while mass_cumul < mass_half
        mass_cumul += mass_sorted[i]
        i += 1
    end

    return 1 // 2 * (sqrt(r²[ind[i - 1]]) + sqrt(r²[ind[i - 2]]))
end

function half_mass_radius(
    r²::AbstractVector{<:Number},
    mass::Union{Number,Nothing,AbstractFillVector}=nothing;
    rmax::Union{Number,Nothing}=nothing,
    sortalg=QuickSort, # not needed due to partialsort, but don't remove!
    sorted=false,
)
    if !isnothing(rmax)
        if sorted
            lastind = searchsortedlast(r², rmax^2)
            r² = @view r²[begin:lastind]
        else
            mask = r² .≤ rmax^2
            r² = r²[mask]
        end
    end

    n = length(r²)
    n_half = n ÷ 2

    # radius sorted by distance to center
    if sorted
        r1, r2 = r²[n_half:(n_half + 1)]
    else
        r1, r2 = partialsort(r², n_half:(n_half + 1))
    end

    if iseven(n)
        return 1 // 2 * (sqrt(r1) + sqrt(r2))
    else
        return sqrt(r2)
    end
end

function half_mass_radius(p::AbstractParticles; posprop=:pos, massprop=:mass, r2prop=nothing, kwargs...)
    if !isnothing(r2prop)
        return half_mass_radius(p[r2prop], p[massprop]; kwargs...)
    else
        return half_mass_radius(p[posprop], p[massprop]; kwargs...)
    end
end


@doc raw"""
    center_of_mass_iterative(pos::AbstractMatrix{<:Number}, mass::Union{Number,AbstractVector{<:Number}}, r_start::Number)
    center_of_mass_iterative(p::AbstractParticles, r_start::Number; posprop=:pos, massprop=:mass)

Returns the center of mass by applying the shrinking sphere method
([Power et al. 2003](https://arxiv.org/abs/astro-ph/0201544)).

The sphere starts centered at the coordinate system's origin with a radius of `r_start` and
shrinks by `shrinking_factor` every iteration. The iteration is stopped when the sphere
contains fewer than `limit_number` and fewer than `limit_fraction` times the number of particles in the
starting sphere, whichever is smaller. In addition, the iteration is stopped when the radius
of the sphere drops below `limit_radius`. Note that the unit of `limit_radius` should be the
same as that of `r_start`. Throws an error if the starting sphere contains no particles.

The radius of the sphere in iteration step ``i`` is ``r_0 (1 - f_\mathrm{shrink})^i``.

# Keyword arguments
- `shrinking_factor::Real=0.025`
- `limit_fraction::Real=0.01`
- `limit_number::Integer=1000`
- `limit_radius::Number=0`
"""
function center_of_mass_iterative(
    pos::AbstractMatrix{T},
    mass,
    r_start::R;
    shrinking_factor::S=0.025,
    limit_fraction::Real=0.01,
    limit_number::Integer=1000,
    limit_radius::Number=zero(R),
) where {T<:Number,R<:Number,S<:Real}
    0 < shrinking_factor < 1 ||
        throw(ArgumentError("shrinking_factor must be between 0 and 1, got $shrinking_factor"))

    r₀ = zeros(T, size(pos, 1))
    r²_max = r_start^2
    r²_limit = (limit_radius / (oneunit(S) - shrinking_factor))^2

    # mask particles in initial sphere
    r² = colnorm2(pos, r₀)
    mask = r² .≤ r²_max # this will be reused further down
    n = count(mask)
    n > 0 || throw(ArgumentError("no particles within r_start = $r_start of the origin"))

    # get number limit for when the algorithm should stop; at least 1, so that the loop ends when
    # the sphere is empty
    nlimit = max(1, min(limit_number, ceil(Int, limit_fraction * n)))
    while n ≥ nlimit && r²_max ≥ r²_limit
        _mean_position!(r₀, pos, mass, mask)
        r²_max *= (oneunit(S) - shrinking_factor)^2

        r² = colnorm2(pos, r₀)
        mask .= r² .≤ r²_max # reuse mask array
        n = count(mask)
    end

    return r₀
end

function center_of_mass_iterative(
    p::AbstractParticles,
    r_start::Number;
    posprop=:pos,
    massprop=:mass,
    kwargs...,
)
    center_of_mass_iterative(p[posprop], p[massprop], r_start; kwargs...)
end

function _mean_position!(r₀::AbstractVector, pos::AbstractMatrix, mass::AbstractVector, mask)
    @views mean!(r₀, pos[:, mask], weights(CosmoParticles.ustrip_lazy(mass[mask])); dims=2)
end
function _mean_position!(
    r₀::AbstractVector,
    pos::AbstractMatrix,
    _::Union{Number,Nothing,AbstractFillVector},
    mask,
)
    @views mean!(r₀, pos[:, mask])
end


"""
    center_of_velocity(vel::AbstractMatrix{<:Number}, mass::Union{Number,AbstractVector{<:Number}})
    center_of_velocity(pos::AbstractMatrix{<:Number}, vel::AbstractMatrix{<:Number}, mass::Union{Number,AbstractVector{<:Number}}, radius::Number)
    center_of_velocity(p::AbstractParticles, [radius::Number]; posprop=:pos, velprop=:vel, massprop=:mass)

Returns the [mass-weighted mean](@extref StatsBase Statistics.mean) velocity vector within
`radius` around `r0`, only considering absolute velocities smaller than the `p`th
[`quantile`](@extref Statistics.quantile) (any value between 0 and 1) when centered around the
[`median`](@extref Statistics.median) velocity. The median and the quantile are not weighted.

# Keyword arguments
- `p::Real=0.9`
- `r0::Union{AbstractVector{<:Number},Nothing}=nothing`: center of the sphere; `nothing` is the
  origin
"""
function center_of_velocity(
    pos::AbstractMatrix{<:Number},
    vel::AbstractMatrix{<:Number},
    mass,
    radius::Number;
    p::Real=0.9,
    r0::Union{AbstractVector{<:Number},Nothing}=nothing,
)
    r² = isnothing(r0) ? colnorm2(pos) : colnorm2(pos, r0)
    mask = r² .≤ radius^2
    vel_mask = vel[:, mask]

    _center_of_velocity(vel_mask, mass, mask, p)
end

function center_of_velocity(vel::AbstractMatrix{<:Number}, mass; p::Real=0.9)
    _center_of_velocity(vel, mass, (:), p)
end

function _center_of_velocity(vel_mask, mass, mask, p)
    v² = colnorm2(vel_mask, dropdims(median(vel_mask; dims=2); dims=2))
    v²_max = quantile(v², p)
    mask2 = v² .≤ v²_max

    return _mean_velocity(vel_mask, mass, mask, mask2)
end

function center_of_velocity(
    p::AbstractParticles,
    radius::Number=Inf;
    posprop=:pos,
    velprop=:vel,
    massprop=:mass,
    kwargs...,
)
    if isinf(radius)
        center_of_velocity(p[velprop], p[massprop]; kwargs...)
    else
        center_of_velocity(p[posprop], p[velprop], p[massprop], radius; kwargs...)
    end
end

function _mean_velocity(vel_mask::AbstractMatrix, mass::AbstractVector, mask, mask2)
    @views vec(mean(vel_mask[:, mask2], weights(CosmoParticles.ustrip_lazy(mass[mask][mask2])); dims=2))
end
function _mean_velocity(vel_mask::AbstractMatrix, _::Union{Number,Nothing,AbstractFillVector}, _, mask2)
    @views vec(mean(vel_mask[:, mask2]; dims=2))
end


"""
    get_smoothing_lengths(pos::AbstractMatrix, k::Integer; periodic=false, boxlengths=zeros(eltype(pos), size(pos, 1)))
    get_smoothing_lengths(p::AbstractParticles, k::Integer; posprop=:pos, periodic=false, boxlengths=zeros(eltype(pos), size(pos, 1)))

Gets the nth particle neighbor distances for the passed positions for each particle.

The distances are determined with [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl)
using a `KDTree` for non-periodic and `BallTree` for periodic nearest neighbors.
The nth-nearest particle does not count the respective particle itself.

# Keyword Arguments
- `periodic=false`: whether the periodicity of the simulation box should be taken into account - this is only
  necessary if the entire box is plotted and is significantly more computational intensive than non-periodic
  smoothing
- `boxlengths::AbstractVector`: vector of the box length in each direction; only needs to be set when `periodic`
  is set to `true`

# Examples
julia```
pos = rand(3, 100)
k = 16
get_smoothing_lengths(pos, k)
```
"""
function get_smoothing_lengths(
    pos::AbstractMatrix,
    k::Integer;
    periodic=false,
    boxlengths=zeros(eltype(pos), size(pos, 1)),
)
    if periodic
        tree = BallTree(pos, PeriodicEuclidean(boxlengths))
    else
        tree = KDTree(pos, Euclidean())
    end

    idxs, dists = knn(tree, pos, k + 1)

    return maximum.(dists)
end

get_smoothing_lengths(p::AbstractParticles, k::Integer; posprop=:pos, kwargs...) =
    get_smoothing_lengths(p[posprop], k; kwargs...)


end # module
