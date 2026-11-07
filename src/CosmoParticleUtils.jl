module CosmoParticleUtils

using CosmoParticles
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

Returns the half-mass radius of the particles around `r0` within radius `rmax`, or of all
particles if `rmax` is `nothing`.

This works for any dimensions of the position matrix (2D, 3D, etc.). Throws an error if there
are no particles, or if the total mass is not positive and finite.

# Keyword arguments
- `r0::Union{AbstractVector{<:Number},Nothing}=nothing`: center from which the radii are measured;
  `nothing` means it uses the origin. Only for positions, not for `r²` or `r2prop`.
- `rmax::Union{Number,Nothing}=nothing`
- `sortalg=nothing`: sorting algorithm passed to
  [`usortperm`](@extref FastUnitfulOperations.usortperm) as `alg`, e.g.
  [`QuickSort`](@extref Base.Sort.QuickSort); `nothing` uses the default of `usortperm`, which is
  faster for large vectors
- `sorted=false`: whether r² can be assumed to be sorted
"""
function half_mass_radius(pos::AbstractMatrix{<:Number}, mass=nothing; r0=nothing, kwargs...)
    r² = isnothing(r0) ? colnorm2(pos) : colnorm2(pos, r0)
    return half_mass_radius(r², mass; kwargs...)
end

function half_mass_radius(
    r²::AbstractVector{<:Number},
    mass::AbstractVector{<:Number};
    rmax::Union{Number,Nothing}=nothing,
    sortalg=nothing,
    sorted=false,
)
    axes(r²) == axes(mass) ||
        throw(DimensionMismatch("r² and mass must have the same indices, got $(axes(r²)) and $(axes(mass))"))

    if !isnothing(rmax)
        if sorted
            lastind = searchsortedlast(r², rmax^2)
            r² = @view r²[begin:lastind]
            mass = @view mass[begin:lastind]
        else
            mask = r² .≤ rmax^2
            r² = r²[mask]
            mass = mass[mask]
        end
    end
    isempty(r²) && _throw_no_particles(rmax)

    if sorted
        perm = eachindex(r²)
    else
        perm = isnothing(sortalg) ? usortperm(r²) : usortperm(r²; alg=sortalg)
    end

    return _half_mass_radius_sorted(r², CosmoParticles.ustrip_lazy(mass), perm)
end

_throw_no_particles(::Nothing) = throw(ArgumentError("no particles given"))
_throw_no_particles(rmax) = throw(ArgumentError("no particles within rmax = $rmax"))

# `perm` lists the indices of `r²` and `m` in order of increasing radius. Only the mass ratios
# matter, so `m` has no units; the masses are summed in at least Float64 precision.
function _half_mass_radius_sorted(r², m, perm)
    S = promote_type(eltype(m), Float64)
    total = sum(S, m)
    total > 0 && isfinite(total) ||
        throw(ArgumentError("the total mass must be positive and finite, got $total"))
    half = total / 2

    # find the first particle whose middle cumulative mass exceeds half the total mass, and
    # interpolate between it and the previous particle
    cumul = zero(S) # mass of all particles before the current one
    c_prev = zero(S)
    j_prev = first(perm)
    for j in perm
        c = cumul + m[j] / 2
        if c > half
            r = sqrt(r²[j])
            j == first(perm) && return r
            r_prev = sqrt(r²[j_prev])
            t = oftype(one(r), (half - c_prev) / (c - c_prev)) # dimensionless, in the precision of r
            return r_prev + t * (r - r_prev)
        end
        cumul += m[j]
        c_prev = c
        j_prev = j
    end

    # half the total mass is at the middle of the outermost particle
    return sqrt(r²[j_prev])
end

function half_mass_radius(
    r²::AbstractVector{<:Number},
    mass::Union{Number,Nothing,AbstractFillVector}=nothing;
    rmax::Union{Number,Nothing}=nothing,
    sortalg=nothing, # unused (partialsort), accepted so that all methods take the same keywords
    sorted=false,
)
    mass isa AbstractVector &&
        axes(mass) != axes(r²) &&
        throw(DimensionMismatch("r² and mass must have the same indices, got $(axes(r²)) and $(axes(mass))"))

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
    n > 0 || _throw_no_particles(rmax)

    # median radius: the middle element for odd n, the mean of the two middle elements for even n
    k = (n + 1) ÷ 2
    if sorted
        r1 = r²[firstindex(r²) + k - 1]
        r2 = r²[lastindex(r²) - k + 1]
    elseif iseven(n)
        r1, r2 = partialsort(r², k:(k + 1))
    else
        r1 = r2 = partialsort(r², k)
    end

    return 1 // 2 * (sqrt(r1) + sqrt(r2))
end

function half_mass_radius(p::AbstractParticles; posprop=:pos, massprop=:mass, r2prop=nothing, kwargs...)
    if !isnothing(r2prop)
        haskey(kwargs, :r0) && throw(ArgumentError("r0 cannot be combined with r2prop"))
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

The sphere starts centered at `r0` with a radius of `r_start` and
shrinks by `shrinking_factor` every iteration. The iteration is stopped when the sphere
contains fewer than `limit_number` and fewer than `limit_fraction` times the number of particles in the
starting sphere, whichever is smaller. In addition, the iteration is stopped when the radius
of the sphere drops below `limit_radius`. Note that the unit of `limit_radius` should be the
same as that of `r_start`. Throws an error if the starting sphere contains no particles.

The radius of the sphere in iteration step ``i`` is ``r_\mathrm{start} (1 - f_\mathrm{shrink})^i``.

# Keyword arguments
- `r0::Union{AbstractVector{<:Number},Nothing}=nothing`: center of the starting sphere; `nothing`
  means it uses the origin
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
    r0::Union{AbstractVector{<:Number},Nothing}=nothing,
) where {T<:Number,R<:Number,S<:Real}
    0 < shrinking_factor < 1 ||
        throw(ArgumentError("shrinking_factor must be between 0 and 1, got $shrinking_factor"))

    # the center is updated in place, so it is a new vector even if r0 is given
    r₀ = zeros(T, size(pos, 1))
    if !isnothing(r0)
        length(r0) == length(r₀) ||
            throw(DimensionMismatch("r0 must have $(length(r₀)) elements, got $(length(r0))"))
        copyto!(r₀, r0)
    end
    r²_max = r_start^2
    r²_limit = (limit_radius / (oneunit(S) - shrinking_factor))^2

    # mask particles in initial sphere
    r² = colnorm2(pos, r₀)
    mask = r² .≤ r²_max # this will be reused further down
    n = count(mask)
    n > 0 || throw(ArgumentError("no particles within r_start = $r_start of r0"))

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
- `r0::Union{AbstractVector{<:Number},Nothing}=nothing`: center of the sphere; `nothing` means it
  uses the origin
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
    get_smoothing_lengths(pos::AbstractMatrix, k::Integer; periodic=false, boxlengths=nothing, boxcorners=nothing)
    get_smoothing_lengths(p::AbstractParticles, k::Integer; posprop=:pos, kwargs...)

Returns the distance of each particle to its `k`th-nearest neighbor, not counting the particle
itself.

The distances are determined with [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl)
using a `KDTree`, which is wrapped in a `PeriodicTree` for periodic boundaries.

For periodic boundaries, either `boxlengths` or `boxcorners` has to be given. With `boxlengths`,
the positions are wrapped into the box from zero to `boxlengths`, which needs a copy of the
positions. With `boxcorners`, no copy is made, and all positions have to lie within the box. The
box values are ignored if `periodic` is `false`.

# Keyword Arguments
- `periodic=false`: whether to take the periodic boundaries of the simulation box into account
- `boxlengths=nothing`: vector of the box length in each direction
- `boxcorners=nothing`: tuple `(lowerleft, upperright)` of the vectors of the lower and upper box bounds
  in each direction

# Examples
```julia
pos = rand(3, 100)
k = 16
get_smoothing_lengths(pos, k)
```
"""
function get_smoothing_lengths(
    pos::AbstractMatrix,
    k::Integer;
    periodic=false,
    boxlengths=nothing,
    boxcorners=nothing,
)
    if periodic
        if !isnothing(boxlengths)
            isnothing(boxcorners) || throw(ArgumentError("give either boxlengths or boxcorners, not both"))
            # periodic distances do not change when positions are shifted by box lengths
            pos = mod.(pos, boxlengths)
            boxmin, boxmax = zero.(boxlengths), boxlengths
        elseif !isnothing(boxcorners)
            length(boxcorners) == 2 || throw(
                ArgumentError("boxcorners must contain the lower and upper box bounds, got $boxcorners"),
            )
            boxmin, boxmax = boxcorners
        else
            throw(ArgumentError("boxlengths or boxcorners must be given when periodic is true"))
        end
        tree = PeriodicTree(KDTree(pos), boxmin, boxmax)
    else
        tree = KDTree(pos)
    end

    _, dists = knn(tree, pos, k + 1)

    return maximum.(dists)
end

get_smoothing_lengths(p::AbstractParticles, k::Integer; posprop=:pos, kwargs...) =
    get_smoothing_lengths(p[posprop], k; kwargs...)


end # module
