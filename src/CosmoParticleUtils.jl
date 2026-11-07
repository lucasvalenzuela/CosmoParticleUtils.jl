module CosmoParticleUtils

using CosmoParticles
using FastUnitfulOperations
using FillArrays
using FillArrays: AbstractFillVector
using NearestNeighbors
using Statistics
using StatsBase

export half_mass_radius, center_of_mass_iterative, center_of_velocity, smoothing_lengths


"""
    half_mass_radius(pos::AbstractMatrix{<:Number}, mass=nothing; kwargs...)
    half_mass_radius(r²::AbstractVector{<:Number}, mass=nothing; kwargs...)
    half_mass_radius(p::AbstractParticles; posprop=:pos, massprop=:mass, r2prop=nothing, kwargs...)

Returns the half-mass radius of the particles around `r0` within radius `rmax`, or of all
particles if `rmax` is `nothing`.

The particles are given by their positions `pos`, a ``d × N`` matrix for any dimension ``d``, by
their squared radii `r²`, or as [`AbstractParticles`](@extref CosmoParticles.AbstractParticles)
with the properties `posprop` and `massprop`, or `r2prop` for squared radii. The masses are a
vector, or a number, a [`Fill`](@extref FillArrays.Fill) vector or `nothing` for equal masses.
Throws an error if there are no particles, or if the total mass is not positive and finite.

# Keyword arguments
- `r0::Union{AbstractVector{<:Number},Nothing}=nothing`: center from which the radii are measured;
  `nothing` means it uses the origin. Only for positions, not for `r²` or `r2prop`.
- `rmax::Union{Number,Nothing}=nothing`: radius within which the particles are considered
- `sortalg=nothing`: sorting algorithm passed to
  [`usortperm`](@extref FastUnitfulOperations.usortperm) as `alg`, e.g.
  [`QuickSort`](@extref Base.Sort.QuickSort); `nothing` uses the default of `usortperm`, which is
  faster for large vectors
- `sorted=false`: whether `r²` is sorted in increasing order, which skips the sorting

# Examples
```jldoctest
julia> using CosmoParticles, FillArrays

julia> pos = [1.0 0.0 -2.0 0.0; 0.0 3.0 0.0 -4.0];

julia> mass = [4.0, 1.0, 1.0, 1.0];

julia> half_mass_radius(pos, mass)
1.6

julia> half_mass_radius(pos, Fill(1.0, 4))
2.5

julia> half_mass_radius(pos, mass; rmax=2.5)
1.2

julia> r0 = [1.0, 2.0];

julia> half_mass_radius(pos .+ r0, mass; r0)
1.6

julia> r² = colnorm2(pos);

julia> half_mass_radius(r², mass)
1.6

julia> p = Particles(:dm, :pos => pos, :mass => mass);

julia> half_mass_radius(p)
1.6
```
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
    else
        # partialsort selects by index in the sorted vector, which has the axes of r²
        i = firstindex(r²) + k - 1
        r1, r2 = iseven(n) ? partialsort(r², i:(i + 1)) : (r=partialsort(r², i); (r, r))
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
    center_of_mass_iterative(pos::AbstractMatrix{<:Number}, mass, r_start::Number; kwargs...)
    center_of_mass_iterative(p::AbstractParticles, r_start::Number; posprop=:pos, massprop=:mass, kwargs...)

Returns the center of mass by applying the shrinking sphere method
([Power et al. 2003](https://arxiv.org/abs/astro-ph/0201544)).

The particles are given by their positions `pos`, a ``d × N`` matrix for any dimension ``d``, or
as [`AbstractParticles`](@extref CosmoParticles.AbstractParticles) with the properties `posprop`
and `massprop`. The masses are a vector, or a number, a [`Fill`](@extref FillArrays.Fill) vector or
`nothing` for equal masses.

The sphere starts centered at `r0` with a radius of `r_start` and moves to the center of mass of
the particles within it, while it shrinks by `shrinking_factor` every iteration. The iteration
is stopped when the sphere contains fewer than `limit_number` and fewer than `limit_fraction`
times the number of particles in the starting sphere, whichever is smaller. In addition, the
iteration is stopped when the radius of the sphere drops below `limit_radius`, which has the same
unit as `r_start`. Throws an error if the starting sphere contains no particles.

The radius of the sphere in iteration step ``i`` is ``r_\mathrm{start} (1 - f_\mathrm{shrink})^i``.
The center is returned as a vector of floating-point numbers.

# Keyword arguments
- `r0::Union{AbstractVector{<:Number},Nothing}=nothing`: center of the starting sphere; `nothing`
  means it uses the origin
- `shrinking_factor::Real=0.025`
- `limit_fraction::Real=0.01`
- `limit_number::Integer=1000`
- `limit_radius::Number=0`

# Examples
```jldoctest
julia> using FillArrays

julia> pos = [4.0 4.2 3.8 4.0 4.0 -6.0 0.0 7.0; 3.0 3.0 3.0 3.2 2.8 0.0 -5.0 -6.0];

julia> center_of_mass_iterative(pos, Fill(1.0, 8), 10.0)
2-element Vector{Float64}:
 4.0
 3.0
```
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

    # colnorm2 converts the center to the element type of the positions; float returns float
    # arrays without copying
    pos = float(pos)

    # the center is updated in place, so it is a new vector even if r0 is given
    r₀ = zeros(float(T), size(pos, 1))
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
        # the center of a single particle is its position, which stays within every smaller sphere
        n == 1 && break
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
    center_of_velocity(vel::AbstractMatrix{<:Number}, mass; kwargs...)
    center_of_velocity(pos::AbstractMatrix{<:Number}, vel::AbstractMatrix{<:Number}, mass, radius::Number; kwargs...)
    center_of_velocity(p::AbstractParticles, radius::Number=Inf; posprop=:pos, velprop=:vel, massprop=:mass, kwargs...)

Returns the [mass-weighted mean](@extref StatsBase Statistics.mean) velocity vector of the
particles within `radius` around `r0`, or of all particles if no positions or an infinite
`radius` are given.

Only the particles whose velocities deviate least from the
[`median`](@extref Statistics.median) velocity are included: those for which the absolute
velocity relative to the median velocity is at most the `p`th
[`quantile`](@extref Statistics.quantile) (any value between 0 and 1) of these magnitudes. The
median and the quantile are not weighted.

The particles are given by their velocities `vel` and positions `pos`, ``d × N`` matrices for any
dimension ``d``, or as [`AbstractParticles`](@extref CosmoParticles.AbstractParticles) with the
properties `posprop`, `velprop` and `massprop`. The masses are a vector, or a number, a
[`Fill`](@extref FillArrays.Fill) vector or `nothing` for equal masses.

# Keyword arguments
- `p::Real=0.9`
- `r0::Union{AbstractVector{<:Number},Nothing}=nothing`: center of the sphere; `nothing` means it
  uses the origin

# Examples
```jldoctest
julia> using FillArrays

julia> vel = [1.0 2.0 3.0 2.0 50.0; 0.0 1.0 0.0 -1.0 0.0];

julia> mass = Fill(1.0, 5);

julia> center_of_velocity(vel, mass)
2-element Vector{Float64}:
 2.0
 0.0

julia> center_of_velocity(vel, mass; p=1.0)
2-element Vector{Float64}:
 11.6
  0.0

julia> pos = [0.5 1.0 -1.0 0.0 8.0; 0.0 0.5 0.0 -1.0 0.0];

julia> center_of_velocity(pos, vel, [2.0, 1.0, 1.0, 1.0, 1.0], 2.0)
2-element Vector{Float64}:
 1.8
 0.0
```
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
    smoothing_lengths(pos::AbstractMatrix, k::Integer; kwargs...)
    smoothing_lengths(p::AbstractParticles, k::Integer; posprop=:pos, kwargs...)

Returns the distance of each particle to its `k`th-nearest neighbor, not counting the particle
itself.

The particles are given by their positions `pos`, a ``d × N`` matrix for any dimension ``d``, or
as [`AbstractParticles`](@extref CosmoParticles.AbstractParticles) with the property `posprop`.
The distances are determined with [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl)
using a `KDTree`, which is wrapped in a `PeriodicTree` for periodic boundaries.

For periodic boundaries, either `boxlengths` or `boxcorners` has to be given. With `boxlengths`,
the positions are wrapped into the box from zero to `boxlengths`, which needs a copy of the
positions. With `boxcorners`, no copy is made, and all positions have to lie within the box. The
box values are ignored if `periodic` is `false`.

# Keyword arguments
- `periodic=false`: whether to take the periodic boundaries of the simulation box into account
- `boxlengths=nothing`: vector of the box length in each direction
- `boxcorners=nothing`: tuple `(lowerleft, upperright)` of the vectors of the lower and upper box
  bounds in each direction

# Examples
```jldoctest
julia> pos = [0.0 1.0 3.0 6.0; 0.0 0.0 0.0 0.0];

julia> smoothing_lengths(pos, 2)
4-element Vector{Float64}:
 3.0
 2.0
 3.0
 5.0

julia> boxlengths = [8.0, 8.0];

julia> smoothing_lengths(pos, 2; periodic=true, boxlengths)
4-element Vector{Float64}:
 2.0
 2.0
 3.0
 3.0

julia> smoothing_lengths(pos, 2; periodic=true, boxcorners=(zero(boxlengths), boxlengths))
4-element Vector{Float64}:
 2.0
 2.0
 3.0
 3.0
```
"""
function smoothing_lengths(
    pos::AbstractMatrix,
    k::Integer;
    periodic=false,
    boxlengths=nothing,
    boxcorners=nothing,
)
    # NearestNeighbors indexes the particles from 1
    Base.require_one_based_indexing(pos)
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

    _, dists = allknn(tree, Int(k))

    return maximum.(dists)
end

smoothing_lengths(p::AbstractParticles, k::Integer; posprop=:pos, kwargs...) =
    smoothing_lengths(p[posprop], k; kwargs...)


end # module
