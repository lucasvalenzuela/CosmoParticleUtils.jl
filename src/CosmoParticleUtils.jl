module CosmoParticleUtils

using CosmoParticles
using FastUnitfulOperations
using FillArrays
using FillArrays: AbstractFillVector
using NearestNeighbors
using Statistics
using StatsBase

export half_mass_radius,
    center_of_mass_iterative, center_of_mass_iterative_info, center_of_velocity, smoothing_lengths


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
  [`usortperm`](@extref FastUnitfulOperations.usortperm) as `alg` to sort the particles by
  radius, e.g. [`QuickSort`](@extref Base.Sort.QuickSort) or a multithreaded algorithm. With
  `nothing`, the particles around half the mass are selected without sorting all particles, which
  is 1.5–4 times faster than single-threaded sorting for 10⁴–10⁷ particles.
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

    m = CosmoParticles.ustrip_lazy(mass)
    sorted && return _half_mass_radius_sorted(r², m, eachindex(r²))
    isnothing(sortalg) && return _half_mass_radius_select(r², m)
    return _half_mass_radius_sorted(r², m, usortperm(r²; alg=sortalg))
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
            j == first(perm) && return sqrt(r²[j])
            return _interpolate_radius(r²[j_prev], r²[j], c_prev, c, half)
        end
        cumul += m[j]
        c_prev = c
        j_prev = j
    end

    # half the total mass is at the middle of the outermost particle
    return sqrt(r²[j_prev])
end

# Radius between the particles with squared radii r²_prev and r², linearly interpolated at `half`
# between their cumulative masses c_prev and c.
function _interpolate_radius(r²_prev, r², c_prev, c, half)
    r_prev = sqrt(r²_prev)
    r = sqrt(r²)
    t = oftype(one(r), (half - c_prev) / (c - c_prev)) # dimensionless, in the precision of r
    return r_prev + t * (r - r_prev)
end

# Same result as `_half_mass_radius_sorted` without sorting all particles: a weighted quickselect
# finds the particle k at which the cumulative mass exceeds half the total mass. Partitioning the
# particles around a pivot radius moves those closer than the pivot to the front; only the part
# that contains k is partitioned further. Particles with equal radii are ordered by index, as in a
# stable sort.
function _half_mass_radius_select(r², m)
    n = length(r²)
    S = promote_type(eltype(m), Float64)
    v = copyto!(Vector{eltype(r²)}(undef, n), r²)
    w = copyto!(Vector{S}(undef, n), m)
    id = collect(1:n) # original positions, which order particles with equal radii
    total = sum(w)
    total > 0 && isfinite(total) ||
        throw(ArgumentError("the total mass must be positive and finite, got $total"))
    half = total / 2

    # v[lo:hi] contains k; all particles in v[1:lo-1] come before it in sorted order and all in
    # v[hi+1:end] after it. `before` is the mass of v[1:lo-1]. Short ranges are sorted directly.
    lo, hi = 1, n
    before = zero(S)
    while hi - lo ≥ 32
        p = _median_of_three(v[lo], v[(lo + hi) ÷ 2], v[hi])
        lt, gt, mass_less, mass_equal = _partition3!(v, w, id, lo, hi, p)
        if before + mass_less > half
            # k is closer than the pivot
            hi = lt - 1
        elseif before + mass_less + mass_equal > half
            # k has the pivot radius: v[lt:gt] only needs to be ordered by index
            before += mass_less
            lo, hi = lt, gt
            break
        else
            # k is farther than the pivot
            before += mass_less + mass_equal
            lo = gt + 1
        end
    end

    # order the remaining range by radius and index, and find k in it
    perm = sortperm(collect(zip(view(v, lo:hi), view(id, lo:hi))))
    v[lo:hi] = v[lo:hi][perm]
    w[lo:hi] = w[lo:hi][perm]
    id[lo:hi] = id[lo:hi][perm]
    # k is the first particle whose cumulative mass including itself exceeds half the total mass,
    # or the last one in the range; `before` becomes the mass of all particles before k
    k = lo
    while k < hi && before + w[k] ≤ half
        before += w[k]
        k += 1
    end

    # interpolate between the middles of k and its predecessor, or of k and its successor; these
    # can lie outside v[lo:hi] and are then the closest particles outside of it in sorted order
    c = before + w[k] / 2
    if c > half
        prev = k > lo ? k - 1 : _argmax_before(v, id, lo)
        isnothing(prev) && return sqrt(v[k])
        return _interpolate_radius(v[prev], v[k], before - w[prev] / 2, c, half)
    else
        next = k < hi ? k + 1 : _argmin_after(v, id, hi)
        isnothing(next) && return sqrt(v[k])
        return _interpolate_radius(v[k], v[next], c, before + w[k] + w[next] / 2, half)
    end
end

_median_of_three(a, b, c) = a < b ? (b < c ? b : max(a, c)) : (a < c ? a : max(b, c))

# Partitions v[lo:hi] (with w and id) into values less than, equal to, and greater than p. Returns
# the range lt:gt of the values equal to p and the masses of the values less than and equal to p.
function _partition3!(v, w, id, lo, hi, p)
    lt, i, gt = lo, lo, hi
    mass_less = zero(eltype(w))
    mass_equal = zero(eltype(w))
    while i ≤ gt
        if v[i] < p
            mass_less += w[i]
            _swap!(v, w, id, i, lt)
            lt += 1
            i += 1
        elseif v[i] > p
            _swap!(v, w, id, i, gt)
            gt -= 1
        else
            mass_equal += w[i]
            i += 1
        end
    end
    return lt, gt, mass_less, mass_equal
end

function _swap!(v, w, id, i, j)
    v[i], v[j] = v[j], v[i]
    w[i], w[j] = w[j], w[i]
    id[i], id[j] = id[j], id[i]
    return nothing
end

# position of the last particle in sorted order among v[1:lo-1], or nothing if there is none
function _argmax_before(v, id, lo)
    lo > 1 || return nothing
    best = 1
    for i in 2:(lo - 1)
        (v[i], id[i]) > (v[best], id[best]) && (best = i)
    end
    return best
end

# position of the first particle in sorted order among v[hi+1:end], or nothing if there is none
function _argmin_after(v, id, hi)
    hi < length(v) || return nothing
    best = hi + 1
    for i in (hi + 2):lastindex(v)
        (v[i], id[i]) < (v[best], id[best]) && (best = i)
    end
    return best
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
[`center_of_mass_iterative_info`](@ref) also returns the final radius, the number of particles, and
why the iteration stopped.

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
function center_of_mass_iterative(pos::AbstractMatrix{<:Number}, mass, r_start::Number; kwargs...)
    return center_of_mass_iterative_info(pos, mass, r_start; kwargs...).center
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

"""
    center_of_mass_iterative_info(pos::AbstractMatrix{<:Number}, mass, r_start::Number; kwargs...)
    center_of_mass_iterative_info(p::AbstractParticles, r_start::Number; posprop=:pos, massprop=:mass, kwargs...)

Returns the center of mass of [`center_of_mass_iterative`](@ref) together with information on
the iteration, as a named tuple with the fields

- `center`: the center of mass
- `radius`: the radius of the last sphere, whose particles determine the center
- `n`: the number of particles in that sphere
- `iterations`: the number of times the center was updated
- `reason`: why the iteration stopped: `:limit_number` if the next sphere contains too few
  particles, `:limit_radius` if its radius is below the limit, or `:single_particle` if the last
  sphere contains a single particle

The arguments and keyword arguments are the same as for [`center_of_mass_iterative`](@ref).

# Examples
```jldoctest
julia> using FillArrays

julia> pos = [4.0 4.2 3.8 4.0 4.0 -6.0 0.0 7.0; 3.0 3.0 3.0 3.2 2.8 0.0 -5.0 -6.0];

julia> mass = Fill(1.0, 8);

julia> info = center_of_mass_iterative_info(pos, mass, 10.0);

julia> info.center
2-element Vector{Float64}:
 4.0
 3.0

julia> info.n, info.reason
(1, :single_particle)

julia> info = center_of_mass_iterative_info(pos, mass, 10.0; limit_fraction=0.5);

julia> info.n, info.reason
(5, :limit_number)
```
"""
function center_of_mass_iterative_info(
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
    r₀_prev = similar(r₀)
    shrink = oneunit(S) - shrinking_factor
    r²_max = r_start^2
    r²_limit = (limit_radius / shrink)^2

    # positions and weights (masses without units) of the particles in the current sphere; the
    # first n columns are used, and particles leaving the sphere are removed in place
    ind = findall(≤(r²_max), colnorm2(pos, r₀))
    n = length(ind)
    n > 0 || throw(ArgumentError("no particles within r_start = $r_start of r0"))
    sphere_pos = pos[:, ind]
    sphere_w = _sphere_weights(mass, ind)
    # weighted sum of the positions in the sphere, in at least Float64 precision
    possum = zeros(typeof(oneunit(float(T)) * 1.0), size(pos, 1))
    wsum = _weighted_sum!(possum, sphere_pos, sphere_w, n)

    # get number limit for when the algorithm should stop; at least 1, so that the loop ends when
    # the sphere is empty
    nlimit = max(1, min(limit_number, ceil(Int, limit_fraction * n)))
    r²_last = r²_max
    n_last = n
    iterations = 0
    while n ≥ nlimit && r²_max ≥ r²_limit
        copyto!(r₀_prev, r₀)
        r₀ .= possum ./ wsum
        iterations += 1
        r²_last = r²_max
        n_last = n
        # the center of a single particle is its position, which stays within every smaller sphere
        n == 1 && return (; center=r₀, radius=sqrt(r²_last), n=n_last, iterations, reason=:single_particle)
        r²_max *= shrink^2

        # the new sphere lies within the last one if the center moved by less than the difference
        # of their radii; then only the particles of the last sphere can be in the new one
        shift = sqrt(sum(i -> abs2(r₀[i] - r₀_prev[i]), eachindex(r₀, r₀_prev)))
        if sqrt(r²_max) + shift ≤ sqrt(r²_last)
            n, wsum = _keep_within!(possum, sphere_pos, sphere_w, n, r₀, r²_max)
        else
            ind = findall(≤(r²_max), colnorm2(pos, r₀))
            n = length(ind)
            sphere_pos = pos[:, ind]
            sphere_w = _sphere_weights(mass, ind)
            wsum = _weighted_sum!(possum, sphere_pos, sphere_w, n)
        end
    end

    reason = n < nlimit ? :limit_number : :limit_radius
    return (; center=r₀, radius=sqrt(r²_last), n=n_last, iterations, reason)
end

function center_of_mass_iterative_info(
    p::AbstractParticles,
    r_start::Number;
    posprop=:pos,
    massprop=:mass,
    kwargs...,
)
    center_of_mass_iterative_info(p[posprop], p[massprop], r_start; kwargs...)
end

# weights of the particles in the sphere; equal masses are not needed for the mean position
_sphere_weights(mass::AbstractVector, ind) = float.(CosmoParticles.ustrip_lazy(view(mass, ind)))
_sphere_weights(::Union{Number,Nothing,AbstractFillVector}, _) = nothing

_weight(w::AbstractVector, j) = w[j]
_weight(::Nothing, _) = true

# Sets `possum` to the weighted sum of the first n positions and returns the sum of the weights.
function _weighted_sum!(possum, sphere_pos, sphere_w, n)
    fill!(possum, zero(eltype(possum)))
    wsum = 0.0
    for j in 1:n
        wj = _weight(sphere_w, j)
        for d in eachindex(possum)
            possum[d] += wj * sphere_pos[d, j]
        end
        wsum += wj
    end
    return wsum
end

# Moves the particles among the first n that lie within the sphere to the front, sets `possum` to
# their weighted sum of positions, and returns their number and the sum of their weights.
function _keep_within!(possum, sphere_pos, sphere_w, n, r₀, r²_max)
    fill!(possum, zero(eltype(possum)))
    wsum = 0.0
    k = 0
    for j in 1:n
        r² = zero(abs2(zero(eltype(sphere_pos))))
        for d in eachindex(r₀)
            r² += abs2(sphere_pos[d, j] - r₀[d])
        end
        r² ≤ r²_max || continue
        k += 1
        wj = _weight(sphere_w, j)
        for d in eachindex(possum)
            sphere_pos[d, k] = sphere_pos[d, j]
            possum[d] += wj * sphere_pos[d, k]
        end
        isnothing(sphere_w) || (sphere_w[k] = wj)
        wsum += wj
    end
    return k, wsum
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
using a `KDTree`, which is wrapped in a `PeriodicTree` for periodic boundaries. The particles are
distributed over all available threads.

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

    n = size(pos, 2)
    k < n ||
        throw(ArgumentError("k must be smaller than the number of particles, got k = $k for $n particles"))
    return _kth_neighbor_distances(tree, pos, Int(k))
end

# Distance of each particle to its kth-nearest neighbor. The particles are processed in parallel
# chunks with their own buffers, so that only one distance per particle is stored.
function _kth_neighbor_distances(tree, pos, k)
    h = Vector{float(eltype(pos))}(undef, size(pos, 2))
    chunks = Iterators.partition(eachindex(h), cld(length(h), 4 * Threads.nthreads()))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            idxs = Vector{Int}(undef, k + 1)
            dists = similar(h, k + 1)
            for i in chunk
                # the k + 1 nearest particles include the particle itself
                knn!(idxs, dists, tree, view(pos, :, i), k + 1)
                h[i] = maximum(dists)
            end
        end
    end
    foreach(wait, tasks)
    return h
end

smoothing_lengths(p::AbstractParticles, k::Integer; posprop=:pos, kwargs...) =
    smoothing_lengths(p[posprop], k; kwargs...)


end # module
