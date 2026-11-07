using CosmoParticleUtils
using Aqua
using ExplicitImports
using JET
using Test

using CosmoParticles
using FillArrays
using OffsetArrays
using Random
using StaticArrays
using Statistics
using Unitful

# interpolation at half the total mass between the middles of the particles' masses
function reference_half_mass_radius(r², mass)
    ind = sortperm(r²)
    r = sqrt.(r²[ind])
    m = mass[ind]
    c = cumsum(m) .- m ./ 2
    half = sum(m) / 2
    j = searchsortedlast(c, half)
    j == 0 && return r[1]
    j == length(c) && return r[end]
    return r[j] + (half - c[j]) / (c[j + 1] - c[j]) * (r[j + 1] - r[j])
end

# Cases (r², mass) for which the half-mass radius from the selection (without sortalg) differs
# from that of the stable sorting path.
function selection_mismatches(cases; rtol=1e-12)
    return filter(cases) do (r², m)
        selected = half_mass_radius(r², m)
        sorted = half_mass_radius(r², m; sortalg=MergeSort)
        return !isapprox(selected, sorted; rtol) || typeof(selected) != typeof(sorted)
    end
end

@testset "CosmoParticleUtils.jl" begin
    @testset "Code quality" begin
        Aqua.test_all(CosmoParticleUtils)
        @test check_no_stale_explicit_imports(CosmoParticleUtils) === nothing
        @test check_no_self_qualified_accesses(CosmoParticleUtils) === nothing
        if JET.JET_AVAILABLE
            JET.test_package(CosmoParticleUtils; target_modules=(CosmoParticleUtils,))
        end
    end

    # create regular grid in d dimensions from -val to val with n points along each dimension
    creategrid(d, n, val) =
        reduce(hcat, collect.(Iterators.product((LinRange(-val, val, n) for _ in 1:d)...)))

    f(x) = 1 / x

    # create particles
    pos2 = f.(creategrid(2, 50, 3))
    pos3 = f.(creategrid(3, 50, 3))
    mass2 = LinRange(0.01, 1, size(pos2, 2)) # masses between 0 and 1
    mass3 = LinRange(0.01, 1, size(pos3, 2)) # masses between 0 and 1

    p2 = Particles(:dm, :pos => pos2, :mass => mass2)
    p3 = Particles(:dm, :pos => pos3, :mass => mass3)

    # squared distance from center
    p2.r2 = colnorm2(p2.pos)
    p3.r2 = colnorm2(p3.pos)

    @testset "Half-mass radius" begin
        @test half_mass_radius(p2.r2, nothing) ≈ half_mass_radius(p2.r2, ones(length(p2.r2)))
        @test half_mass_radius(p3.r2, nothing) ≈ half_mass_radius(p3.r2, ones(length(p3.r2)))
        @test half_mass_radius(p2.r2, nothing) * u"m" ≈ half_mass_radius(p2.r2 * u"m^2", nothing)
        @test half_mass_radius(p3.r2, nothing) * u"m" ≈ half_mass_radius(p3.r2 * u"m^2", nothing)
        @test half_mass_radius(p2.r2, nothing; rmax=1) < 1
        @test half_mass_radius(p3.r2, nothing; rmax=1) < 1

        # sorting algorithms
        @test half_mass_radius(p3.r2, nothing; sortalg=QuickSort) ==
              half_mass_radius(p3.r2, nothing; sortalg=MergeSort)

        # pre-sorted radii
        @test half_mass_radius(sort(p2.r2), nothing; sorted=true) ≈ half_mass_radius(p2.r2, nothing)
        @test half_mass_radius(sort(p2.r2), nothing; rmax=1, sorted=true) ≈
              half_mass_radius(p2.r2, nothing; rmax=1)

        # odd number of particles
        @test half_mass_radius(p2.r2[2:end], nothing) ≈
              half_mass_radius(p2.r2[2:end], ones(length(p2.r2) - 1))
        @test half_mass_radius(p3.r2[2:end], nothing) ≈
              half_mass_radius(p3.r2[2:end], ones(length(p3.r2) - 1))

        rhalf2 = half_mass_radius(p2.r2, p2.mass)
        rhalf3 = half_mass_radius(p3.r2, p3.mass)


        @test isapprox(rhalf2, half_mass_radius(p2.r2, nothing); atol=0.1)
        @test isapprox(rhalf3, half_mass_radius(p3.r2, nothing); atol=0.1)
        @test half_mass_radius(p2.pos, p2.mass) == rhalf2
        @test half_mass_radius(p3.pos, p3.mass) == rhalf3
        @test half_mass_radius(p2.pos, p2.mass; sortalg=MergeSort) == rhalf2
        @test half_mass_radius(p2.pos * u"m", p2.mass * u"kg"; sortalg=MergeSort) == rhalf2 * u"m"

        # Particles
        @test half_mass_radius(p2) == rhalf2
        @test half_mass_radius(p3) == rhalf3
        @test half_mass_radius(p2; r2prop=:r2, sortalg=MergeSort) == rhalf2

        # equal masses, stored as a Fill vector in Particles
        pfill = Particles(:dm, :pos => pos3, :mass => 2.0)
        @test half_mass_radius(pfill) == half_mass_radius(pos3, nothing)
        @test half_mass_radius(pfill) == half_mass_radius(pos3, 2.0)

        rng = Xoshiro(1)
        for n in (1, 2, 9, 10, 1001)
            r² = rand(rng, n)
            m = rand(rng, n)
            @test half_mass_radius(r², m) ≈ reference_half_mass_radius(r², m)
            @test half_mass_radius(sort(r²), m[sortperm(r²)]; sorted=true) ≈ half_mass_radius(r², m)
            @test half_mass_radius(r², fill(3.0, n)) ≈ median(sqrt.(r²))
            @test half_mass_radius(r², nothing) ≈ median(sqrt.(r²))
            @test half_mass_radius(sort(r²), nothing; sorted=true) ≈ median(sqrt.(r²))
        end

        @test half_mass_radius([1.0, 4.0], [1.0, 1.0]) ≈ 1.5
        @test half_mass_radius([1.0, 4.0], [1.0, 1.5]) ≈ 1.6
        @test half_mass_radius([1.0, 4.0], [1.0, 3.0]) ≈ 1.75
        @test half_mass_radius([4.0], [1.0]) == 2.0
        @test half_mass_radius([4.0], nothing) == 2.0
        @test half_mass_radius([4.0, 1.0], [1.0, 0.0]) == 2.0
        @test half_mass_radius([1.0, 4.0, 9.0], [1.0, 1.0, 1.0] * u"kg") ≈ 2.0
        @test half_mass_radius(Float32[1, 4, 9], Float32[1, 1, 1]) isa Float32

        r² = rand(rng, 100)
        m = rand(rng, 100)
        @test half_mass_radius(sort(r²), m[sortperm(r²)]; rmax=0.5, sorted=true) ≈
              half_mass_radius(r², m; rmax=0.5)
        @test half_mass_radius(r², m; rmax=0.5) ≈ reference_half_mass_radius(r²[r² .≤ 0.25], m[r² .≤ 0.25])

        @test_throws "no particles within rmax" half_mass_radius([4.0, 9.0], [1.0, 1.0]; rmax=1)
        @test_throws "no particles within rmax" half_mass_radius([4.0, 9.0], [1.0, 1.0]; rmax=1, sorted=true)
        @test_throws "no particles within rmax" half_mass_radius([4.0, 9.0], nothing; rmax=1)
        @test_throws "no particles given" half_mass_radius(Float64[], nothing)
        @test_throws "no particles given" half_mass_radius(Float64[], Float64[])
        @test_throws "the total mass must be positive and finite" half_mass_radius([1.0, 4.0], [0.0, 0.0])
        @test_throws "the total mass must be positive and finite" half_mass_radius([1.0, 4.0], [1.0, NaN])
        for kwargs in ((;), (; sortalg=MergeSort), (; sorted=true), (; rmax=1.5), (; rmax=1.5, sorted=true))
            @test_throws "the squared radii must not contain NaN" half_mass_radius(
                [1.0, NaN, 4.0],
                [1.0, 1.0, 1.0];
                kwargs...,
            )
            @test_throws "the squared radii must not contain NaN" half_mass_radius([1.0, NaN, 4.0]; kwargs...)
        end
        @test_throws "the squared radii must not contain NaN" half_mass_radius([1.0 NaN; 0.0 0.0])
        @test_throws DimensionMismatch half_mass_radius([1.0, 4.0], [1.0, 1.0, 1.0])
        @test_throws DimensionMismatch half_mass_radius([1.0, 4.0], Fill(1.0, 3))

        # without sortalg, the particles around half the mass are selected without sorting all
        for (r², m) in (
            (rand(rng, 1000), rand(rng, 1000)),
            (Float64.(rand(rng, 1:20, 1000)), rand(rng, 1000)),                  # equal radii
            (rand(rng, 1000), Float64.(rand(rng, 0:2, 1000))),                   # zero masses
            (Float64.(rand(rng, 1:5, 1000)), Float64.(rand(rng, 0:1, 1000))),
        )
            @test half_mass_radius(r², m) ≈ half_mass_radius(r², m; sortalg=MergeSort)
            @test half_mass_radius(r², m) ≈ reference_half_mass_radius(r², m)
        end

        @testset "Selection against sorting" begin
            srng = Xoshiro(42)

            # random radii and masses for sizes around the switch to sorting at 32 particles
            cases = [(rand(srng, n), rand(srng, n)) for n in [1:100; 127:129; 1000; 4097] for _ in 1:20]
            @test isempty(selection_mismatches(cases))

            # few distinct radii, so that many radii equal the pivot
            cases = [
                (Float64.(rand(srng, 1:d, n)), rand(srng, n)) for d in (1, 2, 3, 10) for
                n in (2, 33, 100, 1000, 5000) for _ in 1:10
            ]
            @test isempty(selection_mismatches(cases))

            # orders that are unfavorable for the choice of the pivot
            orders = [
                n -> collect(1.0:n),
                n -> collect(n:-1.0:1),
                n -> Float64[i ≤ n ÷ 2 ? i : n - i for i in 1:n],          # organ pipe
                n -> Float64[i % 7 for i in 1:n],                          # sawtooth
                n -> Float64[isodd(i) ? i : n - i for i in 1:n],           # alternating
                n -> fill(1.0, n),
            ]
            cases =
                [(order(n), rand(srng, n)) for order in orders for n in (1, 2, 31, 32, 33, 64, 1000, 10_000)]
            @test isempty(selection_mismatches(cases))

            # zero masses, a single nonzero mass, and a dominant mass at random positions
            cases = []
            for n in (2, 5, 33, 100, 1000), _ in 1:20
                x² = rand(srng, n)
                push!(cases, (x², Float64.(rand(srng, n) .< 0.5)))
                push!(cases, (x², Float64.(rand(srng, n) .< 0.1)))
                push!(cases, (x², [i == rand(srng, 1:n) ? 1.0 : 0.0 for i in 1:n]))
                dominant = rand(srng, n)
                dominant[rand(srng, 1:n)] = 1e6
                push!(cases, (x², dominant))
                push!(cases, (x², 10.0 .^ rand(srng, -10:10, n)))
            end
            cases = filter(((r², m),) -> sum(m) > 0, cases)
            @test isempty(selection_mismatches(cases))

            # integer masses: half the mass often lies exactly between two particles, so that the
            # predecessor or successor of the particle at half the mass is outside the last range
            cases =
                [(rand(srng, n), Float64.(rand(srng, 1:2, n))) for n in [2:40; 100; 1000; 1001] for _ in 1:20]
            append!(cases, [(rand(srng, n), ones(n)) for n in 1:200])
            append!(cases, [(Float64.(rand(srng, 1:5, n)), Float64.(rand(srng, 0:2, n))) for n in 2:200])
            cases = filter(((r², m),) -> sum(m) > 0, cases)
            @test isempty(selection_mismatches(cases))
            # equal masses give the median radius
            for n in [1:200; 1000; 1001]
                x² = rand(srng, n)
                @test half_mass_radius(x², fill(2.0, n)) ≈ median(sqrt.(x²))
            end

            # other number types
            x² = rand(srng, 1000)
            w = rand(srng, 1000)
            @test isempty(selection_mismatches([(Float32.(x²), Float32.(w))]; rtol=1e-5))
            @test isempty(selection_mismatches([(x² * u"m^2", w * u"kg")]))
            @test isempty(selection_mismatches([(x², rand(srng, 1:10, 1000))]))
            @test isempty(selection_mismatches([(big.(x²), big.(w))]))
            @test half_mass_radius(x² * u"m^2", w * u"kg") ≈ half_mass_radius(x², w) * u"m"

            # indices other than 1:n, views, and rmax
            @test half_mass_radius(OffsetArray(x², -500), OffsetArray(w, -500)) ≈ half_mass_radius(x², w)
            @test half_mass_radius(view(x², 2:2:1000), view(w, 2:2:1000)) ≈
                  half_mass_radius(x²[2:2:1000], w[2:2:1000]; sortalg=MergeSort)
            for rmax in (0.1, 0.5, 0.9)
                @test half_mass_radius(x², w; rmax) ≈ half_mass_radius(x², w; rmax, sortalg=MergeSort)
            end
        end

        # radii measured from r0
        c = [1.0, 2.0, 3.0]
        @test half_mass_radius(pos3 .+ c, mass3; r0=c) ≈ rhalf3
        @test half_mass_radius(pos3 .+ c, mass3; r0=SVector(1.0, 2.0, 3.0)) ≈ rhalf3
        @test half_mass_radius(Particles(:dm, :pos => pos3 .+ c, :mass => mass3); r0=c) ≈ rhalf3
        @test_throws "r0 cannot be combined with r2prop" half_mass_radius(p3; r2prop=:r2, r0=c)

        # indices other than 1:n and views
        or² = OffsetArray(r², -50)
        om = OffsetArray(m, -50)
        perm = sortperm(r²)
        @test half_mass_radius(or², om) ≈ half_mass_radius(r², m)
        @test half_mass_radius(OffsetArray(r²[perm], 7), OffsetArray(m[perm], 7); rmax=0.5, sorted=true) ≈
              half_mass_radius(r², m; rmax=0.5)
        @test half_mass_radius(or², nothing) ≈ half_mass_radius(r², nothing)
        @test half_mass_radius(or²[begin:(end - 1)], nothing) ≈ half_mass_radius(r²[1:(end - 1)], nothing)
        @test half_mass_radius(OffsetArray(sort(r²), 7), nothing; rmax=0.5, sorted=true) ≈
              half_mass_radius(r², nothing; rmax=0.5)
        @test half_mass_radius(view(vcat(r², r²), 101:200), view(vcat(m, m), 101:200)) ≈
              half_mass_radius(r², m)
        @test half_mass_radius(view(hcat(pos3, pos3), :, axes(pos3, 2)), mass3) ≈ rhalf3
        @test_throws "offset arrays are not supported" half_mass_radius(OffsetArray(pos3, 0, -5), mass3)

        # type stability
        @inferred half_mass_radius(pos3, mass3)
        @inferred half_mass_radius(pos3)
        @inferred half_mass_radius(pos3, Fill(2.0, size(pos3, 2)))
        @inferred half_mass_radius(pos3, mass3; r0=SVector(0.1, 0.2, 0.3))
        @inferred half_mass_radius(r², m; rmax=0.5)
        @inferred half_mass_radius(r², m; sortalg=MergeSort)
        @inferred half_mass_radius(sort(r²), m; sorted=true)
        @inferred half_mass_radius(r²; rmax=0.5)
        @inferred half_mass_radius(Float32.(r²), Float32.(m))
        @inferred half_mass_radius(r² * u"m^2", m * u"kg")
    end

    @testset "Center of Mass" begin
        @test isapprox.(center_of_mass_iterative(pos2, ones(length(mass2)), 1), 0; atol=1e-15) |> all
        @test isapprox.(center_of_mass_iterative(pos3, ones(length(mass3)), 1), 0; atol=1e-15) |> all

        @test isapprox.(
            center_of_mass_iterative(pos2 * u"m", ones(length(mass2)) * u"kg", 1u"m"),
            0u"m";
            atol=1e-16u"m",
        ) |> all
        @test isapprox.(
            center_of_mass_iterative(pos3 * u"m", ones(length(mass3)) * u"kg", 1u"m"),
            0u"m";
            atol=1e-16u"m",
        ) |> all

        # test specific centers
        c2 = [0.356, 0.35]
        c3 = [0.374, 0.374, 0.373]
        @test isapprox.(center_of_mass_iterative(pos2, mass2, 1), c2; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos3, mass3, 1), c3; rtol=0.1) |> all

        @test isapprox.(center_of_mass_iterative(pos2, mass2, 1; shrinking_factor=0.1), c2; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos3, mass3, 1; shrinking_factor=0.1), c3; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos2, mass2, 1; limit_fraction=0.2), 0.45; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos3, mass3, 1; limit_fraction=0.2), 0.4; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos2, mass2, 1; limit_number=4000), c2; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos3, mass3, 1; limit_number=4000), c3; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos2, mass2, 1; limit_radius=0.2), 0.45; rtol=0.1) |> all
        @test isapprox.(center_of_mass_iterative(pos3, mass3, 1; limit_radius=0.2), 0.4; rtol=0.1) |> all

        @test center_of_mass_iterative(pos2, mass2, 1) == center_of_mass_iterative(p2, 1)
        @test center_of_mass_iterative(pos3, mass3, 1) == center_of_mass_iterative(p3, 1)

        # equal masses, stored as a Fill vector in Particles
        pfill = Particles(:dm, :pos => pos3, :mass => 2.0)
        com = center_of_mass_iterative(pos3, ones(length(mass3)), 1)
        @test center_of_mass_iterative(pfill, 1) ≈ com
        @test center_of_mass_iterative(pos3, 2.0, 1) ≈ com
        @test center_of_mass_iterative(pos3, nothing, 1) ≈ com

        @test_throws "no particles within r_start" center_of_mass_iterative(pos3 .+ 10, mass3, 1)
        @test_throws "shrinking_factor must be between 0 and 1" center_of_mass_iterative(
            pos3,
            mass3,
            1;
            shrinking_factor=0,
        )
        @test center_of_mass_iterative(pos3, mass3, 1; limit_fraction=0) isa Vector{Float64}

        # few particles: the iteration ends when the sphere contains a single particle
        fewpos = [4.0 4.2 3.8 4.0 4.0 -6.0 0.0 7.0; 3.0 3.0 3.0 3.2 2.8 0.0 -5.0 -6.0]
        @test center_of_mass_iterative(fewpos, Fill(1.0, 8), 10.0) == [4.0, 3.0]
        @test center_of_mass_iterative(fewpos .+ [0.05, 0.0], Fill(1.0, 8), 10.0) ≈ [4.05, 3.0]
        @test center_of_mass_iterative([1.0 5.0; 0.0 0.0], [1.0, 1.0], 2.0) == [1.0, 0.0]

        # information on the iteration
        info = center_of_mass_iterative_info(pos3, mass3, 1)
        @test info.center == center_of_mass_iterative(pos3, mass3, 1)
        @test center_of_mass_iterative_info(p3, 1) == info
        @test info.reason == :limit_number
        @test info.iterations > 0
        info = center_of_mass_iterative_info(pos3, mass3, 1; limit_radius=0.2)
        @test info.reason == :limit_radius
        @test info.radius ≥ 0.2
        info = center_of_mass_iterative_info([1.0 5.0; 0.0 0.0], [1.0, 1.0], 2.0)
        @test info == (; center=[1.0, 0.0], radius=2.0, n=1, iterations=1, reason=:single_particle)
        info = center_of_mass_iterative_info(fewpos, Fill(1.0, 8), 10.0; limit_radius=20.0, r0=[1.0, 1.0])
        @test info == (; center=[1.0, 1.0], radius=10.0, n=8, iterations=0, reason=:limit_radius)

        # starting sphere around r0
        c = [1.0, 2.0, 3.0]
        com = center_of_mass_iterative(pos3, mass3, 1)
        @test center_of_mass_iterative(pos3 .+ c, mass3, 1; r0=c) ≈ com .+ c
        @test center_of_mass_iterative(pos3 .+ c, mass3, 1; r0=SVector(1.0, 2.0, 3.0)) ≈ com .+ c
        @test center_of_mass_iterative(p3, 1; r0=zeros(3)) == com
        @test c == [1.0, 2.0, 3.0]
        @test_throws DimensionMismatch center_of_mass_iterative(pos3, mass3, 1; r0=[0.0, 0.0])

        # integer positions give a floating-point center
        @test center_of_mass_iterative(round.(Int, 10 .* pos3), mass3, 10) isa Vector{Float64}

        @test center_of_mass_iterative(
            view(hcat(pos3, pos3), :, axes(pos3, 2)),
            view(vcat(mass3, mass3), axes(mass3, 1)),
            1,
        ) ≈ com
        @test_throws "offset arrays are not supported" center_of_mass_iterative(
            OffsetArray(pos3, 0, -5),
            mass3,
            1,
        )

        # type stability
        @inferred center_of_mass_iterative(pos3, mass3, 1.0)
        @inferred center_of_mass_iterative(pos3, nothing, 1.0)
        @inferred center_of_mass_iterative(Float32.(pos3), Float32.(mass3), 1.0f0)
        @inferred center_of_mass_iterative(pos3 * u"m", mass3 * u"kg", 1.0u"m")
        @inferred center_of_mass_iterative_info(pos3, mass3, 1.0)
    end

    @testset "Center of Velocity" begin
        @test isapprox.(center_of_velocity(pos2, ones(length(mass2))), 0; atol=1e-15) |> all
        @test isapprox.(center_of_velocity(pos3, ones(length(mass3))), 0; atol=1e-15) |> all
        @test isapprox.(center_of_velocity(pos2, pos2, ones(length(mass2)); rmax=0.8), 0; atol=1e-15) |> all
        @test isapprox.(center_of_velocity(pos3, pos3, ones(length(mass3)); rmax=0.8), 0; atol=1e-15) |> all

        @test isapprox.(
            center_of_velocity(pos2 * u"m/s", ones(length(mass2)) * u"kg"),
            0u"m/s";
            atol=1e-15u"m/s",
        ) |> all
        @test isapprox.(
            center_of_velocity(pos2 * u"m", pos2 * u"m/s", ones(length(mass2)) * u"kg"; rmax=0.8u"m"),
            0u"m/s";
            atol=1e-15u"m/s",
        ) |> all

        @test all(center_of_velocity(pos2, pos2, ones(length(mass2)); rmax=0.8, r0=fill(0.1, 2)) .< 0.3)
        @test all(center_of_velocity(pos3, pos3, ones(length(mass3)); rmax=0.8, r0=fill(0.1, 3)) .< 0.3)

        @test all(center_of_velocity(pos2, pos2, ones(length(mass2)); rmax=0.8, p=0.5) .< 0.3)
        @test all(center_of_velocity(pos3, pos3, ones(length(mass3)); rmax=0.8, p=0.5) .< 0.3)

        @test center_of_velocity(pos2, mass2) == center_of_velocity(p2; velprop=:pos)
        @test center_of_velocity(pos3, mass3) == center_of_velocity(p3; velprop=:pos)

        @test center_of_velocity(pos3, pos3, mass3; rmax=10) == center_of_velocity(p3; rmax=10, velprop=:pos)

        @test center_of_velocity(pos3, mass3) isa Vector{Float64}
        @test center_of_velocity(pos3, pos3, mass3; rmax=0.8) isa Vector{Float64}
        @test center_of_velocity(pos3, 2.0) isa Vector{Float64}

        # the default center is the origin
        cov = center_of_velocity(pos3, pos3, mass3; rmax=0.8)
        @test center_of_velocity(pos3, pos3, mass3; rmax=0.8, r0=nothing) == cov
        @test center_of_velocity(pos3, pos3, mass3; rmax=0.8, r0=zeros(3)) == cov
        @test center_of_velocity(pos3, pos3, mass3; rmax=0.8, r0=SVector(0.0, 0.0, 0.0)) == cov

        # equal masses, stored as a Fill vector in Particles
        pfill = Particles(:dm, :pos => pos3, :vel => pos3, :mass => 2.0)
        @test center_of_velocity(pfill) == center_of_velocity(pos3, nothing)
        @test center_of_velocity(pfill; rmax=0.8) == center_of_velocity(pos3, pos3, nothing; rmax=0.8)

        vpos3 = view(hcat(pos3, pos3), :, axes(pos3, 2))
        vmass3 = view(vcat(mass3, mass3), axes(mass3, 1))
        @test center_of_velocity(vpos3, vmass3) ≈ center_of_velocity(pos3, mass3)
        @test center_of_velocity(vpos3, vpos3, vmass3; rmax=0.8) ≈
              center_of_velocity(pos3, pos3, mass3; rmax=0.8)
        @test_throws "offset arrays are not supported" center_of_velocity(
            OffsetArray(pos3, 0, -5),
            OffsetArray(mass3, -5),
        )
        @test center_of_velocity(pos2 * u"m/s", mass2 * u"kg") isa Vector{<:Unitful.Velocity}
        @test center_of_velocity(pos3 * u"m/s", mass3 * u"kg") ≈ center_of_velocity(pos3, mass3) * u"m/s"

        # rmax=nothing uses all particles, and r0 only with rmax
        @test center_of_velocity(pos3, pos3, mass3) == center_of_velocity(pos3, mass3)
        @test center_of_velocity(pos3, pos3, mass3; r0=fill(0.1, 3)) == center_of_velocity(pos3, mass3)
        @test center_of_velocity(pos3, pos3, mass3; rmax=Inf) == center_of_velocity(pos3, mass3)
        pvel = Particles(:dm, :vel => pos3, :mass => mass3) # no positions
        @test center_of_velocity(pvel) == center_of_velocity(pos3, mass3)
        @test center_of_velocity(p3; r0=fill(0.1, 3), velprop=:pos) == center_of_velocity(pos3, mass3)

        @test_throws "no particles within rmax = 0.5" center_of_velocity(pos3 .+ 5, pos3, mass3; rmax=0.5)
        @test_throws "no particles given" center_of_velocity(zeros(3, 0), Float64[])
        @test_throws "no particles given" center_of_velocity(zeros(3, 0), zeros(3, 0), nothing)
        @test_throws "pos and vel must have the same particle indices" center_of_velocity(
            pos3,
            pos3[:, 2:end],
            nothing;
            rmax=0.8,
        )
        @test_throws "vel and mass must have the same particle indices" center_of_velocity(pos3, mass3[2:end])
        @test_throws "vel and mass must have the same particle indices" center_of_velocity(
            pos3,
            pos3,
            Fill(1.0, 3);
            rmax=0.8,
        )
        @test_throws "vel and mass must have the same particle indices" center_of_velocity(
            pos3,
            pos3,
            mass3[2:end],
        )

        # type stability
        @inferred center_of_velocity(pos3, pos3, mass3; rmax=0.8)
        @inferred center_of_velocity(pos3, mass3)
        @inferred center_of_velocity(pos3, nothing)
        @inferred center_of_velocity(pos3 * u"m", pos3 * u"m/s", mass3 * u"kg"; rmax=0.8u"m")
    end

    @testset "Smoothing lengths" begin
        pos = [0.1 0.3 0.6 0.95 0.5; 0.2 0.2 0.7 0.9 0.15]
        boxlengths = [1.0, 1.0]

        # distance to the kth-nearest other particle by comparing all pairs
        function kth_distance(pos, k, dist)
            map(axes(pos, 2)) do i
                d = sort([dist(pos[:, i], pos[:, j]) for j in axes(pos, 2) if j != i])
                d[k]
            end
        end
        euclid(a, b) = sqrt(sum(abs2, a - b))
        periodic(a, b) = (d=mod.(a - b, boxlengths); sqrt(sum(abs2, min.(d, boxlengths .- d))))

        for k in 1:3
            @test smoothing_lengths(pos, k) ≈ kth_distance(pos, k, euclid)
            @test smoothing_lengths(pos, k; periodic=true, boxlengths) ≈ kth_distance(pos, k, periodic)
        end
        @test smoothing_lengths(pos, 1; periodic=true, boxlengths) != smoothing_lengths(pos, 1)

        # positions outside the box are wrapped into it when only boxlengths are given
        shifted = pos .+ [3.0, -2.0] .* (axes(pos, 2)' .% 2)
        @test smoothing_lengths(shifted, 2; periodic=true, boxlengths) ≈ kth_distance(pos, 2, periodic)
        @test smoothing_lengths(shifted, 2; periodic=true, boxlengths=SVector(1.0, 1.0)) ≈
              kth_distance(pos, 2, periodic)

        # box bounds are used without wrapping
        @test smoothing_lengths(pos, 2; periodic=true, boxcorners=([0.0, 0.0], [1.0, 1.0])) ≈
              kth_distance(pos, 2, periodic)
        @test smoothing_lengths(pos .- 0.5, 2; periodic=true, boxcorners=([-0.5, -0.5], [0.5, 0.5])) ≈
              kth_distance(pos, 2, periodic)
        @test smoothing_lengths(pos, 2; periodic=true, boxcorners=(SVector(0.0, 0.0), SVector(1.0, 1.0))) ≈
              kth_distance(pos, 2, periodic)
        corners = CosmoParticles.geometry_enclosing_corners(CosmoSquare([0.5, 0.5], 0.5))
        @test smoothing_lengths(pos, 2; periodic=true, boxcorners=corners) ≈ kth_distance(pos, 2, periodic)
        @test_throws "outside the periodic box bounds" smoothing_lengths(
            shifted,
            2;
            periodic=true,
            boxcorners=([0.0, 0.0], [1.0, 1.0]),
        )

        @test smoothing_lengths(pos * u"m", 2; periodic=true, boxlengths=boxlengths * u"m") ≈
              kth_distance(pos, 2, periodic) * u"m"

        p = Particles(:gas, :pos => pos)
        @test smoothing_lengths(p, 2) == smoothing_lengths(pos, 2)
        @test smoothing_lengths(p, 2; periodic=true, boxlengths) ==
              smoothing_lengths(pos, 2; periodic=true, boxlengths)

        @test_throws "boxlengths or boxcorners must be given" smoothing_lengths(pos, 2; periodic=true)
        @test_throws "boxcorners must contain the lower and upper box bounds" smoothing_lengths(
            pos,
            2;
            periodic=true,
            boxcorners=([0.0, 0.0],),
        )
        @test_throws "give either boxlengths or boxcorners" smoothing_lengths(
            pos,
            2;
            periodic=true,
            boxlengths,
            boxcorners=([0.0, 0.0], [1.0, 1.0]),
        )
        @test smoothing_lengths(pos, 2; boxlengths) == smoothing_lengths(pos, 2)

        @test smoothing_lengths(view(hcat(pos, pos), :, 6:10), 2) ≈ smoothing_lengths(pos, 2)
        @test smoothing_lengths(pos, Int32(2)) == smoothing_lengths(pos, 2)
        @test smoothing_lengths(Float32.(pos), 2) ≈ smoothing_lengths(pos, 2)
        @test smoothing_lengths(Float32.(pos), 2) isa Vector{Float32}
        @test smoothing_lengths(round.(Int, 10 .* pos), 1) isa Vector{Float64}
        @test_throws "k must be smaller than the number of particles" smoothing_lengths(pos, 5)
        @test_throws "offset arrays are not supported" smoothing_lengths(OffsetArray(pos, 0, -2), 2)

        # type stability
        @inferred smoothing_lengths(pos, 2)
        @inferred smoothing_lengths(Float32.(pos), 2)
        @inferred smoothing_lengths(pos, 2; periodic=true, boxlengths)
        @inferred smoothing_lengths(pos, 2; periodic=true, boxcorners=([0.0, 0.0], [1.0, 1.0]))
    end
end
