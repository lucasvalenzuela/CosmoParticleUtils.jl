using CosmoParticleUtils
using Aqua
using ExplicitImports
using JET
using Test

using CosmoParticles
using FillArrays
using Random
using StaticArrays
using Statistics
using Unitful

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
        @test_throws DimensionMismatch half_mass_radius([1.0, 4.0], [1.0, 1.0, 1.0])
        @test_throws DimensionMismatch half_mass_radius([1.0, 4.0], Fill(1.0, 3))

        # radii measured from r0
        c = [1.0, 2.0, 3.0]
        @test half_mass_radius(pos3 .+ c, mass3; r0=c) ≈ rhalf3
        @test half_mass_radius(pos3 .+ c, mass3; r0=SVector(1.0, 2.0, 3.0)) ≈ rhalf3
        @test half_mass_radius(Particles(:dm, :pos => pos3 .+ c, :mass => mass3); r0=c) ≈ rhalf3
        @test_throws "r0 cannot be combined with r2prop" half_mass_radius(p3; r2prop=:r2, r0=c)
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

        # starting sphere around r0
        c = [1.0, 2.0, 3.0]
        com = center_of_mass_iterative(pos3, mass3, 1)
        @test center_of_mass_iterative(pos3 .+ c, mass3, 1; r0=c) ≈ com .+ c
        @test center_of_mass_iterative(pos3 .+ c, mass3, 1; r0=SVector(1.0, 2.0, 3.0)) ≈ com .+ c
        @test center_of_mass_iterative(p3, 1; r0=zeros(3)) == com
        @test c == [1.0, 2.0, 3.0]
        @test_throws DimensionMismatch center_of_mass_iterative(pos3, mass3, 1; r0=[0.0, 0.0])
    end

    @testset "Center of Velocity" begin
        @test isapprox.(center_of_velocity(pos2, ones(length(mass2))), 0; atol=1e-15) |> all
        @test isapprox.(center_of_velocity(pos3, ones(length(mass3))), 0; atol=1e-15) |> all
        @test isapprox.(center_of_velocity(pos2, pos2, ones(length(mass2)), 0.8), 0; atol=1e-15) |> all
        @test isapprox.(center_of_velocity(pos3, pos3, ones(length(mass3)), 0.8), 0; atol=1e-15) |> all

        @test isapprox.(
            center_of_velocity(pos2 * u"m/s", ones(length(mass2)) * u"kg"),
            0u"m/s";
            atol=1e-15u"m/s",
        ) |> all
        @test isapprox.(
            center_of_velocity(pos2 * u"m", pos2 * u"m/s", ones(length(mass2)) * u"kg", 0.8u"m"),
            0u"m/s";
            atol=1e-15u"m/s",
        ) |> all

        @test all(center_of_velocity(pos2, pos2, ones(length(mass2)), 0.8; r0=fill(0.1, 2)) .< 0.3)
        @test all(center_of_velocity(pos3, pos3, ones(length(mass3)), 0.8; r0=fill(0.1, 3)) .< 0.3)

        @test all(center_of_velocity(pos2, pos2, ones(length(mass2)), 0.8; p=0.5) .< 0.3)
        @test all(center_of_velocity(pos3, pos3, ones(length(mass3)), 0.8; p=0.5) .< 0.3)

        @test center_of_velocity(pos2, mass2) == center_of_velocity(p2; velprop=:pos)
        @test center_of_velocity(pos3, mass3) == center_of_velocity(p3; velprop=:pos)

        @test center_of_velocity(pos3, pos3, mass3, 10) == center_of_velocity(p3, 10; velprop=:pos)

        @test center_of_velocity(pos3, mass3) isa Vector{Float64}
        @test center_of_velocity(pos3, pos3, mass3, 0.8) isa Vector{Float64}
        @test center_of_velocity(pos3, 2.0) isa Vector{Float64}

        # the default center is the origin
        cov = center_of_velocity(pos3, pos3, mass3, 0.8)
        @test center_of_velocity(pos3, pos3, mass3, 0.8; r0=nothing) == cov
        @test center_of_velocity(pos3, pos3, mass3, 0.8; r0=zeros(3)) == cov
        @test center_of_velocity(pos3, pos3, mass3, 0.8; r0=SVector(0.0, 0.0, 0.0)) == cov

        # equal masses, stored as a Fill vector in Particles
        pfill = Particles(:dm, :pos => pos3, :vel => pos3, :mass => 2.0)
        @test center_of_velocity(pfill) == center_of_velocity(pos3, nothing)
        @test center_of_velocity(pfill, 0.8) == center_of_velocity(pos3, pos3, nothing, 0.8)
        @test center_of_velocity(pos2 * u"m/s", mass2 * u"kg") isa Vector{<:Unitful.Velocity}
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
            @test get_smoothing_lengths(pos, k) ≈ kth_distance(pos, k, euclid)
            @test get_smoothing_lengths(pos, k; periodic=true, boxlengths) ≈ kth_distance(pos, k, periodic)
        end
        @test get_smoothing_lengths(pos, 1; periodic=true, boxlengths) != get_smoothing_lengths(pos, 1)

        # positions outside the box are wrapped into it when only boxlengths are given
        shifted = pos .+ [3.0, -2.0] .* (axes(pos, 2)' .% 2)
        @test get_smoothing_lengths(shifted, 2; periodic=true, boxlengths) ≈ kth_distance(pos, 2, periodic)

        # box bounds are used without wrapping
        @test get_smoothing_lengths(pos, 2; periodic=true, boxcorners=([0.0, 0.0], [1.0, 1.0])) ≈
              kth_distance(pos, 2, periodic)
        @test get_smoothing_lengths(pos .- 0.5, 2; periodic=true, boxcorners=([-0.5, -0.5], [0.5, 0.5])) ≈
              kth_distance(pos, 2, periodic)
        @test_throws "outside the periodic box bounds" get_smoothing_lengths(
            shifted,
            2;
            periodic=true,
            boxcorners=([0.0, 0.0], [1.0, 1.0]),
        )

        @test get_smoothing_lengths(pos * u"m", 2; periodic=true, boxlengths=boxlengths * u"m") ≈
              kth_distance(pos, 2, periodic) * u"m"

        p = Particles(:gas, :pos => pos)
        @test get_smoothing_lengths(p, 2) == get_smoothing_lengths(pos, 2)
        @test get_smoothing_lengths(p, 2; periodic=true, boxlengths) ==
              get_smoothing_lengths(pos, 2; periodic=true, boxlengths)

        @test_throws "boxlengths or boxcorners must be given" get_smoothing_lengths(pos, 2; periodic=true)
        @test_throws "boxcorners must contain the lower and upper box bounds" get_smoothing_lengths(
            pos,
            2;
            periodic=true,
            boxcorners=([0.0, 0.0],),
        )
        @test_throws "give either boxlengths or boxcorners" get_smoothing_lengths(
            pos,
            2;
            periodic=true,
            boxlengths,
            boxcorners=([0.0, 0.0], [1.0, 1.0]),
        )
        @test get_smoothing_lengths(pos, 2; boxlengths) == get_smoothing_lengths(pos, 2)
    end
end
