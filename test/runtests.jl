using CosmoParticleUtils
using Aqua
using ExplicitImports
using JET
using Test

using CosmoParticles
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
    end
end
