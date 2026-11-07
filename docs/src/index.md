```@meta
CurrentModule = CosmoParticleUtils
```

# CosmoParticleUtils

[CosmoParticleUtils](https://github.com/lucasvalenzuela/CosmoParticleUtils.jl) provides typical
functions needed for particles in cosmological simulations: half-mass radii, centers of mass and
of velocity, and smoothing lengths. The functions work with position and velocity matrices, with
[Unitful](https://github.com/JuliaPhysics/Unitful.jl) quantities, and with the particles of
[CosmoParticles](https://github.com/lucasvalenzuela/CosmoParticles.jl).

## Installation

This package is available through the public registry
[CosmoAnalysisRegistry](https://github.com/lucasvalenzuela/CosmoAnalysisRegistry). Add the registry
once per Julia installation, together with the General registry, which provides the dependencies:
```julia
using Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(url="https://github.com/lucasvalenzuela/CosmoAnalysisRegistry")
```

The package requires Julia 1.12 or newer and can then be installed with:
```julia
Pkg.add("CosmoParticleUtils")
```

## Usage

```julia
using CosmoParticleUtils, CosmoParticles

# positions and velocities as 3×N matrices
n = 10_000
pos = 100 .* rand(3, n)
vel = 200 .* randn(3, n)
p = Particles(:dm, :pos => pos, :vel => vel, :mass => rand(n))

com = center_of_mass_iterative(p, 50.0)         # shrinking sphere starting at the origin
cov = center_of_velocity(p; rmax=20.0, r0=com)  # mean velocity within 20 around the center
rhalf = half_mass_radius(p; r0=com, rmax=30.0)  # half-mass radius within 30 around the center

# distance of each particle to its 32nd-nearest neighbor in the periodic box from 0 to 100
boxcorners = ([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])
hsml = smoothing_lengths(p, 32; periodic=true, boxcorners)
```

## Reference

```@autodocs
Modules = [CosmoParticleUtils]
```
