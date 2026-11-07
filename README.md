# CosmoParticleUtils

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lucasvalenzuela.github.io/CosmoParticleUtils.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lucasvalenzuela.github.io/CosmoParticleUtils.jl/dev)
[![Build Status](https://github.com/lucasvalenzuela/CosmoParticleUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lucasvalenzuela/CosmoParticleUtils.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lucasvalenzuela/CosmoParticleUtils.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lucasvalenzuela/CosmoParticleUtils.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

CosmoParticleUtils is a Julia package that provides typical functions needed for particles in
cosmological simulations.

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
