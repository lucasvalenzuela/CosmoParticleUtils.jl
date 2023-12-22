```@meta
CurrentModule = CosmoParticleUtils
```

# CosmoParticleUtils

[`CosmoParticleUtils`](https://gitlab.com/juliacosmosims/CosmoParticleUtils.jl) provides some typical functions
needed for dealing with particles in cosmological simulations.

## Installation

To install the package, make sure the JuliaCosmoSims registry is installed
([JuliaCosmoSimsDocs](https://juliacosmosims.gitlab.io/JuliaCosmoSimsDocs/dev/installation)).

The latest version of the package is available for Julia 1.7 and newer versions and can be installed with:

```julia
using Pkg
Pkg.add("CosmoParticleUtils")
```

## Functions

```@docs
half_mass_radius
center_of_mass_iterative
center_of_velocity
```
