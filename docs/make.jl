using CosmoParticleUtils
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(CosmoParticleUtils, :DocTestSetup, :(using CosmoParticleUtils); recursive=true)

links = InterLinks(
    "Julia" => "https://docs.julialang.org/en/v1/",
    "StatsBase" => "https://juliastats.org/StatsBase.jl/stable/",
)

makedocs(;
    plugins=[links],
    modules=[CosmoParticleUtils],
    authors="Lucas Valenzuela",
    sitename="CosmoParticleUtils.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lucasvalenzuela.github.io/CosmoParticleUtils.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/lucasvalenzuela/CosmoParticleUtils.jl", devbranch="main")
