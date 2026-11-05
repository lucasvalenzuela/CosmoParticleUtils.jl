using CosmoParticleUtils
using Documenter

DocMeta.setdocmeta!(CosmoParticleUtils, :DocTestSetup, :(using CosmoParticleUtils); recursive=true)

makedocs(;
    modules=[CosmoParticleUtils],
    authors="Lucas Valenzuela",
    sitename="CosmoParticleUtils.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lucasvalenzuela.github.io/CosmoParticleUtils.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(; repo="github.com/lucasvalenzuela/CosmoParticleUtils.jl", devbranch="main")
