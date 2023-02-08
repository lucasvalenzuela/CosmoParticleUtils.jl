using CosmoParticleUtils
using Documenter

DocMeta.setdocmeta!(CosmoParticleUtils, :DocTestSetup, :(using CosmoParticleUtils); recursive=true)

makedocs(;
    modules=[CosmoParticleUtils],
    authors="Lucas Valenzuela",
    repo="https://gitlab.com/juliacosmosims/CosmoParticleUtils.jl/blob/{commit}{path}#{line}",
    sitename="CosmoParticleUtils.jl",
    devbranch="main",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliacosmosims.gitlab.io/CosmoParticleUtils.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
