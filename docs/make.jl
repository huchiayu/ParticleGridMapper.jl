using ParticleGridMapper
using Documenter

DocMeta.setdocmeta!(ParticleGridMapper, :DocTestSetup, :(using ParticleGridMapper); recursive=true)

makedocs(;
    modules=[ParticleGridMapper],
    authors="Chia-Yu Hu <cyhu.astro@gmail.com> and contributors",
    repo="https://github.com/huchiayu/ParticleGridMapper.jl/blob/{commit}{path}#{line}",
    sitename="ParticleGridMapper.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://huchiayu.github.io/ParticleGridMapper.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/huchiayu/ParticleGridMapper.jl",
)
