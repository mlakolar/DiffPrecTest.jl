using DiffPrecTest
using Test
using Aqua

tests = [
  "util",
  "ustats",
  "skron",
  "variance",
  "inv_hessian",
  "reduced",
  "estimate",
  "support",
  "bootstrap"
]

@testset "DiffPrecTest" begin
  for t in tests
    f = "$t.jl"
    println("* running $f ...")
    include(f)
  end

  @testset "deprecated API" begin
    run(`$(Base.julia_cmd()) --startup-file=no --depwarn=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "deprecated.jl"))`)
  end

  @testset "Aqua" begin
    Aqua.test_all(DiffPrecTest)
  end
end
