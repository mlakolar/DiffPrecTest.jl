using DiffPrecTest
using Test
using Aqua

tests = [
  "util",
  "ustats",
  "skron",
  "variance",
  "invHessian",
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

  @testset "Aqua" begin
    Aqua.test_all(DiffPrecTest)
  end
end
