module ReducedTest

using Test
using Random, LinearAlgebra, Statistics, SparseArrays
using JuMP, Ipopt
using ProximalBase
using ProximalOPT: CDSolver
using DiffPrecTest
using DiffPrecTest: TwoSampleUstatistics, svec, skron

Random.seed!(5)

include("data.jl")
generateData(p) = generateData(p, 100)[1:2]

# minimize x'Ax/2 + b'x + λ ∑ ωᵢ|xᵢ| over x = u - v with u, v ≥ 0
function solve_pen_quad(A, b, λ, ω)
  p = size(A, 1)
  model = Model(Ipopt.Optimizer)
  set_silent(model)
  @variable(model, u[1:p] >= 0)
  @variable(model, v[1:p] >= 0)
  @expression(model, x, u .- v)
  @objective(model, Min, x' * (Matrix(A) / 2.) * x + dot(b, x) + λ * dot(ω, u .+ v))
  optimize!(model)
  termination_status(model) == LOCALLY_SOLVED || error("Ipopt: ", termination_status(model))
  JuMP.value.(x)
end


@testset "_computeVarStep1!" begin
  function createVarianceGrad1(Δ::SparseIterate, rowIndex::Int)
    @inline function h(x, y)::Float64
      # skron(vx_j, vy_k)[ri, :] * Δ
      out = 0.
      for inz = 1:nnz(Δ)
        indColumn = Δ.nzval2ind[inz]          
        out += skron(x, y, rowIndex, indColumn) * Δ.nzval[inz]
      end  
      out + svec(x, x, rowIndex) - svec(y, y, rowIndex)
    end
    TwoSampleUstatistics{1, 1}(h)
  end

  function varStep1Alt!(out, X, Y, θ::SparseIterate)
    for i=1:length(out)
      s = createVarianceGrad1(θ, i)
      out[i] = variance(s, X, Y)
    end
    out
  end

  p = 10
  m = div(p*(p+1), 2)
  X, Y = generateData(p)
  n = size(X, 1)
  Sx = Symmetric( X'X / n )
  Sy = Symmetric( Y'Y / n )  
  out1 = zeros(m)
  out2 = zeros(m)

  θ = SparseIterate(m)
  DiffPrecTest._computeVarStep1!(out1, Sx, Sy, X, Y, θ);
  varStep1Alt!(out2, X, Y, θ);
  @test out1 ≈ sqrt.(out2)

  θ .= sprandn(m, 0.5)
  DiffPrecTest._computeVarStep1!(out1, Sx, Sy, X, Y, θ);
  varStep1Alt!(out2, X, Y, θ);
  @test out1 ≈ sqrt.(out2)
end


@testset "reducedDiffEstimation" begin  

  function reducedDiffEstimationAlt(H::Symmetric, b::Vector, Sx, Sy, X, Y, λ)
    x = SparseIterate(size(H, 1))
    
    ω = Array{eltype(H)}(undef, length(x))
    DiffPrecTest._computeVarStep1!(ω, Sx, Sy, X, Y, x)    

    xtmp = solve_pen_quad(H, b, λ, ω)
    for i=1:length(x)
      x[i] = xtmp[i]
    end

    DiffPrecTest._computeVarStep1!(ω, Sx, Sy, X, Y, x)
    solve_pen_quad(H, b, λ, ω)
  end

  p = 5
  m = div(p*(p+1), 2)
  X, Y = generateData(p)
  n = size(X, 1)
  Sx = Symmetric( X'X / n )
  Sy = Symmetric( Y'Y / n )
  H = Symmetric( skron(Sx, Sy) )
  b = svec(Sx) - svec(Sy)

  λ = DiffPrecTest._penaltyLevel(p)

  out1 = reducedDiffEstimationAlt(H, b, Sx, Sy, X, Y, λ)  
  out2 = DiffPrecTest.reducedDiffEstimation(H, b, Sx, Sy, X, Y, λ)
  
  @test out1 ≈ Vector(out2) atol=1e-4
end


@testset "_computeVarStep2!" begin
  function createVarianceGrad2(Δ::SparseIterate, rowIndex::Int, indElem::Int)
    @inline function h(x, y)::Float64
      # skron(vx_j, vy_k)[ri, :] * Δ
      out = 0.
      for inz = 1:nnz(Δ)
        indColumn = Δ.nzval2ind[inz]          
        out += skron(x, y, rowIndex, indColumn) * Δ.nzval[inz]
      end  
      out + (rowIndex == indElem ? 1. : 0.)
    end
    TwoSampleUstatistics{1, 1}(h)
  end

  function varStep2Alt!(out, indElem, X, Y, θ::SparseIterate)
    for i=1:length(out)
      s = createVarianceGrad2(θ, i, indElem)
      out[i] = variance(s, X, Y)
    end
    out
  end

  p = 10
  m = div(p*(p+1), 2)
  X, Y = generateData(p)
  n = size(X, 1)
  Sx = Symmetric( X'X / n )
  Sy = Symmetric( Y'Y / n )  
  out1 = zeros(m)
  out2 = zeros(m)

  for j=1:m
    θ = SparseIterate(m)
    DiffPrecTest._computeVarStep2!(out1, j, Sx, Sy, X, Y, θ);
    varStep2Alt!(out2, j, X, Y, θ);
    @test out1 ≈ sqrt.(out2)

    θ .= sprandn(m, 0.5)
    DiffPrecTest._computeVarStep2!(out1, j, Sx, Sy, X, Y, θ);
    varStep2Alt!(out2, j, X, Y, θ);
    @test out1 ≈ sqrt.(out2)
  end
end


@testset "invHessianReduced" begin
  function invHessianReducedAlt(H::Symmetric, indRow, Sx, Sy, X, Y, λ)
    x = SparseIterate(size(H, 1))
    x[indRow] = 1. / H[indRow, indRow]

    local b = zeros(size(H, 1))
    b[indRow] = -1.

    ω = Array{eltype(H)}(undef, length(x))
    DiffPrecTest._computeVarStep2!(ω, indRow, Sx, Sy, X, Y, x)    
    ω[indRow] = 0.

    xtmp = solve_pen_quad(H, b, λ, ω)
    for i=1:length(x)
      x[i] = xtmp[i]
    end

    DiffPrecTest._computeVarStep2!(ω, indRow, Sx, Sy, X, Y, x)
    ω[indRow] = 0.
    solve_pen_quad(H, b, λ, ω)
  end

  p = 5
  m = div(p*(p+1), 2)
  X, Y = generateData(p)
  n = size(X, 1)
  Sx = Symmetric( X'X / n )
  Sy = Symmetric( Y'Y / n )
  H = Symmetric( skron(Sx, Sy) )
  b = svec(Sx) - svec(Sy)

  λ = DiffPrecTest._penaltyLevel(p)

  for j=1:m
    out1 = invHessianReducedAlt(H, j, Sx, Sy, X, Y, λ)
    out2 = DiffPrecTest.invHessianReduced(H, j, Sx, Sy, X, Y, λ)

    @test out1 ≈ Vector(out2) atol=1e-4
  end

  # with a negligible penalty the estimates are the rows of inv(H) and svec(Δ)
  opt = CDSolver(; maxpasses=5000, tol=1e-12)
  for j in [1, 2, 7]
    out = DiffPrecTest.invHessianReduced(H, j, Sx, Sy, X, Y, 1e-10, opt)
    @test Vector(out) ≈ inv(H)[:, j] atol=1e-6
  end
  out = DiffPrecTest.reducedDiffEstimation(H, b, Sx, Sy, X, Y, 1e-10, opt)
  @test Vector(out) ≈ svec(Symmetric(inv(Sx) - inv(Sy))) atol=1e-6
end


@testset "varianceOracle" begin

  function createVarianceReducedEstim(ω, I, Δ, J)
    function h(x, y)::Float64
      dot( (skron(x, y, I, J) * Δ + svec(x, x, I) - svec(y, y, I)), ω )
    end
    TwoSampleUstatistics{1, 1}(h)
  end

  for rep=1:10
    p = 5
    m = div(p*(p+1), 2)
    X, Y = generateData(p)
    n = size(X, 1)
    Sx = Symmetric( X'X / n )
    Sy = Symmetric( Y'Y / n )

    indS = 3:9
    C = skron(Sx, Sy, indS)
    b = svec(Sy, indS) - svec(Sx, indS)
  
    tmp = zeros(length(indS))
    tmp[1] = 1.
    ω = C \ tmp
    Δ = C \ b
    v = variance(ReducedOracleNormal(), Sx, Sy, X, Y, ω, collect(indS), Δ, collect(indS))    

    v2 = variance(createVarianceReducedEstim(ω, indS, Δ, indS), X, Y)
    @test v ≈ v2
  end
end


end
