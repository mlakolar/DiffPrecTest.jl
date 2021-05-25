module ReducedTest

using Test
using Random, LinearAlgebra, Statistics, Distributions, SparseArrays
using Convex, SCS
using ProximalBase
using DiffPrecTest
using DiffPrecTest: TwoSampleUstatistics, svec, skron


function generateData(p)
  n = 100
  # generate model
  Ωx = Matrix{Float64}(I, p, p)
  for l=1:p-1
    Ωx[l  , l+1] = 0.6
    Ωx[l+1, l  ] = 0.6
  end
  for l=1:p-2
    Ωx[l  , l+2] = 0.3
    Ωx[l+2, l  ] = 0.3
  end
  Δ = zeros(p, p)
  # generate Delta
  for j=1:p
    Δ[j, j] = rand(Uniform(0.1, 0.2))
  end
  for j=1:p-1
    v = rand(Uniform(0.2, 0.5))
    Δ[j  , j+1] = v
    Δ[j+1, j  ] = v
  end
  Ωy = Ωx - Δ
  d = Vector{Float64}(undef, p)
  rand!(Uniform(0.5, 2.5), d)
  d .= sqrt.(d)
  D = Diagonal(d)
  Σx = inv(Symmetric(D * Ωx * D))
  Σy = inv(Symmetric(D * Ωy * D))

  dist_X = MvNormal(convert(Matrix, Σx))
  dist_Y = MvNormal(convert(Matrix, Σy))

  X = rand(dist_X, n)'
  Y = rand(dist_Y, n)'

  X, Y
end

function solve_pen_quad(A, b, λ, ω)
  p = size(A, 1)
  x = Variable(p)
  expr = 0.
  for i=1:p
    expr += abs(x[i]) * ω[i]
  end
  problem = minimize(quadform(x, A / 2.) + dot(x, b) + expr * λ)
  solve!(problem, () -> SCS.Optimizer(verbose = false))
  x.value
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

  λ = 1.01 * quantile( Normal(), 1. - 0.1 / (p*(p+1)) )

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
    x[indRow] = 1.
    
    b = zeros(size(H, 1))
    b[indRow] = 1.

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

  λ = 1.01 * quantile( Normal(), 1. - 0.1 / (p*(p+1)) )

  for j=1:m
    out1 = invHessianReducedAlt(H, j, Sx, Sy, X, Y, λ)  
    out2 = DiffPrecTest.invHessianReduced(H, j, Sx, Sy, X, Y, λ)
  
    @test out1 ≈ Vector(out2) atol=1e-4
  end
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
