module InvHessian

using Test
using DiffPrecTest
using ProximalBase
using CovSel
using CoordinateDescent
using Random, LinearAlgebra, SparseArrays, Statistics, Distributions



function genData(p)
  Sigmax = Matrix(1.0I, p, p)
  Sigmay = zeros(p,p)
  rho = 0.7
  for i=1:p
    for j=1:p
      Sigmay[i,j]=rho^abs(i-j)
    end
  end
  sqmy = sqrt(Sigmay)
  n = 1000
  X = randn(n,p)
  Y = randn(n,p) * sqmy;
  hSx = cov(X)
  hSy = cov(Y)
  hSx, hSy
end

function Q(p::Int)
  _c = 1. / sqrt(2.)
  rp = div((p+1)*p,2)
  Q = spzeros( rp,  p*p )
  ind = 0
  for ci=1:p
    for ri=ci:p
      ind += 1
      if ri == ci
        Q[ind, (ci-1)*p + ri] = 1.
      else
        Q[ind, (ci-1)*p + ri] = _c
        Q[ind, (ri-1)*p + ci] = _c
      end
    end
  end
  Q
end

function Qt(p::Int)
  _c = 1. / sqrt(2.)
  rp = div((p+1)*p,2)
  Q = spzeros( p*p, rp )
  ind = 0
  for ci=1:p
    for ri=ci:p
      ind += 1
      if ri == ci
        Q[(ci-1)*p + ri, ind] = 1.
      else
        Q[(ci-1)*p + ri, ind] = _c
        Q[(ri-1)*p + ci, ind] = _c
      end
    end
  end
  Q
end


@testset "invert Kroneker" begin
  for rep=1:50
    p = 20
    hSx, hSy = genData(p)
    A = kron(hSy, hSx)

    λ = rand(Uniform(0.05, 0.3))
    g = ProximalBase.ProxL1(λ)

    for i = [(1,1), (3,1), (4,5)]
      ri = i[1]
      ci = i[2]
      x = ProximalBase.SparseIterate(p*p)
      x1 = ProximalBase.SparseIterate(p*p)

      f = DiffPrecTest.CDInverseKroneckerLoss(Symmetric(hSx), Symmetric(hSy), ri, ci)
      b = zeros(Float64, p*p)
      b[(ci-1)*p+ri] = -1.

      f1 = CoordinateDescent.CDQuadraticLoss(A, b)
      CoordinateDescent.coordinateDescent!(x, f, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))
      CoordinateDescent.coordinateDescent!(x1, f1, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))

      @test convert(Vector, x) ≈ convert(Vector, x1) atol=1e-7
    end
  end
end
#
# @testset "invert SymKroneker" begin
#   for rep=1:50
#     p = 20
#     hSx, hSy = genData(p)
#     A = (kron(hSy, hSx) + kron(hSx, hSy)) / 2.
#
#     λ = rand(Uniform(0.05, 0.3))
#     g = ProximalBase.ProxL1(λ)
#
#     for i = [(1,1), (3,1), (4,5)]
#       ri = i[1]
#       ci = i[2]
#       x = ProximalBase.SparseIterate(p*p)
#       x1 = ProximalBase.SparseIterate(p*p)
#
#       f = DiffPrecTest.CDInverseSymKroneckerLoss(Symmetric(hSx), Symmetric(hSy), ri, ci)
#       b = zeros(Float64, p*p)
#       b[(ci-1)*p+ri] = -1.
#
#       f1 = CoordinateDescent.CDQuadraticLoss(A, b)
#       CoordinateDescent.coordinateDescent!(x, f, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))
#       CoordinateDescent.coordinateDescent!(x1, f1, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))
#
#       @test convert(Vector, x) ≈ convert(Vector, x1) atol=1e-7
#     end
#   end
# end
# 
# @testset "invert reduced SymKroneker" begin
#   for rep=1:50
#     p = 20
#     hSx, hSy = genData(p)
#     A = (kron(hSy, hSx) + kron(hSx, hSy)) / 2.
#
#     λ = rand(Uniform(0.05, 0.3))
#     g = ProximalBase.ProxL1(λ)
#
#     for i = [(1,1), (3,1), (4,5)]
#       ri = i[1]
#       ci = i[2]
#       x = ProximalBase.SparseIterate(p*p)
#       x1 = ProximalBase.SparseIterate(p*p)
#
#       f = DiffPrecTest.CDInverseSymKroneckerLoss(Symmetric(hSx), Symmetric(hSy), ri, ci)
#       b = zeros(Float64, p*p)
#       b[(ci-1)*p+ri] = -1.
#
#       f1 = CoordinateDescent.CDQuadraticLoss(A, b)
#       CoordinateDescent.coordinateDescent!(x, f, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))
#       CoordinateDescent.coordinateDescent!(x1, f1, g, CoordinateDescent.CDOptions(;maxIter=5000, optTol=1e-12))
#
#       @test convert(Vector, x) ≈ convert(Vector, x1) atol=1e-7
#     end
#   end
# end



end
