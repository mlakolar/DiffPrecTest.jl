module InvHessian

using Test
using DiffPrecTest
using ProximalBase
using CoordinateDescent
using Random, LinearAlgebra, SparseArrays, Statistics

Random.seed!(4)

function genData(p; n = 1000)
  Sigmay = [0.7^abs(i-j) for i=1:p, j=1:p]
  X = randn(n,p)
  Y = randn(n,p) * sqrt(Sigmay)
  X, Y
end


@testset "invert Kroneker" begin
  for rep=1:20
    p = 20
    X, Y = genData(p)
    hSx, hSy = cov(X), cov(Y)
    A = kron(hSy, hSx)

    λ = 0.05 + 0.25 * rand()
    g = ProxL1(λ)

    for i = [(1,1), (3,1), (4,5)]
      ri = i[1]
      ci = i[2]
      x = SparseIterate(p*p)
      x1 = SparseIterate(p*p)

      f = CDInverseKroneckerLoss(Symmetric(hSx), Symmetric(hSy), ri, ci)
      b = zeros(Float64, p*p)
      b[(ci-1)*p+ri] = -1.

      f1 = CDQuadraticLoss(A, b)
      coordinateDescent!(x, f, g, CDOptions(;maxIter=5000, optTol=1e-12))
      coordinateDescent!(x1, f1, g, CDOptions(;maxIter=5000, optTol=1e-12))

      @test convert(Vector, x) ≈ convert(Vector, x1) atol=1e-7
    end
  end
end

@testset "invert reduced SymKroneker" begin
  for rep=1:20
    p = 10
    X, Y = genData(p)
    hSx, hSy = Symmetric(cov(X)), Symmetric(cov(Y))
    H = DiffPrecTest.skron(hSx, hSy)
    m = size(H, 1)

    λ = 0.05 + 0.25 * rand()
    g = ProxL1(λ)

    for (ri, ci) in [(1,1), (3,1), (5,4)]
      ind = sub2indLowerTriangular(p, ri, ci)
      x = SparseIterate(m)
      x1 = SparseIterate(m)

      f = CDInverseReducedSymKroneckerLoss(hSx, hSy, ind)
      b = zeros(m)
      b[ind] = -1.
      f1 = CDQuadraticLoss(H, b)

      coordinateDescent!(x, f, g, CDOptions(;maxIter=5000, optTol=1e-12))
      coordinateDescent!(x1, f1, g, CDOptions(;maxIter=5000, optTol=1e-12))

      @test convert(Vector, x) ≈ convert(Vector, x1) atol=1e-7
    end
  end
end

@testset "gradient" begin
  p = 5
  X, Y = genData(p)
  hSx, hSy = Symmetric(cov(X)), Symmetric(cov(Y))

  # reduced symmetric Hessian
  H = DiffPrecTest.skron(hSx, hSy)
  m = size(H, 1)
  ind = sub2indLowerTriangular(p, 3, 2)
  x = SparseIterate(m)
  x .= sprandn(m, 0.5)
  f = CDInverseReducedSymKroneckerLoss(hSx, hSy, ind)
  CoordinateDescent.initialize!(f, x)
  e = zeros(m)
  e[ind] = 1.
  @test [CoordinateDescent.gradient(f, x, j) for j=1:m] ≈ H * Vector(x) - e

  # asymmetric Hessian
  A = kron(hSy, hSx)
  x = SparseIterate(p*p)
  x .= sprandn(p*p, 0.5)
  f = CDInverseKroneckerLoss(hSx, hSy, 3, 2)
  CoordinateDescent.initialize!(f, x)
  e = zeros(p*p)
  e[(2-1)*p + 3] = 1.
  @test [CoordinateDescent.gradient(f, x, j) for j=1:p*p] ≈ A * Vector(x) - e
end

@testset "second stage without penalty" begin
  # with a negligible penalty the second stage estimates are the rows of the
  # inverse of the sample Hessian
  p = 5
  X, Y = genData(p; n = 500)
  n = size(X, 1)
  Sx = Symmetric( X'X / n )
  Sy = Symmetric( Y'Y / n )
  opt = CDOptions(; maxIter=5000, optTol=1e-12)

  H = DiffPrecTest.skron(Sx, Sy)
  for (ri, ci) in [(1,1), (4,2)]
    ind = sub2indLowerTriangular(p, ri, ci)
    ω = invQSymHessian(Sx, Sy, ind, X, Y, 1e-10, opt)
    @test Vector(ω) ≈ inv(H)[:, ind] atol=1e-6
  end

  A = kron(Sy, Sx)
  for (ri, ci) in [(1,1), (4,2), (2,4)]
    ω = invAsymHessianEstimation(Sx, Sy, ri, ci, X, Y, 1e-10, opt)
    @test Vector(ω) ≈ inv(A)[:, (ci-1)*p + ri] atol=1e-6
  end
end

@testset "warm start options" begin
  opt = CDOptions(; maxIter=7, optTol=1e-3, randomize=false, warmStart=false, numSteps=3)
  w = DiffPrecTest._warmStart(opt)
  @test w.warmStart
  for f in fieldnames(CDOptions)
    f === :warmStart || @test getfield(w, f) == getfield(opt, f)
  end
  if hasfield(CDOptions, :relativeTol)   # CoordinateDescent 0.5 and later
    @test !DiffPrecTest._warmStart(CDOptions(; relativeTol=false)).relativeTol
  end
end

# CoordinateDescent 0.5 and later measure convergence relative to scales of the loss
if isdefined(CoordinateDescent, :coordinateScale)
  @testset "stopping rule scales" begin
    p = 5
    X, Y = genData(p)
    hSx, hSy = Symmetric(cov(X)), Symmetric(cov(Y))

    # the scales are those of the quadratic loss with the same Hessian and linear term
    A = kron(hSy, hSx)
    t = (2-1)*p + 3
    b = zeros(p*p)
    b[t] = -1.
    f = CDInverseKroneckerLoss(hSx, hSy, 3, 2)
    fq = CDQuadraticLoss(A, b)
    @test [CoordinateDescent.coordinateScale(f, k) for k=1:p*p] ≈ sqrt.(diag(A))
    @test [CoordinateDescent.coordinateScale(f, k) for k=1:p*p] ≈ [CoordinateDescent.coordinateScale(fq, k) for k=1:p*p]
    @test CoordinateDescent.responseScale(f) ≈ CoordinateDescent.responseScale(fq) ≈ 1 / sqrt(A[t, t])

    H = DiffPrecTest.skron(hSx, hSy)
    m = size(H, 1)
    ind = sub2indLowerTriangular(p, 3, 2)
    b = zeros(m)
    b[ind] = -1.
    f = CDInverseReducedSymKroneckerLoss(hSx, hSy, ind)
    fq = CDQuadraticLoss(H, b)
    @test [CoordinateDescent.coordinateScale(f, k) for k=1:m] ≈ sqrt.(diag(H))
    @test [CoordinateDescent.coordinateScale(f, k) for k=1:m] ≈ [CoordinateDescent.coordinateScale(fq, k) for k=1:m]
    @test CoordinateDescent.responseScale(f) ≈ CoordinateDescent.responseScale(fq) ≈ 1 / sqrt(H[ind, ind])

    # so multiplying Σx and Σy by a divides the solution by a² and takes the same
    # number of passes (powers of 2 keep the scaling exact)
    for (loss, m) in (((Sx, Sy) -> CDInverseKroneckerLoss(Sx, Sy, 3, 2), p*p),
                      ((Sx, Sy) -> CDInverseReducedSymKroneckerLoss(Sx, Sy, ind), m))
      passes, sols = Int[], Vector{Float64}[]
      for a in (0.25, 1.0, 4.0)
        info = CoordinateDescent.CDInfo()
        x = SparseIterate(m)
        coordinateDescent!(x, loss(Symmetric(a * hSx), Symmetric(a * hSy)), ProxL1(0.05), CDOptions(; randomize=false), info)
        @test info.converged
        push!(passes, info.iterations)
        push!(sols, a^2 * Vector(x))
      end
      @test passes[1] == passes[2] == passes[3]
      @test sols[1] ≈ sols[2]
      @test sols[3] ≈ sols[2]
    end
  end
end

end
