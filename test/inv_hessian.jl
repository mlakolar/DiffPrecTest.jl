module InvHessian

using Test
using DiffPrecTest
using ProximalBase
using ProximalSolvers: CDSolver, solve!
using Random, LinearAlgebra, SparseArrays, Statistics

Random.seed!(4)

function gen_data(p; n = 1000)
  Sigmay = [0.7^abs(i-j) for i=1:p, j=1:p]
  X = randn(n,p)
  Y = randn(n,p) * sqrt(Sigmay)
  X, Y
end

const tight = CDSolver(; maxpasses=5000, tol=1e-12)

@testset "invert Kroneker" begin
  for rep=1:20
    p = 20
    X, Y = gen_data(p)
    hSx, hSy = cov(X), cov(Y)
    A = kron(hSy, hSx)

    λ = 0.05 + 0.25 * rand()
    g = NormL1(λ)

    for i = [(1,1), (3,1), (4,5)]
      ri = i[1]
      ci = i[2]
      x = SparseIterate(p*p)
      x1 = SparseIterate(p*p)

      f = DiffPrecTest._inverse_kronecker_loss(Symmetric(hSx), Symmetric(hSy), ri, ci)
      b = zeros(Float64, p*p)
      b[(ci-1)*p+ri] = -1.

      f1 = QuadraticFunction(A, b)
      solve!(x, f, g, tight)
      solve!(x1, f1, g, tight)

      @test convert(Vector, x) ≈ convert(Vector, x1) atol=1e-7
    end
  end
end

@testset "invert reduced SymKroneker" begin
  for rep=1:20
    p = 10
    X, Y = gen_data(p)
    hSx, hSy = Symmetric(cov(X)), Symmetric(cov(Y))
    H = DiffPrecTest.skron(hSx, hSy)
    m = size(H, 1)

    λ = 0.05 + 0.25 * rand()
    # unit weights on svec(Θ) are weights 1/√2 on each off-diagonal entry of Θ
    W = DiffPrecTest._sym_weights!(zeros(p, p), ones(m), p)

    for (ri, ci) in [(1,1), (3,1), (5,4)]
      ind = tril_index(p, ri, ci)
      x = SymmetricSparseIterate(p)
      x1 = SparseIterate(m)

      f = DiffPrecTest._inverse_sym_kronecker_loss(hSx, hSy, ind)
      b = zeros(m)
      b[ind] = -1.
      f1 = QuadraticFunction(H, b)

      solve!(x, f, NormL1(λ, W), tight)
      solve!(x1, f1, NormL1(λ), tight)

      @test convert(Vector, DiffPrecTest._svec(x)) ≈ convert(Vector, x1) atol=1e-7
    end
  end
end

@testset "gradient" begin
  p = 5
  X, Y = gen_data(p)
  hSx, hSy = Symmetric(cov(X)), Symmetric(cov(Y))

  # reduced symmetric Hessian: the gradient in svec(Θ) is Hθ - e_ind, which is
  # the gradient G over symmetric matrices with the off-diagonal entries scaled by √2
  H = DiffPrecTest.skron(hSx, hSy)
  m = size(H, 1)
  ind = tril_index(p, 3, 2)
  θ = sprandn(m, 0.5)
  x = SymmetricSparseIterate(p)
  for k in findall(!iszero, θ)
    r, c = tril_cartesian(p, k)
    x[r, c] = r == c ? θ[k] : θ[k] / sqrt(2)
  end
  @test Vector(DiffPrecTest._svec(x)) ≈ Vector(θ)
  G = gradient(DiffPrecTest._inverse_sym_kronecker_loss(hSx, hSy, ind), x)
  e = zeros(m)
  e[ind] = 1.
  @test [r == c ? G[r, c] : sqrt(2) * G[r, c] for c=1:p for r=c:p] ≈ H * Vector(θ) - e

  # asymmetric Hessian
  A = kron(hSy, hSx)
  x = SparseIterate(p*p)
  x .= sprandn(p*p, 0.5)
  f = DiffPrecTest._inverse_kronecker_loss(hSx, hSy, 3, 2)
  e = zeros(p*p)
  e[(2-1)*p + 3] = 1.
  @test vec(gradient(f, x)) ≈ A * Vector(x) - e
end

@testset "second stage without penalty" begin
  # with a negligible penalty the second stage estimates are the rows of the
  # inverse of the sample Hessian
  p = 5
  X, Y = gen_data(p; n = 500)
  n = size(X, 1)
  Sx = Symmetric( X'X / n )
  Sy = Symmetric( Y'Y / n )

  H = DiffPrecTest.skron(Sx, Sy)
  for (ri, ci) in [(1,1), (4,2)]
    ind = tril_index(p, ri, ci)
    ω = inv_q_sym_hessian(Sx, Sy, ind, X, Y, 1e-10, tight)
    @test Vector(ω) ≈ inv(H)[:, ind] atol=1e-6
  end

  A = kron(Sy, Sx)
  for (ri, ci) in [(1,1), (4,2), (2,4)]
    ω = inv_asym_hessian_estimation(Sx, Sy, ri, ci, X, Y, 1e-10, tight)
    @test Vector(ω) ≈ inv(A)[:, (ci-1)*p + ri] atol=1e-6
  end
end

@testset "stopping rule scales" begin
  # the stopping rule is relative to the scales of the loss: multiplying Σx and
  # Σy by a divides the solution by a² and takes the same number of passes
  # (powers of 2 keep the scaling exact)
  p = 5
  X, Y = gen_data(p)
  hSx, hSy = Symmetric(cov(X)), Symmetric(cov(Y))
  ind = tril_index(p, 3, 2)
  for (loss, x0) in (((Sx, Sy) -> DiffPrecTest._inverse_kronecker_loss(Sx, Sy, 3, 2), () -> SparseIterate(p*p)),
                     ((Sx, Sy) -> DiffPrecTest._inverse_sym_kronecker_loss(Sx, Sy, ind), () -> SymmetricSparseIterate(p)))
    passes, sols = Int[], Matrix{Float64}[]
    for a in (0.25, 1.0, 4.0)
      x = x0()
      res = solve!(x, loss(Symmetric(a * hSx), Symmetric(a * hSy)), NormL1(0.05), CDSolver(; order=:cyclic))
      @test res.converged
      push!(passes, res.passes)
      push!(sols, a^2 * reshape(collect(x), p, p))
    end
    @test passes[1] == passes[2] == passes[3]
    @test sols[1] ≈ sols[2]
    @test sols[3] ≈ sols[2]
  end
end

end
