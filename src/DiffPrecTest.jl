module DiffPrecTest

using Statistics, LinearAlgebra
using ProximalBase, CoordinateDescent, CovSel
using StatsBase, Distributions

export
  DiffPrecResultBoot,
  DiffPrecResultNormal,
  AsymmetricOracleBoot,
  SymmetricOracleBoot,
  AsymmetricOracleNormal,
  SymmetricOracleNormal,
  SymmetricNormal,
  estimate,

  # first and second stage functions
  diffEstimation, invHessianEstimation


include("variance.jl")
include("diffEstimation.jl")
include("invHessianEstimation.jl")



# compute kron(A, B)[ind, ind]
function kron_sub!(out, A, B, ind)
  @assert size(out, 1) == size(out, 2) == length(ind)

  m, n = size(A)
  p, q = size(B)
  for col=1:length(ind)
    j = ind[col]
    ac = div(j-1, q) + 1
    bc = mod(j-1, q) + 1

    for row=1:length(ind)
        i = ind[row]

        ar = div(i-1, p) + 1
        br = mod(i-1, p) + 1

        out[row, col] = A[ar, ac] * B[br, bc]
    end
  end
  out
end

### different solvers

abstract type DiffPrecMethod end
struct AsymmetricOracleBoot <: DiffPrecMethod end
struct SymmetricOracleBoot <: DiffPrecMethod end
struct AsymmetricOracleNormal <: DiffPrecMethod end
struct SymmetricOracleNormal <: DiffPrecMethod end
struct SymmetricNormal <: DiffPrecMethod end



###

struct DiffPrecResultBoot
  p::Float64
  boot_p::Vector{Float64}
end

struct DiffPrecResultNormal
  p::Float64
  std::Float64
end



###

# indS --- coordinates for the nonzero element of the true Δ (LinearIndices)
#          the first coordinate of indS is the one we make inference for
function estimate(::AsymmetricOracleBoot, X, Y, indS; bootSamples::Int64=1000)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = cov(X)
  Sy = cov(Y)

  A = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  Δab = (A \ (Sy[indS] - Sx[indS]))[1]

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = sample(1:nx, nx)
     bY_ind = sample(1:ny, ny)
     bSx = cov(X[bX_ind, :])
     bSy = cov(Y[bY_ind, :])

     kron_sub!(A, bSy, bSx, indS)
     boot_est[b] = (A \ (bSy[indS] - bSx[indS]))[1]
  end

  DiffPrecResultBoot(Δab, boot_est)
end


# indS --- coordinates for the nonzero element of the true Δ (LinearIndices)
function estimate(::SymmetricOracleBoot, X, Y, indS; bootSamples::Int64=1000)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = cov(X)
  Sy = cov(Y)

  A = zeros(length(indS), length(indS))
  B = zeros(length(indS), length(indS))
  C = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  kron_sub!(B, Sx, Sy, indS)
  @. C = (A + B) / 2.
  Δab = (C \ (Sy[indS] - Sx[indS]))[1]

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = sample(1:nx, nx)
     bY_ind = sample(1:ny, ny)
     bSx = cov(X[bX_ind, :])
     bSy = cov(Y[bY_ind, :])

     kron_sub!(A, bSy, bSx, indS)
     kron_sub!(B, bSx, bSy, indS)
     @. C = (A + B) / 2.

     boot_est[b] = (C \ (bSy[indS] - bSx[indS]))[1]
  end

  DiffPrecResultBoot(Δab, boot_est)
end


# computes variance of Var(Sx - Sy)[indS, indS]
# where indS is a list of elements in [1:p, 1:p]
function _varSxSy(Sx, nx, Sy, ny, indS)

    varS = zeros(length(indS), length(indS))   # var(Sx) + var(Sy)

    I = CartesianIndices(Sx)
    for ia = 1:length(indS)
      a = indS[ia]
      for ib = 1:length(indS)
        b = indS[ib]

        i, j = Tuple( I[a] )
        k, l = Tuple( I[b] )

        varS[ia, ib] = (Sx[i, k] * Sx[j, l] + Sx[i, l] * Sx[j, k]) / (nx - 1) + (Sy[i, k] * Sy[j, l] + Sy[i, l] * Sy[j, k]) / (ny - 1)
      end
    end

    varS
end

function estimate(
    ::AsymmetricOracleNormal,
    Sx::Symmetric, nx::Int,
    Sy::Symmetric, ny::Int,
    indS)

  A = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  Δab = (A \ (Sy[indS] - Sx[indS]))[1]
  varS  = _varSxSy(Sx, nx, Sy, ny, indS)
  v = ((A \ varS) / A)[1]

  DiffPrecResultNormal(Δab, sqrt(v))
end

function estimate(
    ::SymmetricOracleNormal,
    Sx::Symmetric, nx::Int,
    Sy::Symmetric, ny::Int,
    indS)

  A = zeros(length(indS), length(indS))
  B = zeros(length(indS), length(indS))
  C = zeros(length(indS), length(indS))
  kron_sub!(A, Sy, Sx, indS)
  kron_sub!(B, Sx, Sy, indS)
  @. C = (A + B) / 2.
  Δab = (C \ (Sy[indS] - Sx[indS]))[1]
  varS  = _varSxSy(Sx, nx, Sy, ny, indS)
  v = ((C \ varS) / C)[1]

  DiffPrecResultNormal(Δab, sqrt(v))
end


function estimate(::SymmetricNormal, X, Y, ind)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = Symmetric(cov(X))
  Sy = Symmetric(cov(Y))

  # first stage
  λ = 1.01 * quantile(Normal(), 1. - 0.1 / (px * (px+1)))
  x1 = diffEstimation(Sx, Sy, X, Y, λ)

  S = BitArray(undef, (px, px))
  fill!(S, false)
  for ci=1:px
      for ri=ci:px
          if abs(x1[ri, ci] > 1e-3)
              S[ri, ci] = true
              S[ci, ri] = true
          end
      end
  end

  # second stage
  I = CartesianIndices(Sx)
  ri, ci = Tuple( I[ind] )
  x2 = invHessianEstimation(Sx, Sy, ri, ci, X, Y, λ)
  for ci=1:px
      for ri=1:px
          if abs(x2[ri + (ci-1)*px] > 1e-3)
              S[ri, ci] = true
              S[ci, ri] = true
          end
      end
  end

  # refit stage
  indS = [LinearIndices((px, px))[x] for x in findall( S )]
  pos = findfirst(isequal(ind), indS)
  if  pos === nothing
      pushfirst!(indS, ind)
  else
      indS[pos], indS[1] = indS[1], indS[pos]
  end

  estimate(SymmetricOracleNormal(), Sx, nx, Sy, ny, indS)
end


end
