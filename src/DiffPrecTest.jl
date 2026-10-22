"""
    DiffPrecTest

Inference for the difference of two precision matrices `Δ = Ωx - Ωy` from two
independent samples `X` (`nx × p`) and `Y` (`ny × p`):

* confidence intervals for a single entry of `Δ` ([`estimate`](@ref) with
  [`SymmetricNormal`](@ref), [`AsymmetricNormal`](@ref),
  [`ReducedNormal`](@ref), [`SeparateNormal`](@ref) and the oracle variants);
* simultaneous confidence intervals for all entries by the bootstrap
  ([`bootstrap`](@ref), [`simulCI`](@ref), [`simulCIstudentized`](@ref));
* support recovery ([`supportEstimate`](@ref)).
"""
module DiffPrecTest

using Statistics, LinearAlgebra, SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random
using SpecialFunctions: erfcinv

export
  DiffPrecResultBoot,
  DiffPrecResultNormal,
  SimulationResult,
  AsymmetricOracleBoot,
  SymmetricOracleBoot,
  AsymmetricOracleNormal,
  SymmetricOracleNormal,
  SymmetricNormal,
  AsymmetricNormal,
  SeparateNormal,
  DecorrelatedScoreNormal,
  DecorrelatedScoreOracleNormal,
  ReducedNormal,
  ReducedOracleNormal,
  estimate,
  compute,
  variance,
  computeSimulationResult,

  # first and second stage functions
  diffEstimation, invAsymHessianEstimation, invQSymHessian, invHessianReduced,

  # support estimation
  CDInverseKroneckerLoss, CDInverseReducedSymKroneckerLoss,
  ANTSupport,
  BootStdSupport,
  BootMaxSupport,
  DTraceValidationSupport,
  supportEstimate,

  # bootstrap
  BootstrapEstimates,
  bootstrap,
  simulCI, simulCIstudentized




####################################
#
#   different solvers
#
####################################

abstract type DiffPrecMethod end

"""
    AsymmetricOracleBoot()

Refit on a given support with the asymmetric Hessian `Σy ⊗ Σx`; bootstrap
standard error. See [`estimate`](@ref).
"""
struct AsymmetricOracleBoot <: DiffPrecMethod end

"""
    SymmetricOracleBoot()

Refit on a given support with the symmetrized Hessian `(Σx ⊗ Σy + Σy ⊗ Σx) / 2`;
bootstrap standard error. See [`estimate`](@ref).
"""
struct SymmetricOracleBoot <: DiffPrecMethod end

"""
    AsymmetricOracleNormal()

Refit on a given support with the asymmetric Hessian `Σy ⊗ Σx`; standard error
from the normal approximation. See [`estimate`](@ref).
"""
struct AsymmetricOracleNormal <: DiffPrecMethod end

"""
    SymmetricOracleNormal()

Refit on a given support with the symmetrized Hessian `(Σx ⊗ Σy + Σy ⊗ Σx) / 2`;
standard error from the normal approximation. See [`estimate`](@ref).
"""
struct SymmetricOracleNormal <: DiffPrecMethod end

"""
    AsymmetricNormal()

Like [`SymmetricNormal`](@ref), but the second stage estimates a row of the
inverse of the asymmetric Hessian `Σy ⊗ Σx` ([`invAsymHessianEstimation`](@ref))
and the refit uses [`AsymmetricOracleNormal`](@ref).
"""
struct AsymmetricNormal <: DiffPrecMethod end

"""
    SymmetricNormal()

The three stage procedure for one entry of `Δ`:

1. estimate `Δ` by the weighted ℓ1 penalized direct difference estimator
   ([`diffEstimation`](@ref));
2. estimate the row of the inverse of the Hessian `(Σx ⊗ Σy + Σy ⊗ Σx) / 2`
   that belongs to the entry ([`invQSymHessian`](@ref));
3. refit on the union of the two supports ([`SymmetricOracleNormal`](@ref)).

See [`estimate`](@ref).
"""
struct SymmetricNormal <: DiffPrecMethod end

"""
    SeparateNormal()

Estimate `Ωx` and `Ωy` separately by node-wise lasso regressions and combine
the bias corrected entries (Xia, Cai and Cai, 2015). See [`estimate`](@ref).
"""
struct SeparateNormal <: DiffPrecMethod end

struct OneStep <: DiffPrecMethod end             # not implemented yet
struct OneStepRefit <: DiffPrecMethod end        # not implemented yet

"""
    DecorrelatedScoreNormal()

Not implemented yet; see [`DecorrelatedScoreOracleNormal`](@ref).
"""
struct DecorrelatedScoreNormal <: DiffPrecMethod end

"""
    DecorrelatedScoreOracleNormal()

Decorrelated score estimator for given estimates of `Δ` and of a row of the
inverse Hessian. See [`estimate`](@ref).
"""
struct DecorrelatedScoreOracleNormal <: DiffPrecMethod end

"""
    ReducedNormal()

The procedure of [`SymmetricNormal`](@ref) in the half-vectorized
parametrization `svec(Δ)` of a symmetric `Δ` ([`reducedDiffEstimation`](@ref),
[`invHessianReduced`](@ref), [`ReducedOracleNormal`](@ref)). See
[`estimate`](@ref).
"""
struct ReducedNormal <: DiffPrecMethod end

"""
    ReducedOracleNormal()

Refit on a given support in the half-vectorized parametrization `svec(Δ)`;
standard error from the normal approximation. See [`estimate`](@ref).
"""
struct ReducedOracleNormal <: DiffPrecMethod end

####################################
#
#   different outputs
#
####################################

"""
    DiffPrecResultBoot(p, boot_p)

Point estimate `p` of an entry of `Δ` and its bootstrap replicates `boot_p`.
"""
struct DiffPrecResultBoot
  p::Float64
  boot_p::Vector{Float64}
end

"""
    DiffPrecResultNormal(p, std)

Point estimate `p` of an entry of `Δ` and its standard error `std`.
"""
struct DiffPrecResultNormal
  p::Float64
  std::Float64
end

"""
    SimulationResult

Summary of repeated estimates of one entry of `Δ` returned by
[`computeSimulationResult`](@ref): `bias`, `coverage` and average length
`lenCoverage` of the confidence intervals, and the mean `p_mean` and variance
`p_var` of the estimates.
"""
struct SimulationResult
  bias::Float64
  coverage::Float64
  lenCoverage::Float64
  p_mean::Float64
  p_var::Float64
end




########################################

include("skron.jl")
include("ustats.jl")
include("support.jl")
include("variance.jl")
include("diffEstimation.jl")
include("invHessianEstimation.jl")
include("util.jl")
include("bootstrap.jl")
include("reducedDiffEstim.jl")


##############################
#
#   our procedure
#
##############################

"""
    estimate(method, X, Y, row, col; Sx, Sy, Δ, suppΔ, ω, suppω)

Estimate the entry `Δ[row, col]` of `Δ = Ωx - Ωy` from the samples `X` and `Y`
with `method` one of [`SymmetricNormal`](@ref) or [`AsymmetricNormal`](@ref).
Returns `(res, Δ, suppΔ, ω, suppω, indS)`: the [`DiffPrecResultNormal`](@ref)
`res`, the first stage estimate `Δ` and its support `suppΔ`, the second stage
estimate `ω` and its support `suppω` (both supports as `p × p` `BitMatrix`),
and the linear indices `indS` of the refitted entries.

The sample covariances `Sx = X'X/nx`, `Sy = Y'Y/ny` and the first and second
stage estimates are computed unless given; when estimating several entries,
pass the first stage results of the first call to the later ones.

    estimate(method, X, Y, indElem; Sx, Sy, H, Δ, suppΔ, ω, suppω)

The same with `method = ReducedNormal()` for the entry with index `indElem` in
the lower triangle (see `ProximalBase.sub2indLowerTriangular`). Here `Δ` and `ω`
are vectors of length `p(p+1)/2` with supports `suppΔ` and `suppω` given as
`BitVector`s.

    estimate(SeparateNormal(), X, Y, row, col; Sx, Sy)

Estimator of Xia, Cai and Cai (2015). Returns a [`DiffPrecResultNormal`](@ref).

    estimate(method, Sx, Sy, X, Y, row, col, indS)

Refit on the support `indS`, given as linear indices into a `p × p` matrix,
with `method` one of [`SymmetricOracleNormal`](@ref),
[`AsymmetricOracleNormal`](@ref) (returning a [`DiffPrecResultNormal`](@ref)),
[`SymmetricOracleBoot`](@ref) or [`AsymmetricOracleBoot`](@ref) (returning a
[`DiffPrecResultBoot`](@ref); keyword arguments `bootSamples=1000` and
`rng=Random.default_rng()`). The entries `(row, col)` and `(col, row)` are moved
to the front of `indS`, which is modified.

    estimate(ReducedOracleNormal(), Sx, Sy, X, Y, indElem, indS)

Refit on the support `indS`, given as indices into `svec(Δ)`.

    estimate(DecorrelatedScoreOracleNormal(), Sx, Sy, X, Y, row, col, ω, indω, Δn, indΔn)

Decorrelated score estimator of `Δ[row, col]` given the values `ω` (on the
linear indices `indω`) of a row of the inverse of the Hessian
`(Σx ⊗ Σy + Σy ⊗ Σx) / 2` and the values `Δn` (on the linear indices `indΔn`) of
the remaining entries of `Δ`, which do not include `(row, col)` or `(col, row)`.

!!! note
    `Distributions` also exports a function named `estimate`; when both packages
    are loaded, call this one as `DiffPrecTest.estimate`.
"""
function estimate(::SymmetricNormal, X, Y, row, col;
    Sx::Union{Symmetric, Nothing}=nothing,
    Sy::Union{Symmetric, Nothing}=nothing,
    Δ::Union{SymmetricSparseIterate, Nothing}=nothing,
    suppΔ::Union{BitArray{2}, Nothing}=nothing,
    ω::Union{SparseIterate, Nothing}=nothing,
    suppω::Union{BitArray{2}, Nothing}=nothing
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  # the second stage works with the lower triangle
  if row < col
    row, col = col, row
  end

  if Sx === nothing
      Sx = Symmetric( X'X / nx )
  end
  if Sy === nothing
      Sy = Symmetric( Y'Y / ny )
  end

  # first stage
  λ = _penaltyLevel(p)
  if Δ === nothing && suppΔ === nothing
    Δ = diffEstimation(Sx, Sy, X, Y, λ)
    suppΔ = getSupport(Δ)
  end
  if suppΔ === nothing
    suppΔ = getSupport(Δ)
  end

  # second stage
  if ω === nothing && suppω === nothing
    j = sub2indLowerTriangular(p, row, col)
    ω = invQSymHessian(Sx, Sy, j, X, Y, λ)
    suppω = getSupport(ω, p)
  end
  if suppω === nothing
    suppω = getSupport(ω, p)
  end

  # refit stage
  indS = getLinearSupport(row, col, suppΔ, suppω)
  estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, indS), Δ, suppΔ, ω, suppω, indS
end



####################################
#
#   competing procedures
#
####################################


function estimate(::AsymmetricNormal, X, Y, row, col;
    Sx::Union{Symmetric, Nothing}=nothing,
    Sy::Union{Symmetric, Nothing}=nothing,
    Δ::Union{SymmetricSparseIterate, Nothing}=nothing,
    suppΔ::Union{BitArray{2}, Nothing}=nothing,
    ω::Union{SparseIterate, Nothing}=nothing,
    suppω::Union{BitArray{2}, Nothing}=nothing
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  if Sx === nothing
      Sx = Symmetric( X'X / nx )
  end
  if Sy === nothing
      Sy = Symmetric( Y'Y / ny)
  end

  # first stage
  λ = _penaltyLevel(p)
  if Δ === nothing && suppΔ === nothing
      Δ = diffEstimation(Sx, Sy, X, Y, λ)
      suppΔ = getSupport(Δ)
  end
  if suppΔ === nothing
      suppΔ = getSupport(Δ)
  end

  # second stage
  if ω === nothing && suppω === nothing
      ω = invAsymHessianEstimation(Sx, Sy, row, col, X, Y, λ)
      suppω = getSupport(ω, p)
  end
  if suppω === nothing
      suppω = getSupport(ω, p)
  end

  # refit stage
  indS = getLinearSupport(row, col, suppΔ, suppω)
  estimate(AsymmetricOracleNormal(), Sx, Sy, X, Y, row, col, indS), Δ, suppΔ, ω, suppω, indS
end


function estimate(
    ::SeparateNormal,
    X,
    Y,
    ri,
    ci;
    Sx::Union{Symmetric, Nothing}=nothing,
    Sy::Union{Symmetric, Nothing}=nothing
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  if Sx === nothing
      Sx = Symmetric( X'X / nx )
  end
  if Sy === nothing
      Sy = Symmetric( Y'Y / ny)
  end

  if ri == ci
      lasso_x_r = lasso(X[:, 1:p .!= ri], X[:, ri], 2. * sqrt(Sx[ri, ri] * log(p) / nx))
      lasso_y_r = lasso(Y[:, 1:p .!= ri], Y[:, ri], 2. * sqrt(Sy[ri, ri] * log(p) / ny))

      T1 = 1. / lasso_x_r.σ^2.
      T2 = 1. / lasso_y_r.σ^2.

      Δab = T1 - T2

      v1 = 2. / (nx * lasso_x_r.σ^4.)
      v2 = 2. / (ny * lasso_y_r.σ^4.)

      return DiffPrecResultNormal(Δab, sqrt(v1 + v2))
  else
      if ri > ci
          ri, ci = ci, ri
      end

      lasso_x_r = lasso(X[:, 1:p .!= ri], X[:, ri], 2. * sqrt(Sx[ri, ri] * log(p) / nx))
      lasso_x_c = lasso(X[:, 1:p .!= ci], X[:, ci], 2. * sqrt(Sx[ci, ci] * log(p) / nx))
      lasso_y_r = lasso(Y[:, 1:p .!= ri], Y[:, ri], 2. * sqrt(Sy[ri, ri] * log(p) / ny))
      lasso_y_c = lasso(Y[:, 1:p .!= ci], Y[:, ci], 2. * sqrt(Sy[ci, ci] * log(p) / ny))

      T1 = dot(lasso_x_r.residuals, lasso_x_c.residuals) / nx + lasso_x_r.σ^2. * lasso_x_c.x[ri] + lasso_x_c.σ^2. * lasso_x_r.x[ci-1]
      T2 = dot(lasso_y_r.residuals, lasso_y_c.residuals) / ny + lasso_y_r.σ^2. * lasso_y_c.x[ri] + lasso_y_c.σ^2. * lasso_y_r.x[ci-1]

      T1 = T1 / ( lasso_x_r.σ^2. * lasso_x_c.σ^2.)
      T2 = T2 / ( lasso_y_r.σ^2. * lasso_y_c.σ^2.)

      Δab = T2 - T1

      v1 = (1. + lasso_x_c.x[ri]^2. * lasso_x_r.σ^2. / lasso_x_c.σ^2.) / (nx * lasso_x_r.σ^2. * lasso_x_c.σ^2.)
      v2 = (1. + lasso_y_c.x[ri]^2. * lasso_y_r.σ^2. / lasso_y_c.σ^2.) / (ny * lasso_y_r.σ^2. * lasso_y_c.σ^2.)

      return DiffPrecResultNormal(Δab, sqrt(v1 + v2))
  end
end

####################################
#
#   oracle procedures
#
####################################


# indS --- coordinates for the nonzero element of the true Δ (LinearIndices)
function estimate(::AsymmetricOracleBoot,
    Sx::Symmetric,
    Sy::Symmetric,
    X,
    Y,
    row,
    col,
    indS::Vector{Int64};
    bootSamples::Int64=1000,
    rng::AbstractRNG=Random.default_rng()
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  makeFirst!(indS, p, row, col)

  A = kron_sub(Sy, Sx, indS, indS)
  Δ = A \ (Sy[indS] - Sx[indS])
  Δab = row == col ? Δ[1] : ( Δ[1] + Δ[2] ) / 2.

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = rand(rng, 1:nx, nx)
     bY_ind = rand(rng, 1:ny, ny)
     bSx = cov(X[bX_ind, :])
     bSy = cov(Y[bY_ind, :])
     kron_sub!(A, bSy, bSx, indS, indS)
     Δb = A \ (bSy[indS] - bSx[indS])
     boot_est[b] = row == col ? Δb[1] : ( Δb[1] + Δb[2] ) / 2.
  end

  DiffPrecResultBoot(Δab, boot_est)
end


# indS --- coordinates for the nonzero element of the true Δ (LinearIndices)
function estimate(::SymmetricOracleBoot,
    Sx::Symmetric,
    Sy::Symmetric,
    X,
    Y,
    row,
    col,
    indS::Vector{Int64};
    bootSamples::Int64=1000,
    rng::AbstractRNG=Random.default_rng()
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  makeFirst!(indS, p, row, col)

  C = skron_sub(Sx, Sy, indS, indS)
  Δab = (C \ (Sy[indS] - Sx[indS]))[1]

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = rand(rng, 1:nx, nx)
     bY_ind = rand(rng, 1:ny, ny)
     bSx = cov(X[bX_ind, :])
     bSy = cov(Y[bY_ind, :])
     skron_sub!(C, bSx, bSy, indS, indS)
     boot_est[b] = (C \ (bSy[indS] - bSx[indS]))[1]
  end

  DiffPrecResultBoot(Δab, boot_est)
end





function estimate(
    ::AsymmetricOracleNormal,
    Sx::Symmetric,
    Sy::Symmetric,
    X,
    Y,
    row,
    col,
    indS::Vector{Int64}
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  makeFirst!(indS, p, row, col)

  A = kron_sub(Sy, Sx, indS, indS)
  Δ = A \ (Sy[indS] - Sx[indS])

  tmp = zeros(length(indS))
  tmp[1] = 1.
  ω = A \ tmp

  v = variance(AsymmetricOracleNormal(), Sx, Sy, X, Y, ω, indS, Δ, indS)
  DiffPrecResultNormal(Δ[1], sqrt(v))
end






function estimate(
    ::SymmetricOracleNormal,
    Sx::Symmetric,
    Sy::Symmetric,
    X,
    Y,
    row,
    col,
    indS::Vector{Int64}
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  makeFirst!(indS, p, row, col)
  C = skron_sub(Sx, Sy, indS, indS)

  tmp = zeros(length(indS))
  tmp[1] = 1.
  ω = C \ tmp
  Δ = C \ (Sy[indS] - Sx[indS])
  v = variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, indS, Δ, indS)

  DiffPrecResultNormal(Δ[1], sqrt(v))
end



# indΔn does not contain coordinates we are doing inference about
function estimate(
    ::DecorrelatedScoreOracleNormal,
    Sx::Symmetric,
    Sy::Symmetric,
    X,
    Y,
    row,
    col,
    ω::Vector{Float64},
    indω::Vector{Int64},
    Δn::Vector{Float64},
    indΔn::Vector{Int64}
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  v = 0.
  for ci=1:length(indΔn)
      for ri=1:length(indω)
          @inbounds v += _getElemSKron(Sx, Sy, indω[ri], indΔn[ci]) * ω[ri] * Δn[ci]
      end
  end
  for ri=1:length(indω)
      @inbounds v -= ω[ri] * (Sy[indω[ri]] - Sx[indω[ri]])
  end
  a = 0.
  if row == col
      ind = (col-1)*p + row
      for ri=1:length(indω)
          @inbounds a += ω[ri] * _getElemSKron(Sx, Sy, indω[ri], ind)
      end
  else
      ind1 = (col-1)*p + row
      ind2 = (row-1)*p + col
      for ri=1:length(indω)
          @inbounds a += ω[ri] * (_getElemSKron(Sx, Sy, indω[ri], ind1) + _getElemSKron(Sx, Sy, indω[ri], ind2))
      end
  end
  Δab = -v / a

  # variance of the score at the estimate, with Δab filled in for (row, col)
  if row == col
      indΔ = [(col-1)*p + row; indΔn]
      Δ = [Δab; Δn]
  else
      indΔ = [(col-1)*p + row; (row-1)*p + col; indΔn]
      Δ = [Δab; Δab; Δn]
  end
  v = variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, indω, Δ, indΔ) / a^2

  DiffPrecResultNormal(Δab, sqrt(v))
end








end
