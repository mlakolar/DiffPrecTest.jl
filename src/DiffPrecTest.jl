"""
    DiffPrecTest

Inference for the difference of two precision matrices `Δ = Ωx - Ωy` from two
independent samples `X` (`nx × p`) and `Y` (`ny × p`):

* confidence intervals for a single entry of `Δ` ([`estimate`](@ref) with
  [`SymmetricNormal`](@ref), [`AsymmetricNormal`](@ref),
  [`ReducedNormal`](@ref), [`SeparateNormal`](@ref) and the oracle variants);
* simultaneous confidence intervals for all entries by the bootstrap
  ([`bootstrap`](@ref), [`simul_ci`](@ref), [`simul_ci_studentized`](@ref));
* support recovery ([`support_estimate`](@ref)).
"""
module DiffPrecTest

using Statistics, LinearAlgebra, SparseArrays
using ProximalBase, CovarianceSelection
using ProximalSolvers: ProximalSolvers, CDSolver, cdmodel, solve!, solve_path
using Random
using SpecialFunctions: erfcinv

# Canonical and deprecated keywords share one implementation; reject an
# ambiguous call instead of silently choosing one spelling.
function _deprecated_keyword(value, legacy, canonical::Symbol, old::Symbol, default)
  if legacy !== nothing
    value === nothing || throw(ArgumentError("pass $canonical or $old, not both"))
    Base.depwarn("keyword $old is deprecated; use $canonical.", old)
    return legacy
  end
  value === nothing ? default : value
end


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
  compute_simulation_result,

  # first and second stage functions
  diff_estimation, inv_asym_hessian_estimation, inv_q_sym_hessian, inv_hessian_reduced,

  # support estimation
  ANTSupport,
  BootStdSupport,
  BootMaxSupport,
  DTraceValidationSupport,
  support_estimate,

  # bootstrap
  BootstrapEstimates,
  bootstrap,
  simul_ci, simul_ci_studentized




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
inverse of the asymmetric Hessian `Σy ⊗ Σx` ([`inv_asym_hessian_estimation`](@ref))
and the refit uses [`AsymmetricOracleNormal`](@ref).
"""
struct AsymmetricNormal <: DiffPrecMethod end

"""
    SymmetricNormal()

The three stage procedure for one entry of `Δ`:

1. estimate `Δ` by the weighted ℓ1 penalized direct difference estimator
   ([`diff_estimation`](@ref));
2. estimate the row of the inverse of the Hessian `(Σx ⊗ Σy + Σy ⊗ Σx) / 2`
   that belongs to the entry ([`inv_q_sym_hessian`](@ref));
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
parametrization `svec(Δ)` of a symmetric `Δ` ([`reduced_diff_estimation`](@ref),
[`inv_hessian_reduced`](@ref), [`ReducedOracleNormal`](@ref)). See
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
[`compute_simulation_result`](@ref): `bias`, `coverage` and average length
`len_coverage` of the confidence intervals, and the mean `p_mean` and variance
`p_var` of the estimates.
"""
struct SimulationResult
  bias::Float64
  coverage::Float64
  len_coverage::Float64
  p_mean::Float64
  p_var::Float64
end




########################################

include("skron.jl")
include("ustats.jl")
include("support.jl")
include("variance.jl")
include("diff_estimation.jl")
include("inv_hessian_estimation.jl")
include("util.jl")
include("bootstrap.jl")
include("reduced_diff_estim.jl")


##############################
#
#   our procedure
#
##############################

"""
    estimate(method, X, Y, row, col; Sx, Sy, Δ, suppΔ, ω, suppω)

Estimate the entry `Δ[row, col]` of `Δ = Ωx - Ωy` from the samples `X` and `Y`
with `method` one of [`SymmetricNormal`](@ref) or [`AsymmetricNormal`](@ref).
Returns `(res, Δ, suppΔ, ω, suppω, ind_s)`: the [`DiffPrecResultNormal`](@ref)
`res`, the first stage estimate `Δ` and its support `suppΔ`, the second stage
estimate `ω` and its support `suppω` (both supports as `p × p` `BitMatrix`),
and the linear indices `ind_s` of the refitted entries.

The sample covariances `Sx = X'X/nx`, `Sy = Y'Y/ny` and the first and second
stage estimates are computed unless given; when estimating several entries,
pass the first stage results of the first call to the later ones.

    estimate(method, X, Y, ind_elem; Sx, Sy, H, Δ, suppΔ, ω, suppω)

The same with `method = ReducedNormal()` for the entry with index `ind_elem` in
the lower triangle (see `ProximalBase.tril_index`). Here `Δ` and `ω`
are vectors of length `p(p+1)/2` with supports `suppΔ` and `suppω` given as
`BitVector`s.

    estimate(SeparateNormal(), X, Y, row, col; Sx, Sy)

Estimator of Xia, Cai and Cai (2015). Returns a [`DiffPrecResultNormal`](@ref).

    estimate(method, Sx, Sy, X, Y, row, col, ind_s)

Refit on the support `ind_s`, given as linear indices into a `p × p` matrix,
with `method` one of [`SymmetricOracleNormal`](@ref),
[`AsymmetricOracleNormal`](@ref) (returning a [`DiffPrecResultNormal`](@ref)),
[`SymmetricOracleBoot`](@ref) or [`AsymmetricOracleBoot`](@ref) (returning a
[`DiffPrecResultBoot`](@ref); keyword arguments `boot_samples=1000` and
`rng=Random.default_rng()`). The entries `(row, col)` and `(col, row)` are moved
to the front of `ind_s`, which is modified.

    estimate(ReducedOracleNormal(), Sx, Sy, X, Y, ind_elem, ind_s)

Refit on the support `ind_s`, given as indices into `svec(Δ)`.

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
  λ = _penalty_level(p)
  if Δ === nothing && suppΔ === nothing
    Δ = diff_estimation(Sx, Sy, X, Y, λ)
    suppΔ = get_support(Δ)
  end
  if suppΔ === nothing
    suppΔ = get_support(Δ)
  end

  # second stage
  if ω === nothing && suppω === nothing
    j = tril_index(p, row, col)
    ω = inv_q_sym_hessian(Sx, Sy, j, X, Y, λ)
    suppω = get_support(ω, p)
  end
  if suppω === nothing
    suppω = get_support(ω, p)
  end

  # refit stage
  ind_s = get_linear_support(row, col, suppΔ, suppω)
  estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, ind_s), Δ, suppΔ, ω, suppω, ind_s
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
  λ = _penalty_level(p)
  if Δ === nothing && suppΔ === nothing
      Δ = diff_estimation(Sx, Sy, X, Y, λ)
      suppΔ = get_support(Δ)
  end
  if suppΔ === nothing
      suppΔ = get_support(Δ)
  end

  # second stage
  if ω === nothing && suppω === nothing
      ω = inv_asym_hessian_estimation(Sx, Sy, row, col, X, Y, λ)
      suppω = get_support(ω, p)
  end
  if suppω === nothing
      suppω = get_support(ω, p)
  end

  # refit stage
  ind_s = get_linear_support(row, col, suppΔ, suppω)
  estimate(AsymmetricOracleNormal(), Sx, Sy, X, Y, row, col, ind_s), Δ, suppΔ, ω, suppω, ind_s
end


# The lasso ‖y - Xβ‖²/(2n) + λ‖β‖₁ of a node-wise regression, by ProximalSolvers'
# coordinate descent: the coefficients x, the residuals, and their standard
# deviation σ
function _lasso(X, y, λ)
  n = size(X, 1)
  x = SparseIterate(size(X, 2))
  solve!(x, (1 / n) * LeastSquaresLoss(y, X), NormL1(λ), CDSolver())
  r = y - X * x
  (x = x, residuals = r, σ = std(r))
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
      lasso_x_r = _lasso(X[:, 1:p .!= ri], X[:, ri], 2. * sqrt(Sx[ri, ri] * log(p) / nx))
      lasso_y_r = _lasso(Y[:, 1:p .!= ri], Y[:, ri], 2. * sqrt(Sy[ri, ri] * log(p) / ny))

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

      lasso_x_r = _lasso(X[:, 1:p .!= ri], X[:, ri], 2. * sqrt(Sx[ri, ri] * log(p) / nx))
      lasso_x_c = _lasso(X[:, 1:p .!= ci], X[:, ci], 2. * sqrt(Sx[ci, ci] * log(p) / nx))
      lasso_y_r = _lasso(Y[:, 1:p .!= ri], Y[:, ri], 2. * sqrt(Sy[ri, ri] * log(p) / ny))
      lasso_y_c = _lasso(Y[:, 1:p .!= ci], Y[:, ci], 2. * sqrt(Sy[ci, ci] * log(p) / ny))

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


# ind_s --- coordinates for the nonzero element of the true Δ (LinearIndices)
function estimate(::AsymmetricOracleBoot,
    Sx::Symmetric,
    Sy::Symmetric,
    X,
    Y,
    row,
    col,
    ind_s::Vector{Int64};
    boot_samples=nothing, bootSamples=nothing,
    rng::AbstractRNG=Random.default_rng()
    )
  boot_samples = _deprecated_keyword(boot_samples, bootSamples, :boot_samples, :bootSamples, 1000)

  nx, p = size(X)
  ny    = size(Y, 1)

  make_first!(ind_s, p, row, col)

  A = kron_sub(Sy, Sx, ind_s, ind_s)
  Δ = A \ (Sy[ind_s] - Sx[ind_s])
  Δab = row == col ? Δ[1] : ( Δ[1] + Δ[2] ) / 2.

  boot_est = zeros(Float64, boot_samples)
  for b=1:boot_samples
     b_x_ind = rand(rng, 1:nx, nx)
     b_y_ind = rand(rng, 1:ny, ny)
     b_sx = cov(X[b_x_ind, :])
     b_sy = cov(Y[b_y_ind, :])
     kron_sub!(A, b_sy, b_sx, ind_s, ind_s)
     Δb = A \ (b_sy[ind_s] - b_sx[ind_s])
     boot_est[b] = row == col ? Δb[1] : ( Δb[1] + Δb[2] ) / 2.
  end

  DiffPrecResultBoot(Δab, boot_est)
end


# ind_s --- coordinates for the nonzero element of the true Δ (LinearIndices)
function estimate(::SymmetricOracleBoot,
    Sx::Symmetric,
    Sy::Symmetric,
    X,
    Y,
    row,
    col,
    ind_s::Vector{Int64};
    boot_samples=nothing, bootSamples=nothing,
    rng::AbstractRNG=Random.default_rng()
    )
  boot_samples = _deprecated_keyword(boot_samples, bootSamples, :boot_samples, :bootSamples, 1000)

  nx, p = size(X)
  ny    = size(Y, 1)

  make_first!(ind_s, p, row, col)

  C = skron_sub(Sx, Sy, ind_s, ind_s)
  Δab = (C \ (Sy[ind_s] - Sx[ind_s]))[1]

  boot_est = zeros(Float64, boot_samples)
  for b=1:boot_samples
     b_x_ind = rand(rng, 1:nx, nx)
     b_y_ind = rand(rng, 1:ny, ny)
     b_sx = cov(X[b_x_ind, :])
     b_sy = cov(Y[b_y_ind, :])
     skron_sub!(C, b_sx, b_sy, ind_s, ind_s)
     boot_est[b] = (C \ (b_sy[ind_s] - b_sx[ind_s]))[1]
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
    ind_s::Vector{Int64}
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  make_first!(ind_s, p, row, col)

  A = kron_sub(Sy, Sx, ind_s, ind_s)
  Δ = A \ (Sy[ind_s] - Sx[ind_s])

  tmp = zeros(length(ind_s))
  tmp[1] = 1.
  ω = A \ tmp

  v = variance(AsymmetricOracleNormal(), Sx, Sy, X, Y, ω, ind_s, Δ, ind_s)
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
    ind_s::Vector{Int64}
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  make_first!(ind_s, p, row, col)
  C = skron_sub(Sx, Sy, ind_s, ind_s)

  tmp = zeros(length(ind_s))
  tmp[1] = 1.
  ω = C \ tmp
  Δ = C \ (Sy[ind_s] - Sx[ind_s])
  v = variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, ind_s, Δ, ind_s)

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
          @inbounds v += _get_elem_skron(Sx, Sy, indω[ri], indΔn[ci]) * ω[ri] * Δn[ci]
      end
  end
  for ri=1:length(indω)
      @inbounds v -= ω[ri] * (Sy[indω[ri]] - Sx[indω[ri]])
  end
  a = 0.
  if row == col
      ind = (col-1)*p + row
      for ri=1:length(indω)
          @inbounds a += ω[ri] * _get_elem_skron(Sx, Sy, indω[ri], ind)
      end
  else
      ind1 = (col-1)*p + row
      ind2 = (row-1)*p + col
      for ri=1:length(indω)
          @inbounds a += ω[ri] * (_get_elem_skron(Sx, Sy, indω[ri], ind1) + _get_elem_skron(Sx, Sy, indω[ri], ind2))
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

include("deprecated.jl")

include("precompile.jl")

end
