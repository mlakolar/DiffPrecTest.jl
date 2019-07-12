module DiffPrecTest

using Statistics, LinearAlgebra, SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using StatsBase, Distributions

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
  estimate,
  variance,
  computeSimulationResult,

  # first and second stage functions
  diffEstimation, invHessianEstimation, invAsymHessianEstimation,

  # support estimation
  ANTSupport,
  BootStdSupport,
  BootMaxSupport,
  DTraceValidationSupport,
  supportEstimate



####################################
#
#   different solvers
#
####################################

abstract type DiffPrecMethod end
struct AsymmetricOracleBoot <: DiffPrecMethod end
struct SymmetricOracleBoot <: DiffPrecMethod end
struct AsymmetricOracleNormal <: DiffPrecMethod end
struct SymmetricOracleNormal <: DiffPrecMethod end
struct AsymmetricNormal <: DiffPrecMethod end
struct SymmetricNormal <: DiffPrecMethod end
struct SeparateNormal <: DiffPrecMethod end

struct OneStep <: DiffPrecMethod end             # not implemented yet
struct OneStepRefit <: DiffPrecMethod end        # not implemented yet
struct DecorrelatedScoreNormal <: DiffPrecMethod end
struct DecorrelatedScoreOracleNormal <: DiffPrecMethod end

abstract type DiffPrecSupport end
struct ANTSupport <: DiffPrecSupport end
struct BootStdSupport <: DiffPrecSupport end
struct BootMaxSupport <: DiffPrecSupport end   # not implemented yet
struct DTraceValidationSupport <: DiffPrecSupport end

####################################
#
#   different outputs
#
####################################

struct DiffPrecResultBoot
  p::Float64
  boot_p::Vector{Float64}
end

struct DiffPrecResultNormal
  p::Float64
  std::Float64
end

struct SimulationResult
  bias::Float64
  coverage::Float64
  lenCoverage::Float64
end

########################################

include("variance.jl")
include("diffEstimation.jl")
include("invHessianEstimation.jl")
include("util.jl")



##############################
#
#   our procedure
#
##############################

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

  if Sx === nothing
      Sx = Symmetric( X'X / nx )
  end
  if Sy === nothing
      Sy = Symmetric( Y'Y / ny)
  end

  # first stage
  λ = 1.01 * quantile( Normal(), 1. - 0.1 / (p*(p+1)) )
  if Δ === nothing && suppΔ === nothing
      Δ = diffEstimation(Sx, Sy, X, Y, λ)
      suppΔ = getSupport(Δ)
  end
  if suppΔ === nothing
      suppΔ = getSupport(Δ)
  end

  # second stage
  if ω === nothing && suppω === nothing
      ω = invHessianEstimation(Sx, Sy, row, col, X, Y, λ)
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
  λ = 1.01 * quantile( Normal(), 1. - 0.1 / (p*(p+1)) )
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
    bootSamples::Int64=1000
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  makeFirst!(indS, p, row, col)

  A = kron_sub(Sy, Sx, indS, indS)
  Δ = A \ (Sy[indS] - Sx[indS])
  Δab = row == col ? Δ[1] : ( Δ[1] + Δ[2] ) / 2.

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = sample(1:nx, nx)
     bY_ind = sample(1:ny, ny)
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
    bootSamples::Int64=1000
    )

  nx, p = size(X)
  ny    = size(Y, 1)

  makeFirst!(indS, p, row, col)

  C = skron_sub(Sx, Sy, indS, indS)
  Δab = (C \ (Sy[indS] - Sx[indS]))[1]

  boot_est = zeros(Float64, bootSamples)
  for b=1:bootSamples
     bX_ind = sample(1:nx, nx)
     bY_ind = sample(1:ny, ny)
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



# indΔ does not contain coordinates we are doing inference about
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
  for ci=1:length(indΔ)
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


  v = variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, indω, Δ, indΔ)

  DiffPrecResultNormal(Δab, sqrt(v))
end



####################################
#
#   support estimator
#
####################################

function __initSupport(
    Sx::Symmetric, Sy::Symmetric, X, Y)

    nx, p = size(X)
    ny     = size(Y, 1)

    estimSupp = Array{BitArray}(undef, div((p + 1)*p, 2))

    # first stage
    λ = 1.01 * quantile(Normal(), 1. - 0.1 / (p * (p+1)))
    x1 = diffEstimation(Sx, Sy, X, Y, λ)
    S1 = getSupport(x)

    # second stage
    ind = 0
    for ci=1:p
        for ri=ci:p
            x2 = invHessianEstimation(Sx, Sy, ri, ci, X, Y, λ)
            S2 = getSupport(x2, p)
            ind = ind + 1
            estimSupp[ind] = S1 .| S2
        end
    end

    estimSupp
end

function supportEstimate(::ANTSupport, X, Y; estimSupport::Union{Array{BitArray},Nothing}=nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport
    out = Array{DiffPrecResultNormal}(undef, div((p+1)*p, 2))

    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            indS = getLinearSupport(row, col, eS[it])

            out[it] = estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, indS)
        end
    end

    Δ = zeros(Float64, p, p)
    n = min(nx, ny)
    it = 0
    τ0 = 2. * sqrt(log(p))
    for col=1:p
        for row=col:p
            it = it + 1
            τ = τ0 * out[it].std
            v = abs(out[it].p) > τ ? out[it].p : 0.
            if row == col
                Δ[row, row] = v
            else
                Δ[row, col] = v
                Δ[col, row] = v
            end
        end
    end

    Δ, out, eS
end

function supportEstimate(::BootStdSupport, X, Y; estimSupport::Union{Array{BitArray},Nothing}=nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport
    out = Array{DiffPrecResultBoot}(undef, div((p+1)*p, 2))

    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            indS = getLinearSupport(row, col, eS[it])

            out[it] = estimate(SymmetricOracleBoot(), X, Y, indS)
        end
    end

    Δ = zeros(Float64, p, p)
    n = min(nx, ny)
    it = 0
    τ0 = 2. * sqrt(log(p))
    for col=1:p
        for row=col:p
            it = it + 1
            τ = τ0 * std( out[it].boot_p )
            v = abs(out[it].p) > τ ? out[it].p : 0.
            if row == col
                Δ[row, row] = v
            else
                Δ[row, col] = v
                Δ[col, row] = v
            end
        end
    end

    Δ, out, eS
end

####################################
#
#   support estimator  -- DTrace
#
####################################

# Λarr should be in descending order
function supportEstimate(::DTraceValidationSupport,
    Sx::Symmetric, Sy::Symmetric,
    SxValid::Symmetric, SyValid::Symmetric,
    Λarr::Vector{Float64},
    options=CDOptions())


    numΛ = length(Λarr)
    eΔarr = Vector{SparseMatrixCSC{Float64,Int64}}(undef, numΛ)
    loss2arr = Vector{SparseMatrixCSC{Float64,Int64}}(undef, numΛ)
    lossInfarr = Vector{SparseMatrixCSC{Float64,Int64}}(undef, numΛ)

    f = CDDirectDifferenceLoss(Sx, Sy)
    x = SymmetricSparseIterate(f.p)
    g = ProxL1(Λarr[1])
    coordinateDescent!(x, f, g, options)
    eΔarr[1] = sparse(Matrix(x))
    loss2arr[1] = CovSel.diffLoss(SxValid, x, SyValid, 2)
    lossInfarr[1] = CovSel.diffLoss(SxValid, x, SyValid, Inf)

    opt = CDOptions(;
        maxIter=options.maxIter,
        optTol=options.optTol,
        randomize=options.randomize,
        warmStart=true,
        numSteps=options.numSteps)

    for i=2:numΛ
        g = ProxL1(λ)
        coordinateDescent!(x, f, g, opt)
        eΔarr[i] = sparse(Matrix(x))
        loss2arr[i] = CovSel.diffLoss(SxValid, x, SyValid, 2)
        lossInfarr[i] = CovSel.diffLoss(SxValid, x, SyValid, Inf)
    end

    # find min loss
    i2 = argmin(loss2arr)
    iInf = argmin(loss2arr)

    eΔarr, i2, iInf, loss2arr, lossInfarr
end


end
