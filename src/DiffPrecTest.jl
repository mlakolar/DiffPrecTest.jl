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
struct SymmetricNormal <: DiffPrecMethod end
struct AsymmetricNormal <: DiffPrecMethod end
struct SeparateNormal <: DiffPrecMethod end
struct OneStep <: DiffPrecMethod end             # not implemented yet
struct OneStepRefit <: DiffPrecMethod end        # not implemented yet


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

  v = variance(AsymmetricOracleNormal(), Sx, Sy, X, Y, ω, Δ, indS)
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
  v = variance(SymmetricOracleNormal(), Sx, Sy, X, Y, ω, Δ, indS)

  DiffPrecResultNormal(Δ[1], sqrt(v))
end



####################################
#
#   support estimator
#
####################################

function __initSupport(
    Sx::Symmetric, Sy::Symmetric, X, Y)
    nx, px = size(X)
    ny, py = size(Y)

    estimSupp = Array{BitArray}(undef, div((px + 1)*px, 2))

    # first stage
    λ = 1.01 * quantile(Normal(), 1. - 0.1 / (px * (px+1)))
    x1 = diffEstimation(Sx, Sy, X, Y, λ)

    S1 = BitArray(undef, (px, px))
    fill!(S1, false)
    for ci=1:px
        for ri=ci:px
            if abs(x1[ri, ci] > 1e-3)
                S1[ri, ci] = true
                S1[ci, ri] = true
                S1[ri, ri] = true
                S1[ci, ci] = true
            end
        end
    end

    # second stage
    ind = 0
    for ci=1:px
        for ri=ci:px
            x2 = invHessianEstimation(Sx, Sy, ri, ci, X, Y, λ)
            ind = ind + 1
            estimSupp[ind] = copy(S1)

            for _ci=1:px
                for _ri=1:px
                    if abs(x2[_ri + (_ci-1)*px] > 1e-3)
                        estimSupp[ind][_ri, _ci] = true
                        estimSupp[ind][_ci, _ri] = true
                        estimSupp[ind][_ri, _ri] = true
                        estimSupp[ind][_ci, _ci] = true
                    end
                end
            end
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
            ind = (col - 1) * p + row

            indS = [LinearIndices((p, p))[x] for x in findall( eS[it] )]
            pos = findfirst(isequal(ind), indS)
            if  pos === nothing
                pushfirst!(indS, ind)
            else
                indS[pos], indS[1] = indS[1], indS[pos]
            end

            out[it] = estimate(SymmetricOracleNormal(), Sx, nx, Sy, ny, indS)
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
            ind = (col - 1) * p + row

            indS = [LinearIndices((p, p))[x] for x in findall( eS[it] )]
            pos = findfirst(isequal(ind), indS)
            if  pos === nothing
                pushfirst!(indS, ind)
            else
                indS[pos], indS[1] = indS[1], indS[pos]
            end

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
