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
              S[ri, ri] = true
              S[ci, ci] = true
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
              S[ri, ri] = true
              S[ci, ci] = true
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

  estimate(SymmetricOracleNormal(), Sx, nx, Sy, ny, indS), x1, x2, indS
end



####################################
#
#   competing procedures
#
####################################


function estimate(::AsymmetricNormal, X, Y, ind;
    x1::Union{SymmetricSparseIterate, Nothing}=nothing)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = Symmetric(cov(X))
  Sy = Symmetric(cov(Y))

  # first stage
  λ = 1.01 * quantile(Normal(), 1. - 0.1 / (px * (px+1)))
  if x1 === nothing
      x1 = diffEstimation(Sx, Sy, X, Y, λ)
  end

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
  x2 = invAsymHessianEstimation(Sx, Sy, ri, ci, X, Y, λ)
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

  estimate(AsymmetricOracleNormal(), Sx, nx, Sy, ny, indS), x1, x2, indS
end


function estimate(::SeparateNormal, X, Y, ind)
  nx, px = size(X)
  ny, py = size(Y)
  @assert px == py

  Sx = Symmetric(cov(X))
  Sy = Symmetric(cov(Y))

  I = CartesianIndices(Sx)
  ri, ci = Tuple( I[ind] )

  if ri == ci
      lasso_x_r = lasso(X[:, 1:px .!= ri], X[:, ri], 2. * sqrt(Sx[ri, ri] * log(px) / nx))
      lasso_y_r = lasso(Y[:, 1:px .!= ri], Y[:, ri], 2. * sqrt(Sy[ri, ri] * log(px) / ny))

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

      lasso_x_r = lasso(X[:, 1:px .!= ri], X[:, ri], 2. * sqrt(Sx[ri, ri] * log(px) / nx))
      lasso_x_c = lasso(X[:, 1:px .!= ci], X[:, ci], 2. * sqrt(Sx[ci, ci] * log(px) / nx))
      lasso_y_r = lasso(Y[:, 1:px .!= ri], Y[:, ri], 2. * sqrt(Sy[ri, ri] * log(px) / ny))
      lasso_y_c = lasso(Y[:, 1:px .!= ci], Y[:, ci], 2. * sqrt(Sy[ci, ci] * log(px) / ny))

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




function variance(::SymmetricOracleNormal, Sx, Sy, X, Y, Δab)
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
    loss2arr[1] = diffLoss(SxValid, x, SyValid, 2)
    lossInfarr[1] = diffLoss(SxValid, x, SyValid, Inf)

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
        loss2arr[i] = diffLoss(SxValid, x, SyValid, 2)
        lossInfarr[i] = diffLoss(SxValid, x, SyValid, Inf)
    end

    # find min loss
    i2 = argmin(loss2arr)
    iInf = argmin(loss2arr)

    eΔarr, i2, iInf, loss2arr, lossInfarr
end


end
