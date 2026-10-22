abstract type DiffPrecSupport end

"""
    ANTSupport()

Estimate the support of `Δ` by thresholding each entry at `τ` times its
standard error from the normal approximation ([`SymmetricOracleNormal`](@ref)).
See [`supportEstimate`](@ref).
"""
struct ANTSupport <: DiffPrecSupport end

"""
    BootStdSupport()

Estimate the support of `Δ` by thresholding each entry at `τ` times its
bootstrap standard error ([`SymmetricOracleBoot`](@ref)). See
[`supportEstimate`](@ref).
"""
struct BootStdSupport <: DiffPrecSupport end

"""
    BootMaxSupport()

Not implemented yet.
"""
struct BootMaxSupport <: DiffPrecSupport end   # not implemented yet

"""
    DTraceValidationSupport()

Estimate `Δ` along a path of penalty parameters with the ℓ1 penalized direct
difference (D-trace) estimator and choose the penalty on validation data. See
[`supportEstimate`](@ref).
"""
struct DTraceValidationSupport <: DiffPrecSupport end


####################################
#
#   support estimator
#
####################################

# For each entry of the lower triangle of Δ, in the order of sub2indLowerTriangular,
# the indices into svec(Δ) of the union of the supports of the first stage estimate
# of Δ and of the second stage estimate of the row of the inverse Hessian.
function __initSupport(
    Sx::Symmetric, Sy::Symmetric, X, Y)

    p = size(X, 2)
    np = div((p + 1)*p, 2)

    estimSupp = Vector{Vector{Int64}}(undef, np)

    # first stage
    λ = _penaltyLevel(p)
    H = Symmetric( skron(Sx, Sy) )
    b = svec(Sx) - svec(Sy)
    x1 = reducedDiffEstimation(H, b, Sx, Sy, X, Y, λ)
    S1 = getReducedSupport(x1)

    # second stage
    ind = 0
    for ci=1:p
      for ri=ci:p
        indElem = sub2indLowerTriangular(p, ri, ci)
        x2 = invHessianReduced(H, indElem, Sx, Sy, X, Y, λ)
        S2 = getReducedSupport(x2)
        ind = ind + 1
        estimSupp[ind] = (1:np)[S1 .| S2]
      end
    end

    estimSupp
end

# linear indices into a p × p matrix of a support given as a symmetric p × p
# Bool matrix, or as indices into svec(Δ)
_linearSupport(row, col, S::AbstractMatrix{Bool}, p) = getLinearSupport(row, col, S)

function _linearSupport(row, col, S::AbstractVector{<:Integer}, p)
    supp = Vector{Int64}()
    for k in S
        r, c = ind2subLowerTriangular(p, k)
        push!(supp, (c-1)*p + r)
        if r != c
            push!(supp, (r-1)*p + c)
        end
    end
    makeFirst!(supp, p, row, col)
end

function __threshold(Δpoint::Vector{DiffPrecResultNormal}, p::Int, τ0::Float64)
    Δ = zeros(Float64, p, p)
    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            τ = τ0 * Δpoint[it].std
            v = abs(Δpoint[it].p) > τ ? Δpoint[it].p : 0.
            if row == col
                Δ[row, row] = v
            else
                Δ[row, col] = v
                Δ[col, row] = v
            end
        end
    end
    sparse(Δ)
end

function __threshold(Δpoint::Vector{DiffPrecResultBoot}, p::Int, τ0::Float64)
    Δ = zeros(Float64, p, p)
    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            τ = τ0 * std( Δpoint[it].boot_p )
            v = abs(Δpoint[it].p) > τ ? Δpoint[it].p : 0.
            if row == col
                Δ[row, row] = v
            else
                Δ[row, col] = v
                Δ[col, row] = v
            end
        end
    end
    sparse(Δ)
end


function __getΔ(::ANTSupport, X, Y; estimSupport::Union{AbstractVector,Nothing}=nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport
    Δpoint = Array{DiffPrecResultNormal}(undef, div((p+1)*p, 2))

    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            indS = _linearSupport(row, col, eS[it], p)

            Δpoint[it] = estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, indS)
        end
    end

    Δpoint, eS
end


"""
    supportEstimate(method, X, Y, τ; estimSupport=nothing)

Estimate the support of `Δ = Ωx - Ωy` with `method` one of
[`ANTSupport`](@ref) or [`BootStdSupport`](@ref): every entry in the lower
triangle is refitted and kept when it exceeds `τ` times its standard error.
Returns `(Δsupp, Δpoint, eS)`: the thresholded estimate as a sparse `p × p`
matrix (a vector of them when `τ` is a vector of thresholds), the refitted
entries in the order of `ProximalBase.sub2indLowerTriangular`, and the supports
used for the refits.

`estimSupport` gives the support used to refit each entry, as computed by the
first and second stage estimators unless given: a vector with one element per
entry of the lower triangle, either a vector of indices into `svec(Δ)` or a
symmetric `p × p` Bool matrix.

    supportEstimate(DTraceValidationSupport(), Sx, Sy, SxValid, SyValid, Λarr, options=CDOptions())

Compute the ℓ1 penalized direct difference estimator from the covariances `Sx`,
`Sy` for each penalty parameter in `Λarr` (in decreasing order) and evaluate the
loss `Sx⋅Δ⋅Sy - Sy + Sx` on the validation covariances `SxValid`, `SyValid`.
Returns `(eΔarr, i2, iInf, loss2arr, lossInfarr)`: the estimates as sparse
matrices, the indices of the penalty parameters that minimize the Frobenius and
the maximum norm of the validation loss, and the two losses.
"""
function supportEstimate(::ANTSupport, X, Y, τ0::Float64; estimSupport::Union{AbstractVector,Nothing}=nothing)
    p = size(X, 2)

    Δpoint, eS = __getΔ(ANTSupport(), X, Y; estimSupport=estimSupport)
    Δsupp = __threshold(Δpoint, p, τ0)

    Δsupp, Δpoint, eS
end


function supportEstimate(::ANTSupport, X, Y, τArr::Vector{Float64}; estimSupport::Union{AbstractVector,Nothing}=nothing)
    p = size(X, 2)

    Δpoint, eS = __getΔ(ANTSupport(), X, Y; estimSupport=estimSupport)

    Δsupp = Array{SparseMatrixCSC{Float64,Int64}}(undef, length(τArr))
    for i=1:length(τArr)
        Δsupp[i] = __threshold(Δpoint, p, τArr[i])
    end

    Δsupp, Δpoint, eS
end


function __getΔ(::BootStdSupport, X, Y; estimSupport::Union{AbstractVector,Nothing}=nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport
    Δpoint = Array{DiffPrecResultBoot}(undef, div((p+1)*p, 2))

    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            indS = _linearSupport(row, col, eS[it], p)

            Δpoint[it] = estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, row, col, indS)
        end
    end

    Δpoint, eS
end


function supportEstimate(::BootStdSupport, X, Y, τ0::Float64; estimSupport::Union{AbstractVector,Nothing}=nothing)
    p = size(X, 2)

    Δpoint, eS = __getΔ(BootStdSupport(), X, Y; estimSupport=estimSupport)
    Δsupp = __threshold(Δpoint, p, τ0)

    Δsupp, Δpoint, eS
end


function supportEstimate(::BootStdSupport, X, Y, τArr::Vector{Float64}; estimSupport::Union{AbstractVector,Nothing}=nothing)
    p = size(X, 2)

    Δpoint, eS = __getΔ(BootStdSupport(), X, Y; estimSupport=estimSupport)

    Δsupp = Array{SparseMatrixCSC{Float64,Int64}}(undef, length(τArr))
    for i=1:length(τArr)
        Δsupp[i] = __threshold(Δpoint, p, τArr[i])
    end

    Δsupp, Δpoint, eS
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
    loss2arr = Vector{Float64}(undef, numΛ)
    lossInfarr = Vector{Float64}(undef, numΛ)

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
        g = ProxL1(Λarr[i])
        coordinateDescent!(x, f, g, opt)
        eΔarr[i] = sparse(Matrix(x))
        loss2arr[i] = CovSel.diffLoss(SxValid, x, SyValid, 2)
        lossInfarr[i] = CovSel.diffLoss(SxValid, x, SyValid, Inf)
    end

    # find min loss
    i2 = argmin(loss2arr)
    iInf = argmin(lossInfarr)

    eΔarr, i2, iInf, loss2arr, lossInfarr
end
