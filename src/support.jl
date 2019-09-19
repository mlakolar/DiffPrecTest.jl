abstract type DiffPrecSupport end
struct ANTSupport <: DiffPrecSupport end
struct BootStdSupport <: DiffPrecSupport end
struct BootMaxSupport <: DiffPrecSupport end   # not implemented yet
struct DTraceValidationSupport <: DiffPrecSupport end


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
    S1 = getSupport(x1)

    # second stage
    ind = 0
    for ci=1:p
        for ri=ci:p
            j = sub2indLowerTriangular(p, ri, ci)
            x2 = invQSymHessian(Sx, Sy, j, X, Y, λ)
            S2 = getSupport(x2, p)
            ind = ind + 1
            estimSupp[ind] = S1 .| S2
        end
    end

    estimSupp
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


function __getΔ(::ANTSupport, X, Y; estimSupport::Union{Array{BitArray},Nothing}=nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    @time eS = estimSupport === nothing ? __initSupport(Sx, Sy, X, Y) : estimSupport
    Δpoint = Array{DiffPrecResultNormal}(undef, div((p+1)*p, 2))

    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            indS = getLinearSupport(row, col, eS[it])

            Δpoint[it] = estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, indS)
        end
    end

    Δpoint, eS
end


function supportEstimate(::ANTSupport, X, Y, τ0::Float64; estimSupport::Union{Array{BitArray},Nothing}=nothing)
    p = size(X, 2)

    Δpoint, eS = __getΔ(ANTSupport(), X, Y; estimSupport=estimSupport)
    Δsupp = __threshold(Δpoint, p, τ0)

    Δsupp, Δpoint, eS
end


function supportEstimate(::ANTSupport, X, Y, τArr::Vector{Float64}; estimSupport::Union{Array{BitArray},Nothing}=nothing)
    p = size(X, 2)

    Δpoint, eS = __getΔ(ANTSupport(), X, Y; estimSupport=estimSupport)

    Δsupp = Array{SparseMatrixCSC{Float64,Int64}}(undef, length(τArr))
    for i=1:length(τArr)
        Δsupp[i] = __threshold(Δpoint, p, τArr[i])
    end

    Δsupp, Δpoint, eS
end


function __getΔ(::BootStdSupport, X, Y; estimSupport::Union{Array{BitArray},Nothing}=nothing)
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
            indS = getLinearSupport(row, col, eS[it])

            Δpoint[it] = estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, row, col, indS)
        end
    end

    Δpoint, eS
end


function supportEstimate(::BootStdSupport, X, Y, τ0::Float64; estimSupport::Union{Array{BitArray},Nothing}=nothing)
    p = size(X, 2)

    Δpoint, eS = __getΔ(BootStdSupport(), X, Y; estimSupport=estimSupport)
    Δsupp = __threshold(Δpoint, p, τ0)

    Δsupp, Δpoint, eS
end


function supportEstimate(::BootStdSupport, X, Y, τArr::Vector{Float64}; estimSupport::Union{Array{BitArray},Nothing}=nothing)
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
    iInf = argmin(loss2arr)

    eΔarr, i2, iInf, loss2arr, lossInfarr
end
