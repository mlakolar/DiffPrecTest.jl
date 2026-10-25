abstract type DiffPrecSupport end

"""
    ANTSupport()

Estimate the support of `Δ` by thresholding each entry at `τ` times its
standard error from the normal approximation ([`SymmetricOracleNormal`](@ref)).
See [`support_estimate`](@ref).
"""
struct ANTSupport <: DiffPrecSupport end

"""
    BootStdSupport()

Estimate the support of `Δ` by thresholding each entry at `τ` times its
bootstrap standard error ([`SymmetricOracleBoot`](@ref)). See
[`support_estimate`](@ref).
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
[`support_estimate`](@ref).
"""
struct DTraceValidationSupport <: DiffPrecSupport end


####################################
#
#   support estimator
#
####################################

# For each entry of the lower triangle of Δ, in the order of tril_index,
# the indices into svec(Δ) of the union of the supports of the first stage estimate
# of Δ and of the second stage estimate of the row of the inverse Hessian.
function __init_support(
    Sx::Symmetric, Sy::Symmetric, X, Y)

    p = size(X, 2)
    np = div((p + 1)*p, 2)

    estim_supp = Vector{Vector{Int64}}(undef, np)

    # first stage
    λ = _penalty_level(p)
    H = Symmetric( skron(Sx, Sy) )
    b = svec(Sx) - svec(Sy)
    x1 = reduced_diff_estimation(H, b, Sx, Sy, X, Y, λ)
    S1 = get_reduced_support(x1)

    # second stage
    ind = 0
    for ci=1:p
      for ri=ci:p
        ind_elem = tril_index(p, ri, ci)
        x2 = inv_hessian_reduced(H, ind_elem, Sx, Sy, X, Y, λ)
        S2 = get_reduced_support(x2)
        ind = ind + 1
        estim_supp[ind] = (1:np)[S1 .| S2]
      end
    end

    estim_supp
end

# linear indices into a p × p matrix of a support given as a symmetric p × p
# Bool matrix, or as indices into svec(Δ)
_linear_support(row, col, S::AbstractMatrix{Bool}, p) = get_linear_support(row, col, S)

function _linear_support(row, col, S::AbstractVector{<:Integer}, p)
    supp = Vector{Int64}()
    for k in S
        r, c = tril_cartesian(p, k)
        push!(supp, (c-1)*p + r)
        if r != c
            push!(supp, (r-1)*p + c)
        end
    end
    make_first!(supp, p, row, col)
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


function _get_delta(::ANTSupport, X, Y; estim_support=nothing, estimSupport=nothing)
  estim_support = _deprecated_keyword(estim_support, estimSupport, :estim_support, :estimSupport, nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    e_s = estim_support === nothing ? __init_support(Sx, Sy, X, Y) : estim_support
    Δpoint = Array{DiffPrecResultNormal}(undef, div((p+1)*p, 2))

    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            ind_s = _linear_support(row, col, e_s[it], p)

            Δpoint[it] = estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, row, col, ind_s)
        end
    end

    Δpoint, e_s
end


"""
    support_estimate(method, X, Y, τ; estim_support=nothing)

Estimate the support of `Δ = Ωx - Ωy` with `method` one of
[`ANTSupport`](@ref) or [`BootStdSupport`](@ref): every entry in the lower
triangle is refitted and kept when it exceeds `τ` times its standard error.
Returns `(Δsupp, Δpoint, e_s)`: the thresholded estimate as a sparse `p × p`
matrix (a vector of them when `τ` is a vector of thresholds), the refitted
entries in the order of `ProximalBase.tril_index`, and the supports
used for the refits.

`estim_support` gives the support used to refit each entry, as computed by the
first and second stage estimators unless given: a vector with one element per
entry of the lower triangle, either a vector of indices into `svec(Δ)` or a
symmetric `p × p` Bool matrix.

    support_estimate(DTraceValidationSupport(), Sx, Sy, SxValid, SyValid, Λarr, method=CDSolver())

Compute the ℓ1 penalized direct difference estimator from the covariances `Sx`,
`Sy` for each penalty parameter in `Λarr` (in decreasing order) and evaluate the
loss `Sx⋅Δ⋅Sy - Sy + Sx` on the validation covariances `SxValid`, `SyValid`.
Returns `(eΔarr, i2, i_inf, loss2arr, loss_infarr)`: the estimates as sparse
matrices, the indices of the penalty parameters that minimize the Frobenius and
the maximum norm of the validation loss, and the two losses.
"""
function support_estimate(::ANTSupport, X, Y, τ0::Float64; estim_support=nothing, estimSupport=nothing)
  estim_support = _deprecated_keyword(estim_support, estimSupport, :estim_support, :estimSupport, nothing)
    p = size(X, 2)

    Δpoint, e_s = _get_delta(ANTSupport(), X, Y; estim_support=estim_support)
    Δsupp = __threshold(Δpoint, p, τ0)

    Δsupp, Δpoint, e_s
end


function support_estimate(::ANTSupport, X, Y, τArr::Vector{Float64}; estim_support=nothing, estimSupport=nothing)
  estim_support = _deprecated_keyword(estim_support, estimSupport, :estim_support, :estimSupport, nothing)
    p = size(X, 2)

    Δpoint, e_s = _get_delta(ANTSupport(), X, Y; estim_support=estim_support)

    Δsupp = Array{SparseMatrixCSC{Float64,Int64}}(undef, length(τArr))
    for i=1:length(τArr)
        Δsupp[i] = __threshold(Δpoint, p, τArr[i])
    end

    Δsupp, Δpoint, e_s
end


function _get_delta(::BootStdSupport, X, Y; estim_support=nothing, estimSupport=nothing)
  estim_support = _deprecated_keyword(estim_support, estimSupport, :estim_support, :estimSupport, nothing)
    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = Symmetric( cov(X) )
    Sy = Symmetric( cov(Y) )

    e_s = estim_support === nothing ? __init_support(Sx, Sy, X, Y) : estim_support
    Δpoint = Array{DiffPrecResultBoot}(undef, div((p+1)*p, 2))

    it = 0
    for col=1:p
        for row=col:p
            it = it + 1
            ind_s = _linear_support(row, col, e_s[it], p)

            Δpoint[it] = estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, row, col, ind_s)
        end
    end

    Δpoint, e_s
end


function support_estimate(::BootStdSupport, X, Y, τ0::Float64; estim_support=nothing, estimSupport=nothing)
  estim_support = _deprecated_keyword(estim_support, estimSupport, :estim_support, :estimSupport, nothing)
    p = size(X, 2)

    Δpoint, e_s = _get_delta(BootStdSupport(), X, Y; estim_support=estim_support)
    Δsupp = __threshold(Δpoint, p, τ0)

    Δsupp, Δpoint, e_s
end


function support_estimate(::BootStdSupport, X, Y, τArr::Vector{Float64}; estim_support=nothing, estimSupport=nothing)
  estim_support = _deprecated_keyword(estim_support, estimSupport, :estim_support, :estimSupport, nothing)
    p = size(X, 2)

    Δpoint, e_s = _get_delta(BootStdSupport(), X, Y; estim_support=estim_support)

    Δsupp = Array{SparseMatrixCSC{Float64,Int64}}(undef, length(τArr))
    for i=1:length(τArr)
        Δsupp[i] = __threshold(Δpoint, p, τArr[i])
    end

    Δsupp, Δpoint, e_s
end

####################################
#
#   support estimator  -- DTrace
#
####################################

# Λarr should be in descending order
function support_estimate(::DTraceValidationSupport,
    Sx::Symmetric, Sy::Symmetric,
    SxValid::Symmetric, SyValid::Symmetric,
    Λarr::Vector{Float64},
    method::CDSolver=CDSolver())

    numΛ = length(Λarr)
    loss2arr = Vector{Float64}(undef, numΛ)
    loss_infarr = Vector{Float64}(undef, numΛ)

    # each λ warm started from the solution at the previous one
    function validate(i, λ, x, m)
        loss2arr[i] = CovarianceSelection.diff_loss(SxValid, x, SyValid, 2)
        loss_infarr[i] = CovarianceSelection.diff_loss(SxValid, x, SyValid, Inf)
        false
    end
    path = solve_path(direct_difference_loss(Sx, Sy), NormL1(1.0), Λarr, method;
                      x=SymmetricSparseIterate(size(Sx, 1)), screening=false, callback=validate)
    eΔarr = path.solutions

    # find min loss
    i2 = argmin(loss2arr)
    i_inf = argmin(loss_infarr)

    eΔarr, i2, i_inf, loss2arr, loss_infarr
end
