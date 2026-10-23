####################################
#
# loss tr(Σx⋅Θ⋅Σy⋅Θ')/2 - Θ[a, b]    ---> computes inverse of (Σy ⊗ Σx)
#
####################################

# The loss tr(Σx⋅Θ⋅Σy⋅Θ')/2 - Θ[a, b] of a p × p matrix Θ, held as vec(Θ) in a
# SparseIterate, as ProximalBase's KronQuadratic. Its minimizer is the row
# (b - 1) p + a of the inverse of Σy ⊗ Σx.
function _inverseKroneckerLoss(Σx::Symmetric, Σy::Symmetric, a::Int, b::Int)
  E = zeros(size(Σx))
  E[a, b] = -1.
  KronQuadratic(Σx, Σy, E)
end


############################################
#
#   asymmetric version of invHessian
#
############################################

# A = Sx⋅Θ⋅Sy
function _computeAsymVarElem(
    q, r,
    A, a, b,
    SxΘ, ΘSy,
    X, Y, row, col)

    nx, p = size(X)
    ny    = size(Y, 1)

    δ = row == a && col == b ? - 1. : 0.

    fill!(q, 0.)
    fill!(r, 0.)

    # compute t_ab
    @inbounds t_ab = A[row, col] + δ

    # compute qk
    for k=1:nx
        v = 0.
        for l=1:p
            @inbounds v += X[k, l] * ΘSy[l, col]
        end
        @inbounds q[k] = X[k, row] * v + δ
    end

    # compute rk
    for k=1:ny
        v = 0.
        for l=1:p
            @inbounds v += SxΘ[row, l] * Y[k, l]
        end
        @inbounds r[k] = v * Y[k, col] + δ
    end

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t_ab^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t_ab^2

    return (σ1/nx + σ2/ny)

end

# the weights of the penalty on vec(Θ); the entry (a, b) is not penalized
function _computeAsymVar!(ω, Sx, Sy, a, b, X, Y, Θ)
    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    SxΘ = _mul(Sx, Θ)
    ΘSy = _mul(Θ, Sy)
    A = SxΘ * Sy

    for col=1:p
        for row=1:p
            linInd = (col-1)*p + row
            ω[linInd] = sqrt(_computeAsymVarElem(q, r, A, a, b, SxΘ, ΘSy, X, Y, row, col))
        end
    end
    ω[(b-1)*p + a] = 0.
    ω
end



"""
    invAsymHessianEstimation(Sx, Sy, ri, ci, X, Y, λ, method=CDSolver())

Second stage of [`AsymmetricNormal`](@ref): estimate the row `(ci - 1) p + ri`
of the inverse of `Σy ⊗ Σx` by minimizing the loss `tr(Σx⋅Θ⋅Σy⋅Θ')/2 - Θ[ri, ci]`
of a `p × p` matrix `Θ` plus a weighted ℓ1 penalty with level `λ` that leaves
that entry unpenalized, by ProximalOPT's coordinate descent with the options
`method`. The weights are standard errors of the gradient estimated from `X`
and `Y`, computed at the starting point, the indicator of the entry, and again
at the first solution. Returns a `SparseIterate` holding `vec(Θ)`, of length
`p²`.
"""
function invAsymHessianEstimation(Sx::Symmetric, Sy::Symmetric, ri, ci, X, Y, λ, method::CDSolver=CDSolver())
    p = size(X, 2)
    x = SparseIterate(p*p)
    x[(ci-1)*p + ri] = 1.
    ω = zeros(p*p)
    _twoStage!(x, _inverseKroneckerLoss(Sx, Sy, ri, ci), λ, ω,
      (ω, x) -> _computeAsymVar!(ω, Sx, Sy, ri, ci, X, Y, x), method)
end
