


# A = Sx⋅Δ⋅Sy
function _computeVarElem(
    q, r,
    Sx, Sy, A,
    SxΔ, SyΔ,
    X, Y, row, col)

    nx, p = size(X)
    ny    = size(Y, 1)

    fill!(q, 0.)
    fill!(r, 0.)

    if row == col
        # if diagonal element

        # compute t_aa
        @inbounds t_aa = A[row, row] - (Sy[row, row] - Sx[row, row])

        # compute qk
        for k=1:nx
            v = 0.
            for l=1:p
                @inbounds v += SyΔ[row, l] * X[k, l]
            end

            @inbounds q[k] = v * X[k, col] - Sy[row, row] + X[k, row]^2
        end

        # compute rk
        for k=1:ny

            v = 0.
            for l=1:p
                @inbounds v += SxΔ[row, l] * Y[k, l]
            end

            @inbounds r[k] = v * Y[k, row] - Y[k, row]^2 + Sx[row, row]
        end

        σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t_aa^2
        σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t_aa^2

        return σ1/nx + σ2/ny
    else
        # if off-diagonal element

        # compute t_ab
        @inbounds t_ab = A[row, col] + A[col, row] - (Sy[row, col] - Sx[row, col]) * 2.

        # compute qk
        for k=1:nx
            v1 = 0.
            v2 = 0.
            for l=1:p
                @inbounds v1 += SyΔ[row, l] * X[k, l]
                @inbounds v2 += SyΔ[col, l] * X[k, l]
            end

            @inbounds q[k] = (v1 * X[k, col] + v2 * X[k, row]) - (Sy[row, col] - X[k, row] * X[k, col]) * 2.
        end

        # compute rk
        for k=1:ny
            v1 = 0.
            v2 = 0.
            for l=1:p
                @inbounds v1 += SxΔ[row, l] * Y[k, l]
                @inbounds v2 += SxΔ[col, l] * Y[k, l]
            end

            @inbounds r[k] = (v2 * Y[k, row] + v1 * Y[k, col])  - (Y[k, row] * Y[k, col] - Sx[row, col]) * 2.
        end

        σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t_ab^2
        σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t_ab^2

        return (σ1/nx + σ2/ny) / 4.
    end
end



function _mul(S::Symmetric, θ::SymmetricSparseIterate)
    p = size(S, 1)

    out = zeros(p, p)
    for ci=1:p
        for ri=1:p
            for k=1:p
                @inbounds out[ri, ci] += S[ri, k] * θ[k, ci]
            end
        end
    end
    out
end


# the weights of the penalty on Δ as a p × p matrix, symmetric
function _computeVar!(ω, Sx, Sy, X, Y, Δ::SymmetricSparseIterate)
    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    SxΔ = _mul(Sx, Δ)
    SyΔ = _mul(Sy, Δ)
    A = SxΔ * Sy

    for col=1:p
        for row=col:p
            ω[row, col] = ω[col, row] = sqrt(_computeVarElem(q, r, Sx, Sy, A, SxΔ, SyΔ, X, Y, row, col))
        end
    end
    ω
end

# The estimators of both stages solve twice: with the weights ω of the ℓ1
# penalty computed by weights!(ω, x) at the starting point x, then, warm
# started, with the weights recomputed at the first solution.
function _twoStage!(x, f, λ, ω, weights!, method::CDSolver)
    m = cdmodel(f, x)
    for stage=1:2
        weights!(ω, x)
        solve!(x, m, NormL1(λ, ω), method)
    end
    x
end



"""
    diffEstimation(Sx, Sy, X, Y, λ, method=CDSolver())

First stage of [`SymmetricNormal`](@ref) and [`AsymmetricNormal`](@ref):
estimate `Δ = Ωx - Ωy` by minimizing the direct difference loss
`tr(Sx⋅Δ⋅Sy⋅Δ)/2 - tr(Δ⋅(Sy - Sx))` (`CovarianceSelection.directDifferenceLoss`) plus a
weighted ℓ1 penalty with level `λ`, by ProximalSolvers' coordinate descent with
the options `method`. The weights are standard errors of the gradient
estimated from `X` and `Y`, computed at zero and again at the first solution.
Returns a `SymmetricSparseIterate`.
"""
function diffEstimation(Sx::Symmetric, Sy::Symmetric, X, Y, λ, method::CDSolver=CDSolver())
    p = size(Sx, 1)
    x = SymmetricSparseIterate(p)
    ω = zeros(p, p)
    _twoStage!(x, directDifferenceLoss(Sx, Sy), λ, ω, (ω, x) -> _computeVar!(ω, Sx, Sy, X, Y, x), method)
end
