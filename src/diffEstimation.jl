


function _computeVarElem(
    q, r,
    f::CDDirectDifferenceLoss,
    SxΔ, SyΔ,
    X, Y, row, col)

    nx, p = size(X)
    ny    = size(Y, 1)

    Sx = f.Σx
    Sy = f.Σy
    A = f.A

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


function _computeVar!(ω, f::CDDirectDifferenceLoss, X, Y, Δ)
    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    SxΔ = _mul(f.Σx, Δ)
    SyΔ = _mul(f.Σy, Δ)

    for col=1:p
        for row=col:p
            linInd = sub2indLowerTriangular(p, row, col)
            ω[linInd] = sqrt(_computeVarElem(q, r, f, SxΔ, SyΔ, X, Y, row, col))
        end
    end

end



# our method
function diffEstimation(Sx::Symmetric, Sy::Symmetric, X, Y, λ, options=CDOptions())
    nx, p = size(X)
    ny = size(Y, 1)

    f = CDDirectDifferenceLoss(Sx, Sy)
    x = SymmetricSparseIterate(f.p)


    ##################
    #
    #  first stage
    #
    ##################
    # compute initial variance
    ω = Array{eltype(Sx)}(undef, length(x))
    _computeVar!(ω, f, X, Y, x)

    # compute initial estimate
    g = ProxL1(λ, ω)
    coordinateDescent!(x, f, g, options)

    ##################
    #
    #  second stage
    #
    ##################

    # recompute variance
    opt1 = CDOptions(;
        maxIter=options.maxIter,
        optTol=options.optTol,
        randomize=options.randomize,
        warmStart=true,
        numSteps=options.numSteps)

    _computeVar!(ω, f, X, Y, x)

    # recompute estimate
    g = ProxL1(λ, ω)
    coordinateDescent!(x, f, g, opt1)

    return x
end
