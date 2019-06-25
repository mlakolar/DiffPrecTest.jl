


function _computeVarElem(Sx, Sy, X, Y, Δ, row, col)
    nx = size(X, 1)
    ny = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    if row == col
        # if diagonal element
        tmp1 = zeros(size(Sx, 1))

        # compute t_aa
        t_aa = A_mul_X_mul_B_rc(Sx, Δ, Sy, row, row) - (Sy[row, row] - Sx[row, row])

        # compute qk
        mul!(tmp1, Δ, view(Sy, row, :))
        for k=1:nx
            q[k] = dot(tmp1, X[k, :]) * X[k, col] - Sy[row, row] + X[k, row]^2
        end

        # compute rk
        mul!(tmp1, Δ, view(Sx, :, col))
        for k=1:ny
            r[k] = dot(tmp1, Y[k, :]) * Y[k, row] - Y[k, row]^2 + Sx[row, row]
        end

        σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t_aa^2
        σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t_aa^2

        return σ1/nx + σ2/ny
    else
        # if off-diagonal element
        tmp1 = zeros(size(Sx, 1))
        tmp2 = zeros(size(Sx, 1))
        # compute t_ab
        t_ab = (A_mul_X_mul_B_rc(Sx, Δ, Sy, row, col) + A_mul_X_mul_B_rc(Sy, Δ, Sx, row, col))  - (Sy[row, col] - Sx[row, col]) * 2.

        # compute qk
        mul!(tmp1, Δ, view(Sy, row, :))
        mul!(tmp2, Δ, view(Sy, col, :))
        # eaSyΔ = view(Sy, row, :)' * Δ
        # ΔSyeb = Δ * view(Sy, :, col)
        for k=1:nx
            q[k] = (dot(tmp1, X[k, :]) * X[k, col] + dot(tmp2, X[k, :]) * X[k, row]) - (Sy[row, col] - X[k, row] * X[k, col]) * 2.
        end

        # compute rk
        mul!(tmp1, Δ, view(Sx, row, :))
        mul!(tmp2, Δ, view(Sx, col, :))
        # eaSxΔ = view(Sx, row, :)' * Δ
        # ΔSxeb = Δ * view(Sx, :, col)
        for k=1:ny
            r[k] = (dot(tmp2, Y[k, :]) * Y[k, row] + dot(tmp1, Y[k, :]) * Y[k, col])  - (Y[k, row] * Y[k, col] - Sx[row, col]) * 2.
        end

        σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t_ab^2
        σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t_ab^2

        return (σ1/nx + σ2/ny) / 4.
    end
end



function _computeVar!(ω, Sx, Sy, X, Y, Δ)
    p = size(Sx, 2)

    for col=1:p
        for row=col:p
            linInd = sub2indLowerTriangular(p, row, col)
            ω[linInd] = sqrt(_computeVarElem(Sx, Sy, X, Y, Δ, row, col))
        end
    end

end


# our method
function diffEstimation(X, Y, λ, options=CDOptions())
    nx, p = size(X)
    ny = size(Y, 1)

    Sx = Symmetric(X'X / nx)
    Sy = Symmetric(Y'Y / ny)

    f = CDDirectDifferenceLoss(Sx, Sy)
    x = SymmetricSparseIterate(f.p)


    ##################
    #
    #  first stage
    #
    ##################
    # compute initial variance
    ω = Array{eltype(Sx)}(undef, length(x))
    _computeVar!(ω, Sx, Sy, X, Y, x)

    # compute initial estimate
    g = ProxL1(λ, ω)
    coordinateDescent!(x, f, g, options)

    # x1 = SymmetricSparseIterate(f.p)
    # g = ProxL1(λ)
    # coordinateDescent!(x1, f, g, options)

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

    _computeVar!(ω, Sx, Sy, X, Y, x)

    # recompute estimate
    g = ProxL1(λ, ω)
    coordinateDescent!(x, f, g, opt1)

    return x
end
