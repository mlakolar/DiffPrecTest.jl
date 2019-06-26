

function _computeVarElem(f::CDInverseSymKroneckerLoss, X, Y, θ, row, col)
    nx, p = size(X)
    ny = size(Y, 1)

    Sx = f.Σx
    Sy = f.Σy
    A = f.A
    B = f.B
    a = f.a
    b = f.b

    δ = row == a && col == b ? - 1. : 0.

    q = zeros(nx)
    r = zeros(ny)

    tmp1 = zeros(size(Sx, 1))
    tmp2 = zeros(size(Sx, 1))

    # compute t_ab
    t_ab = (A[row, col] + B[row, col]) / 2. + δ

    # compute qk
    # tmp1 = θ[1:p, 1:p] * Sy[:, col]
    for ci=1:p
        for ri=1:p
            tmp1[ri] += θ[(ci-1)*p + ri] * Sy[ci, col]
        end
    end
    # tmp2 = Sy[row, :] * θ[1:p, 1:p]
    for ci=1:p
        for ri=1:p
            tmp2[ri] += θ[(ci-1)*p + ri] * Sy[ci, row]
        end
    end
    for k=1:nx
        q[k] = ( X[k, row] * dot(X[k, :], tmp1) + dot(tmp2, X[k, :]) * X[k, col] ) / 2. + δ
    end

    # compute rk
    fill!(tmp1, 0.)
    fill!(tmp2, 0.)
    # tmp1 = θ[1:p, 1:p] * Sx[:, col]
    for ci=1:p
        for ri=1:p
            tmp1[ri] += θ[(ci-1)*p + ri] * Sx[ci, col]
        end
    end
    # tmp2 = Sx[row, :] * θ[1:p, 1:p]
    for ci=1:p
        for ri=1:p
            tmp2[ri] += θ[(ci-1)*p + ri] * Sx[ci, row]
        end
    end
    for k=1:ny
        r[k] = ( Y[k, row] * dot(Y[k, :], tmp1) + dot(tmp2, Y[k, :]) * Y[k, col] ) / 2. + δ
    end

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t_ab^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t_ab^2

    return (σ1/nx + σ2/ny)

end

function _computeVar!(ω, f::CDInverseSymKroneckerLoss, X, Y, Θ)
    p = size(X, 2)

    for col=1:p
        for row=1:p
            linInd = (col-1)*p + row
            ω[linInd] = sqrt(_computeVarElem(f, X, Y, Θ, row, col))
        end
    end

end


# our method
function invHessianEstimation(Sx::Symmetric, Sy::Symmetric, ri, ci, X, Y, λ, options=CDOptions())
    nx, p = size(X)
    ny = size(Y, 1)

    f = CDInverseSymKroneckerLoss(Sx, Sy, ri, ci)
    x = SparseIterate(p*p)

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
