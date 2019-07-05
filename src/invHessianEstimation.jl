########################
#
# util functions
#
########################
function _mul(S::Symmetric, θ::SparseIterate)
    p = size(S, 1)

    out = zeros(p, p)
    for ci=1:p
        for ri=1:p
            for k=1:p
                @inbounds out[ri, ci] += S[ri, k] * θ[(ci-1)*p + k]
            end
        end
    end
    out
end

function _mul(θ::SparseIterate, S::Symmetric)
    p = size(S, 1)

    out = zeros(p, p)
    for ci=1:p
        for ri=1:p
            for k=1:p
                @inbounds out[ri, ci] += θ[(k-1)*p + ri] * S[k, ci]
            end
        end
    end
    out
end



########################
#
# method with symmetric hessian
#
########################

function _computeVarElem(
    q, r,
    f::CDInverseSymKroneckerLoss,
    SxΘ, ΘSx, SyΘ, ΘSy,
    X, Y, row, col)

    nx, p = size(X)
    ny    = size(Y, 1)

    A = f.A
    B = f.B
    a = f.a
    b = f.b

    δ = row == a && col == b ? - 1. : 0.

    fill!(q, 0.)
    fill!(r, 0.)

    # compute t_ab
    @inbounds t_ab = (A[row, col] + B[row, col]) / 2. + δ

    # compute qk
    for k=1:nx
        v1 = 0.
        v2 = 0.
        for l=1:p
            @inbounds v1 += X[k, l] * ΘSy[l, col]
            @inbounds v2 += SyΘ[row, l] * X[k, l]
        end

        @inbounds q[k] = ( X[k, row] * v1 + v2 * X[k, col] ) / 2. + δ
    end

    # compute rk
    for k=1:ny
        v1 = 0.
        v2 = 0.
        for l=1:p
            @inbounds v1 += Y[k, l] * ΘSx[l, col]
            @inbounds v2 += SxΘ[row, l] * Y[k, l]
        end

        @inbounds r[k] = ( Y[k, row] * v1 + v2 * Y[k, col] ) / 2. + δ
    end

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t_ab^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t_ab^2

    return (σ1/nx + σ2/ny)

end

function _computeVar!(ω, f::CDInverseSymKroneckerLoss, X, Y, Θ)
    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    SxΘ = _mul(f.Σx, Θ)
    ΘSx = _mul(Θ, f.Σx)
    SyΘ = _mul(f.Σy, Θ)
    ΘSy = _mul(Θ, f.Σy)

    for col=1:p
        for row=1:p
            linInd = (col-1)*p + row
            ω[linInd] = sqrt(_computeVarElem(q, r, f, SxΘ, ΘSx, SyΘ, ΘSy, X, Y, row, col))
        end
    end

end

function invHessianEstimation(Sx::Symmetric, Sy::Symmetric, ri, ci, X, Y, λ, options=CDOptions())
    nx, p = size(X)
    ny = size(Y, 1)

    f = CDInverseSymKroneckerLoss(Sx, Sy, ri, ci)
    x = SparseIterate(p*p)
    x[(ci-1)*p + ri] = 1.
    CoordinateDescent.initialize!(f, x)

    ##################
    #
    #  first stage
    #
    ##################
    # compute initial variance
    ω = Array{eltype(Sx)}(undef, length(x))
    _computeVar!(ω, f, X, Y, x)
    ω[(ci-1)*p + ri] = 0.

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
    ω[(ci-1)*p + ri] = 0.

    # recompute estimate
    g = ProxL1(λ, ω)
    coordinateDescent!(x, f, g, opt1)

    return x
end

# function _computeVarAltH!(ω, Sx, nx, Sy, ny, a, b)
#     p = size(Sx, 2)
#
#     for col=1:p
#         for row=1:p
#             linInd = (col-1)*p + row
#
#             vxca = (Sx[row, row] * Sx[a, a] + Sx[row, a] * Sx[a, row]) / (nx - 1)
#             vydb = (Sy[col, col] * Sy[b, b] + Sy[col, b] * Sy[b, col]) / (ny - 1)
#
#             vyca = (Sy[row, row] * Sy[a, a] + Sy[row, a] * Sy[a, row]) / (ny - 1)
#             vxdb = (Sx[col, col] * Sx[b, b] + Sx[col, b] * Sx[b, col]) / (nx - 1)
#
#             ω[linInd] = sqrt( 2. * (vxca*vydb + vyca*vxdb) )
#         end
#     end
#
# end


# function invHessianEstimationAlt1(Sx::Symmetric, Sy::Symmetric, ri, ci, X, Y, λ, options=CDOptions())
#     nx, p = size(X)
#     ny = size(Y, 1)
#
#     f = CDInverseSymKroneckerLoss(Sx, Sy, ri, ci)
#     x = SparseIterate(p*p)
#     x[(ci-1)*p + ri] = 1.
#     CoordinateDescent.initialize!(f, x)
#
#     ##################
#     #
#     #  first stage
#     #
#     ##################
#     # compute initial variance
#     ω = Array{eltype(Sx)}(undef, length(x))
#     _computeVar!(ω, f, X, Y, x)
#     ω[(ci-1)*p + ri] = 0.
#     @show ω
#
#     # compute initial estimate
#     g = ProxL1(λ, ω)
#     coordinateDescent!(x, f, g, options)
#
#     return x
# end
#
# function invHessianEstimationAlt(Sx::Symmetric, Sy::Symmetric, ri, ci, X, Y, λ, options=CDOptions())
#     nx, p = size(X)
#     ny = size(Y, 1)
#
#     f = CDInverseSymKroneckerLoss(Sx, Sy, ri, ci)
#     x = SparseIterate(p*p)
#
#     ω = Array{eltype(Sx)}(undef, length(x))
#     _computeVarAltH!(ω, Sx, nx, Sy, ny, ri, ci)
#     ω[(ci-1)*p + ri] = 0.
#     @show ω
#
#     g = ProxL1(λ, ω)
#     coordinateDescent!(x, f, g, options)
#
#     return x
# end



############################################
#
#   asymmetric version of invHessian
#
############################################

function _computeVarElem(
    q, r,
    f::CDInverseKroneckerLoss,
    SxΘ, ΘSy,
    X, Y, row, col)

    nx, p = size(X)
    ny    = size(Y, 1)

    A = f.A
    a = f.a
    b = f.b

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

function _computeVar!(ω, f::CDInverseKroneckerLoss, X, Y, Θ)
    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    SxΘ = _mul(f.Σx, Θ)
    ΘSy = _mul(Θ, f.Σy)

    for col=1:p
        for row=1:p
            linInd = (col-1)*p + row
            ω[linInd] = sqrt(_computeVarElem(q, r, f, SxΘ, ΘSy, X, Y, row, col))
        end
    end

end



function invAsymHessianEstimation(Sx::Symmetric, Sy::Symmetric, ri, ci, X, Y, λ, options=CDOptions())
    nx, p = size(X)
    ny = size(Y, 1)

    f = CDInverseKroneckerLoss(Sx, Sy, ri, ci)
    x = SparseIterate(p*p)
    x[(ci-1)*p + ri] = 1.
    CoordinateDescent.initialize!(f, x)


    ##################
    #
    #  first stage
    #
    ##################
    # compute initial variance
    ω = Array{eltype(Sx)}(undef, length(x))
    _computeVar!(ω, f, X, Y, x)
    ω[(ci-1)*p + ri] = 0.

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
    ω[(ci-1)*p + ri] = 0.

    # recompute estimate
    g = ProxL1(λ, ω)
    coordinateDescent!(x, f, g, opt1)

    return x
end
