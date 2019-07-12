
# computes the variance of
#
#     ω'(H Δ - (Sy - Sx))
#
# where H = (Sx ⊗ Sy + Sy ⊗ Sx) / 2
# and ω, Δ are fixed
function variance(
    ::SymmetricOracleNormal,
    Sx::Symmetric,
    Sy::Symmetric,
    X::AbstractMatrix,
    Y::AbstractMatrix,
    ω::Vector,
    Δ::Vector,
    indS::Vector{Int64}
    )

    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    # compute the value of the U-statistics
    t = 0.
    for ci=1:length(indS)
        for ri=1:length(indS)
            @inbounds t += _getElemSKron(Sx, Sy, indS[ri], indS[ci]) * ω[ri] * Δ[ci]
        end
        @inbounds t -= ω[ci] * (Sy[indS[ci]] - Sx[indS[ci]])
    end

    # compute qk
    for k=1:nx
        v = 0.
        for ci=1:length(indS)
            for ri=1:length(indS)
                @inbounds v += _getElemSKron(Sy, view(X, k, :), indS[ri], indS[ci]) * ω[ri] * Δ[ci]
            end
            row, col = Tuple(CartesianIndices((p, p))[indS[ci]])
            @inbounds v -= ω[ci] * (Sy[indS[ci]] - X[k, row] * X[k, col])
        end
        @inbounds q[k] = v
    end
    # compute rk
    for k=1:ny
        v = 0.
        for ci=1:length(indS)
            for ri=1:length(indS)
                @inbounds v += _getElemSKron(Sx, view(Y, k, :), indS[ri], indS[ci]) * ω[ri] * Δ[ci]
            end
            row, col = Tuple(CartesianIndices((p, p))[indS[ci]])
            @inbounds v -= ω[ci] * (Y[k, row] * Y[k, col] - Sx[indS[ci]])
        end
        @inbounds r[k] = v
    end

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2

    return σ1/nx + σ2/ny
end



# computes the variance of
#
#     ω'(H Δ - (Sy - Sx))
#
# where H = Sy ⊗ Sx
# and ω, Δ are fixed
function variance(
    ::AsymmetricOracleNormal,
    Sx::Symmetric,
    Sy::Symmetric,
    X::AbstractMatrix,
    Y::AbstractMatrix,
    ω::Vector,
    Δ::Vector,
    indS::Vector{Int64}
    )

    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    # compute the value of the U-statistics
    t = 0.
    for ci=1:length(indS)
        for ri=1:length(indS)
            @inbounds t += _getElemKron(Sy, Sx, indS[ri], indS[ci]) * ω[ri] * Δ[ci]
        end
        @inbounds t -= ω[ci] * (Sy[indS[ci]] - Sx[indS[ci]])
    end

    # compute qk
    for k=1:nx
        v = 0.
        for ci=1:length(indS)
            for ri=1:length(indS)
                @inbounds v += _getElemKron(Sy, view(X, k, :), indS[ri], indS[ci]) * ω[ri] * Δ[ci]
            end
            row, col = Tuple(CartesianIndices((p, p))[indS[ci]])
            @inbounds v -= ω[ci] * (Sy[indS[ci]] - X[k, row] * X[k, col])
        end
        @inbounds q[k] = v
    end
    # compute rk
    for k=1:ny
        v = 0.
        for ci=1:length(indS)
            for ri=1:length(indS)
                @inbounds v += _getElemKron(view(Y, k, :), Sx, indS[ri], indS[ci]) * ω[ri] * Δ[ci]
            end
            row, col = Tuple(CartesianIndices((p, p))[indS[ci]])
            @inbounds v -= ω[ci] * (Y[k, row] * Y[k, col] - Sx[indS[ci]])
        end
        @inbounds r[k] = v
    end

    σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
    σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2

    return σ1/nx + σ2/ny
end
