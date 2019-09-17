####################################
#
# loss tr(Σx⋅θ⋅Σy⋅θ)/2 - tr(Θ⋅E_ab)    ---> computes inverse of (Σy ⊗ Σx)
#
####################################
struct CDInverseKroneckerLoss{T<:AbstractFloat, S} <: CoordinateDifferentiableFunction
  Σx::Symmetric{T, S}
  Σy::Symmetric{T, S}
  A::Matrix{T}    # stores Σx⋅Θ⋅Σy
  a::Int
  b::Int
  p::Int64
end

function CDInverseKroneckerLoss(Σx::Symmetric{T,S}, Σy::Symmetric{T,S}, a::Int, b::Int) where {T<:AbstractFloat} where S
  (issymmetric(Σx) && issymmetric(Σy)) || throw(DimensionMismatch())
  (p = size(Σx, 1)) == size(Σy, 1) || throw(DimensionMismatch())
  CDInverseKroneckerLoss{T,S}(Σx, Σy, zeros(T, p, p), a, b, p)
end

CoordinateDescent.numCoordinates(f::CDInverseKroneckerLoss) = f.p*f.p
function CoordinateDescent.initialize!(f::CDInverseKroneckerLoss, x::SparseIterate)
# compute residuals for the loss

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  for ac=1:p, ar=1:p
      @inbounds A[ar,ac] = A_mul_X_mul_B_rc(Σx, x, Σy, ar, ac)
  end

  nothing
end

function CoordinateDescent.gradient(
  f::CDInverseKroneckerLoss{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  # ri, ci = ind2sub(Σx, j)
  ri, ci = Tuple(CartesianIndices(Σx)[j])
  @inbounds v = A[ri,ci]
  return ri == f.a && ci == f.b ? v - 1. : v
end


function CoordinateDescent.descendCoordinate!(
  f::CDInverseKroneckerLoss{T},
  g::ProxL1{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = size(Σx, 1)

  # ri, ci = ind2sub(Σx, j)
  ri, ci = Tuple(CartesianIndices(Σx)[j])

  a = zero(T)
  b = zero(T)
  @inbounds a  = Σx[ri,ri] * Σy[ci,ci]
  @inbounds b = A[ri,ci]
  b = ri == f.a && ci == f.b ? b - 1. : b

  @inbounds oldVal = x[j]
  a = one(T) / a
  @inbounds x[j] -= b * a
  newVal = cdprox!(g, x, j, a)
  h = newVal - oldVal

  # update internals
  for ac=1:p, ar=1:p
    @inbounds A[ar, ac] += h * Σx[ar, ri] * Σy[ci, ac]
  end
  h
end


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
