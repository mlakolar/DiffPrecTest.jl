####################################
#
# computes row ind of inverse of (1/2) ⋅ Q (Σy ⊗ Σx + Σx ⊗ Σy) Q'
#
####################################
struct CDInverseReducedSymKroneckerLoss{T<:AbstractFloat, S} <: CoordinateDifferentiableFunction
  Σx::Symmetric{T, S}
  Σy::Symmetric{T, S}
  A::Matrix{T}    # stores Σx⋅Θ⋅Σy
  ind::Int64
  p::Int64
end

function CDInverseReducedSymKroneckerLoss(Σx::Symmetric{T,S}, Σy::Symmetric{T,S}, ind::Int) where {T<:AbstractFloat} where S
  (issymmetric(Σx) && issymmetric(Σy)) || throw(DimensionMismatch())
  (p = size(Σx, 1)) == size(Σy, 1) || throw(DimensionMismatch())
  CDInverseReducedSymKroneckerLoss{T,S}(Σx, Σy, zeros(T, p, p), ind, p)
end

CoordinateDescent.numCoordinates(f::CDInverseReducedSymKroneckerLoss) = div((f.p + 1) * f.p, 2)


function _A_mul_symX_mul_B_rc(
  A::Symmetric{T},
  X::SparseIterate{T},
  B::Symmetric{T},
  r::Int,
  c::Int
  ) where {T<:AbstractFloat}

  _c = sqrt(2.)
  p = size(A, 1)
  v = zero(T)
  for j=1:nnz(X)
    ind = X.nzval2ind[j]
    ri, ci = ind2subLowerTriangular(p, ind)
    if ri == ci
      @inbounds v += A[ri, r] * B[ci, c] * X.nzval[j]
    else
      @inbounds v += (A[ri, r] * B[ci, c] + A[ci, r] * B[ri, c]) * X.nzval[j] / _c
    end
  end
  v
end

function CoordinateDescent.initialize!(f::CDInverseReducedSymKroneckerLoss, x::SparseIterate)
# compute residuals for the loss

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = f.p

  for ac=1:p, ar=1:p
      @inbounds A[ar,ac] = _A_mul_symX_mul_B_rc(Σx, x, Σy, ar, ac)
  end

  nothing
end

function CoordinateDescent.gradient(
  f::CDInverseReducedSymKroneckerLoss{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  A = f.A
  B = f.B
  ri, ci = ind2subLowerTriangular(f.p, j)
  @inbounds v = (A[ri,ci] + A[ci,ri]) / 2.
  v = ri == ci ? v : sqrt(2.) * v
  return j == f.ind ? v - 1. : v
end

function CoordinateDescent.descendCoordinate!(
  f::CDInverseReducedSymKroneckerLoss{T},
  g::ProxL1{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  p = size(Σx, 1)

  ri, ci = ind2subLowerTriangular(f.p, j)

  a = zero(T)
  b = zero(T)
  if ri == ci
    @inbounds a = Σx[ri,ri] * Σy[ci,ci]
    @inbounds b = A[ri,ri]
  else
    @inbounds a = (Σx[ri,ri] * Σy[ci,ci] + Σx[ci,ci] * Σy[ri,ri]) / 2. + Σx[ri,ci] * Σy[ri,ci]
    @inbounds b = (A[ri,ci] + A[ci,ri]) / sqrt(2.)
  end
  b = j == f.ind ? b - 1. : b

  @inbounds oldVal = x[j]
  a = one(T) / a
  @inbounds x[j] -= b * a
  newVal = cdprox!(g, x, j, a)
  h = newVal - oldVal

  # update internals
  if ri == ci
    for ac=1:p, ar=1:p
      @inbounds A[ar, ac] += h * Σx[ar, ri] * Σy[ri, ac]
    end
  else
    _c = h / sqrt(2.)
    for ac=1:p, ar=1:p
      @inbounds A[ar, ac] += _c * (Σx[ar, ri] * Σy[ci, ac] + Σx[ar, ci] * Σy[ri, ac])
    end
  end
  h
end


############################################
#
#   compressed version of invHessian
#
############################################

function _mul_symX_S(θ::SparseIterate, S::Symmetric)
    p = size(S, 1)
    rp = length(θ)
    _c = 1. / sqrt(2.)

    out = zeros(p, p)

    i = 1
    @inbounds while i <= θ.nnz
      v = θ.nzval[i]
      ind = θ.nzval2ind[i]
      r, c = ind2subLowerTriangular(p, ind)
      if r == c
          for l=1:p
              @inbounds out[r, l] += S[c, l] * v
          end
      else
          for l=1:p
              t = v * _c
              @inbounds out[r, l] += t * S[c, l]
              @inbounds out[c, l] += t * S[r, l]
          end
      end
      i += 1
    end
    out
end


function _computeReducedVar!(ω, f::CDInverseReducedSymKroneckerLoss, X, Y, Θ)
    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    A = f.A
    ind = f.ind
    _c = 1. / sqrt(2.)

    ΘSx = _mul_symX_S(Θ, f.Σx)
    ΘSy = _mul_symX_S(Θ, f.Σy)

    V = X * ΘSy
    W = Y * ΘSx

    j = 0
    for col=1:p
        for row=col:p
            j += 1

            δ = ind == j ? - 1. : 0.

            fill!(q, 0.)
            fill!(r, 0.)
            t = 0.

            if row == col
                @inbounds t = A[row, row] + δ

                for k=1:nx
                    @inbounds q[k] = X[k, row] * V[k, row] + δ
                end
                for k=1:ny
                    @inbounds r[k] = Y[k, row] * W[k, row] + δ
                end
            else
                @inbounds t = (A[row, col] + A[col, row]) * _c + δ

                for k=1:nx
                    @inbounds q[k] = (X[k, row] * V[k, col] + X[k, col] * V[k, row]) * _c  + δ
                end
                for k=1:ny
                    @inbounds r[k] = (Y[k, row] * W[k, col] + Y[k, col] * W[k, row]) * _c + δ
                end
            end

            σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
            σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2

            ω[j] = sqrt((σ1/nx + σ2/ny))
        end
    end

end

function invQSymHessian(Sx::Symmetric, Sy::Symmetric, ind, X, Y, λ, options=CDOptions())
    nx, p = size(X)
    ny = size(Y, 1)

    rp = div((p+1)*p, 2)
    f = CDInverseReducedSymKroneckerLoss(Sx, Sy, ind)
    x = SparseIterate(rp)
    x[ind] = 1. / skron(Sx, Sy, ind, ind)
    CoordinateDescent.initialize!(f, x)

    ##################0
    #
    #  first stage
    #
    ##################
    # compute initial variance
    ω = Array{eltype(Sx)}(undef, length(x))
    _computeReducedVar!(ω, f, X, Y, x)
    ω[ind] = 0.    

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

    _computeReducedVar!(ω, f, X, Y, x)
    ω[ind] = 0.

    # recompute estimate
    g = ProxL1(λ, ω)
    coordinateDescent!(x, f, g, opt1)

    return x
end
