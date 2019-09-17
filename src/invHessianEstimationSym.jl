####################################
#
# loss (1/2)⋅tr[(Σx⋅θ⋅Σy⋅θ + Σy⋅θ⋅Σx⋅θ)/2] - tr(Θ⋅E_ab)    ---> computes inverse of (1/2)⋅(Σy ⊗ Σx + Σx ⊗ Σy)
#
####################################
struct CDInverseSymKroneckerLoss{T<:AbstractFloat, S} <: CoordinateDifferentiableFunction
  Σx::Symmetric{T, S}
  Σy::Symmetric{T, S}
  A::Matrix{T}    # stores Σx⋅Θ⋅Σy
  B::Matrix{T}    # stores Σy⋅Θ⋅Σx
  a::Int
  b::Int
  p::Int64
end

function CDInverseSymKroneckerLoss(Σx::Symmetric{T,S}, Σy::Symmetric{T,S}, a::Int, b::Int) where {T<:AbstractFloat} where S
  (issymmetric(Σx) && issymmetric(Σy)) || throw(DimensionMismatch())
  (p = size(Σx, 1)) == size(Σy, 1) || throw(DimensionMismatch())
  CDInverseSymKroneckerLoss{T,S}(Σx, Σy, zeros(T, p, p), zeros(T, p, p), a, b, p)
end

CoordinateDescent.numCoordinates(f::CDInverseSymKroneckerLoss) = f.p*f.p
function CoordinateDescent.initialize!(f::CDInverseSymKroneckerLoss, x::SparseIterate)
# compute residuals for the loss

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  B = f.B
  p = f.p

  for ac=1:p, ar=1:p
      @inbounds A[ar,ac] = A_mul_X_mul_B_rc(Σx, x, Σy, ar, ac)
      @inbounds B[ar,ac] = A_mul_X_mul_B_rc(Σy, x, Σx, ar, ac)
  end

  nothing
end

function CoordinateDescent.gradient(
  f::CDInverseSymKroneckerLoss{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  B = f.B
  # ri, ci = ind2sub(Σx, j)
  ri, ci = Tuple(CartesianIndices(Σx)[j])
  @inbounds v = (A[ri,ci] + B[ri,ci]) / 2.
  return ri == f.a && ci == f.b ? v - 1. : v
end

function CoordinateDescent.descendCoordinate!(
  f::CDInverseSymKroneckerLoss{T},
  g::ProxL1{T},
  x::SparseIterate{T},
  j::Int64) where {T <: AbstractFloat}

  Σx = f.Σx
  Σy = f.Σy
  A = f.A
  B = f.B
  p = size(Σx, 1)

  # ri, ci = ind2sub(Σx, j)
  ri, ci = Tuple(CartesianIndices(Σx)[j])

  a = zero(T)
  b = zero(T)
  @inbounds a = (Σx[ri,ri] * Σy[ci,ci] + Σx[ci,ci] * Σy[ri,ri]) / 2.
  @inbounds b = (A[ri,ci] + B[ri,ci]) / 2.
  b = ri == f.a && ci == f.b ? b - 1. : b

  @inbounds oldVal = x[j]
  a = one(T) / a
  @inbounds x[j] -= b * a
  newVal = cdprox!(g, x, j, a)
  h = newVal - oldVal

  # update internals
  for ac=1:p, ar=1:p
    @inbounds A[ar, ac] += h * Σx[ar, ri] * Σy[ci, ac]
    @inbounds B[ar, ac] += h * Σy[ar, ri] * Σx[ci, ac]
  end
  h
end
