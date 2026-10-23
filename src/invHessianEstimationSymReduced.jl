####################################
#
# computes row ind of inverse of (1/2) ⋅ Q (Σy ⊗ Σx + Σx ⊗ Σy) Q'
#
####################################
# The loss θ'Hθ/2 - θ[ind], with H = Q (Σx ⊗ Σy + Σy ⊗ Σx) Q' / 2 the Hessian of
# the direct difference loss in the half-vectorized parametrization θ = svec(Θ)
# of a symmetric Θ, is ⟨Θ, ΣxΘΣy⟩/2 + ⟨C, Θ⟩ as a function of Θ: ProximalBase's
# KronQuadratic on a SymmetricSparseIterate, with C = -E_rr on the diagonal and
# C = -(E_rc + E_cr)/√2 off it, (r, c) the entry of ind. In svec(Θ) the
# off-diagonal entries are scaled by √2, so the penalty weight ω of θ[k] is ω on
# Θ[r, r], and ω/√2 on each of Θ[r, c] and Θ[c, r].
function _inverseSymKroneckerLoss(Σx::Symmetric, Σy::Symmetric, ind::Int)
  p = size(Σx, 1)
  r, c = tril_cartesian(p, ind)
  C = zeros(p, p)
  if r == c
    C[r, r] = -1.
  else
    C[r, c] = C[c, r] = -1. / sqrt(2.)
  end
  KronQuadratic(Σx, Σy, C)
end

# svec(Θ) of the symmetric Θ, as a SparseIterate of length p(p+1)/2
function _svec(x::SymmetricSparseIterate{T}) where {T}
  p = size(x, 1)
  θ = SparseIterate(T, div(p * (p + 1), 2))
  cx = coordinates(x)
  for k in activecoordinates(x)
    v = cx[k]
    iszero(v) && continue
    r, c = tril_cartesian(p, k)
    θ[k] = r == c ? v : sqrt(2.) * v
  end
  θ
end

# the weights W of the penalty on the symmetric Θ from the weights ω of svec(Θ)
function _symWeights!(W, ω, p)
  k = 0
  for c=1:p, r=c:p
    k += 1
    W[r, c] = W[c, r] = r == c ? ω[k] : ω[k] / sqrt(2.)
  end
  W
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
      r, c = tril_cartesian(p, ind)
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


# the weights of the penalty on svec(Θ); the entry ind is not penalized
function _computeReducedVar!(ω, Sx, Sy, ind, X, Y, Θ)
    nx, p = size(X)
    ny    = size(Y, 1)

    q = zeros(nx)
    r = zeros(ny)

    _c = 1. / sqrt(2.)

    ΘSx = _mul_symX_S(Θ, Sx)
    ΘSy = _mul_symX_S(Θ, Sy)
    A = Sx * ΘSy      # Sx⋅Θ⋅Sy

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
    ω[ind] = 0.
    ω
end

"""
    invQSymHessian(Sx, Sy, ind, X, Y, λ, method=CDSolver())

Second stage of [`SymmetricNormal`](@ref): estimate the row `ind` of the inverse
of `H = Q (Σx ⊗ Σy + Σy ⊗ Σx) Q' / 2`, where `ind` indexes the lower triangle
(see `ProximalBase.tril_index`), by minimizing `θ'Hθ/2 - θ[ind]` plus a weighted
ℓ1 penalty with level `λ` that leaves the entry `ind` unpenalized, by
ProximalSolvers' coordinate descent with the options `method`. The weights are
standard errors of the gradient estimated from `X` and `Y`, computed at the
starting point `e_ind / H[ind, ind]` and again at the first solution. Returns a
`SparseIterate` of length `p(p+1)/2`.
"""
function invQSymHessian(Sx::Symmetric, Sy::Symmetric, ind, X, Y, λ, method::CDSolver=CDSolver())
    p = size(X, 2)

    # coordinate descent on the symmetric Θ = smat(θ), which takes the steps of
    # coordinate descent on θ
    x = SymmetricSparseIterate(p)
    ri, ci = tril_cartesian(p, ind)
    θind = 1. / skron(Sx, Sy, ind, ind)
    x[ri, ci] = ri == ci ? θind : θind / sqrt(2.)

    ω = zeros(div((p+1)*p, 2))
    W = zeros(p, p)
    weights!(W, x) = _symWeights!(W, _computeReducedVar!(ω, Sx, Sy, ind, X, Y, _svec(x)), p)
    _twoStage!(x, _inverseSymKroneckerLoss(Sx, Sy, ind), λ, W, weights!, method)
    _svec(x)
end
