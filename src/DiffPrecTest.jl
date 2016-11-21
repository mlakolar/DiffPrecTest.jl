module DiffPrecTest


# soft-thersholding operator
# sign(z)(|z|-λ)_+
function SoftThreshold(z::Float64, lambda::Float64)
    abs(z) < lambda ? zero(z) : z > 0. ? z - lambda : z + lambda
en


# computes (Sx*M*Sy)[r,c]
function _mult_aMb(m::SparseMatrixCSC{Float64},
                  Sx::StridedMatrix{Float64},
                  Sy::StridedMatrix{Float64},
                  r::Int64, c::Int64)

  p = size(m, 1)

  colptr = m.colptr
  rowval = m.rowval
  nzval = m.nzval

  a = view(Sx, :, r)
  b = view(Sy, :, c)
  val = 0.
  for j = 1:p
    dval = 0.
    # dval = dot(a, view(m, :, j))
    for k = colptr[j]:(colptr[j+1]-1)
      dval += a[rowval[k]] * nzval[k]
    end
    val += dval * b[j]
  end
  val
end


# finds index to add to the active_set
function add_violating_index!(m::SparseMatrixCSC{Float64},
                         Sx::StridedMatrix{Float64},
                         Sy::StridedMatrix{Float64},
                         a::Int64, b::Int64,
                         λ::Float64)
  p = size(Sx, 2)
  colptr = m.colptr
  rowval = m.rowval

  #
  ia = 0
  ib = 0
  v = 0.

  for c=1:p
    for r=setdiff(1:p, rowval[colptr[c]:(colptr[c+1]-1)])
      t = _mult_aMb(m, Sx, Sy, r, c)
      if (r == a) && (c == b)
        t -= 1
      end
      t = abs(t)
      if t > lambda[j]
        if t > v
          v = t
          ia = r
          ib = c
        end
      end
    end
  end

  if ia != 0
    m[ia, ib] = eps()
  end

  return ia+(ib-1)*p
end



function minimize_active_set!(m::SparseMatrixCSC{Float64},
                         Sx::StridedMatrix{Float64},
                         Sy::StridedMatrix{Float64},
                         a::Int64, b::Int64,
                         λ::Float64;
                         maxIter::Int64=1000, optTol::Float64=1e-7)

  p = size(m, 2)
  colptr = m.colptr
  rowval = m.rowval
  nzval = m.nzval

  iter = 1
  while iter <= maxIter
    fDone = true

    # A = Sx * m * Sy
    A = zeros(p, p)
    for c=1:p
      for r=1:p
        A[r,c] = _mult_aMb(m, Sx, Sy, r, c)
      end
    end

    for c=1:p
      for k=colptr[c]:(colptr[c+1]-1)
        r = rowval[k]

        # compute new value for element (r,c)
        c1 = Sx[r,c] * Sy[r,c]
        c2 = 2 * A[r, c]
        if (r == a) && (c == b)
          c2 -= 1.
        end
        z = SoftThreshold(m[r,c] - c2 / (2*c1), λ/(2*c1))
        h = z - nzval[k]
        nzval[k] = z
      end
    end

    iter = iter + 1
    if fDone
      break
    end
  end

  m = sparse(m)
  nothing

end

# invert Sy⊗Sx by solving a lasso like optimization problem
# this function obtains one row of the inverse indexed by a and b
# in particular a vector m of dimension is returned such that
#     mat( Sy⊗Sx m ) ≈ E_ab
# where E_ab is a matrix with one in position (a,b)
#
# the following optimization problem is solved
#   min_m  (1/2) m' Sy⊗Sx m - m' E_ab + λ |m|_1
#
function invertKronecker!(m::SparseMatrixCSC{Float64},
                         Sx::StridedMatrix{Float64},
                         Sy::StridedMatrix{Float64},
                         a::Int64, b::Int64,
                         λ::Float64;
                         maxIter::Int64=2000, maxInnerIter::Int64=1000, optTol::Float64=1e-7)

  p = size(Sx, 2)
  if nnz(m) == 0
    ind = add_violating_index!(m, Sx, Sy, a, b, λ)
    if ind == 0
      return
    end
  end

  iter = 1
  while iter < maxIter
    minimize_active_set!(m, Sx, Sy, a, b, λ; maxIter=maxInnerIter, optTol=optTol)
    ind = add_violating_index!(m, Sx, Sy, a, b, λ)

    iter = iter + 1;
    if ind == 0
      break
    end
  end

end


end
