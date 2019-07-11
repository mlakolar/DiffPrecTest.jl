# compute kron(A, B)[ind, ind]
function kron_sub!(out, A, B, ind)
  @assert size(out, 1) == size(out, 2) == length(ind)

  m, n = size(A)
  p, q = size(B)
  for col=1:length(ind)
    j = ind[col]
    ac = div(j-1, q) + 1
    bc = mod(j-1, q) + 1

    for row=1:length(ind)
        i = ind[row]

        ar = div(i-1, p) + 1
        br = mod(i-1, p) + 1

        out[row, col] = A[ar, ac] * B[br, bc]
    end
  end
  out
end

function _getElemKron(A::AbstractMatrix, B::AbstractMatrix, row, col)
    m, n = size(A)
    p, q = size(B)

    ac = div(col-1, p) + 1
    bc = mod(col-1, p) + 1
    ar = div(row-1, p) + 1
    br = mod(row-1, p) + 1

    @inbounds A[ar, ac] * B[br, bc]
end


function _getElemSKron(S::AbstractMatrix, xi::AbstractVector, row, col)
    p = size(S, 1)

    ac = div(col-1, p) + 1
    bc = mod(col-1, p) + 1
    ar = div(row-1, p) + 1
    br = mod(row-1, p) + 1

    @inbounds ( S[ar, ac] * xi[br] * xi[bc] + S[br, bc] * xi[ar] * xi[ac] ) / 2.
end

_getElemSKron(A::AbstractMatrix, B::AbstractMatrix, row, col) =
    ( _getElemKron(A, B, row, col) + _getElemKron(B, A, row, col) ) / 2.





# computes variance of Var(Sx - Sy)[indS, indS]
# where indS is a list of elements in [1:p, 1:p]
function _varSxSy(Sx, nx, Sy, ny, indS)

    varS = zeros(length(indS), length(indS))   # var(Sx) + var(Sy)

    I = CartesianIndices(Sx)
    for ia = 1:length(indS)
      a = indS[ia]
      for ib = 1:length(indS)
        b = indS[ib]

        i, j = Tuple( I[a] )
        k, l = Tuple( I[b] )

        varS[ia, ib] = (Sx[i, k] * Sx[j, l] + Sx[i, l] * Sx[j, k]) / (nx - 1) + (Sy[i, k] * Sy[j, l] + Sy[i, l] * Sy[j, k]) / (ny - 1)
      end
    end

    varS
end


function computeSimulationResult(res::Vector{DiffPrecResultNormal}, trueParam::Float64, α::Float64=0.05)
  Eestim = 0.
  bias = 0.
  coverage = 0
  lenCoverage = 0.

  num_rep = length(res)
  zα = quantile(Normal(), 1. - α / 2.)

  for rep=1:num_rep
    if res[rep].p - zα * res[rep].std < trueParam && res[rep].p + zα * res[rep].std > trueParam
      coverage += 1
    end
    lenCoverage += 2. * zα * res[rep].std
    Eestim += res[rep].p
  end

  bias = Eestim / num_rep - trueParam

  SimulationResult(bias, coverage / num_rep, lenCoverage / num_rep)
end

function computeSimulationResult(res::Vector{DiffPrecResultBoot}, trueParam::Float64, α::Float64=0.05)
  Eestim = 0.
  bias = 0.
  coverage = 0
  lenCoverage = 0.

  num_rep = length(res)
  z1 = α / 2.
  z2 = 1. - α / 2.

  for rep=1:num_rep
    lb, ub = quantile!(res[rep].boot_p, [z1, z2])
    if lb < trueParam && ub > trueParam
      coverage += 1
    end
    lenCoverage += (ub - lb)
    Eestim += res[rep].p
  end

  bias = Eestim / num_rep - trueParam

  SimulationResult(bias, coverage / num_rep, lenCoverage / num_rep)
end
