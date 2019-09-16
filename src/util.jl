# compute kron(A, B)[Irow, Icol]
function kron_sub!(out, A, B, Irow, Icol)
  @assert size(out, 1) == length(Irow)
  @assert size(out, 2) == length(Icol)

  m, n = size(A)
  p, q = size(B)
  for col=1:length(Icol)
    j = Icol[col]
    ac = div(j-1, q) + 1
    bc = mod(j-1, q) + 1

    for row=1:length(Irow)
        i = Irow[row]
        ar = div(i-1, p) + 1
        br = mod(i-1, p) + 1

        out[row, col] = A[ar, ac] * B[br, bc]
    end
  end
  out
end
kron_sub(A, B, Irow, Icol) = kron_sub!(Matrix{Float64}(undef, length(Irow), length(Icol)), A, B, Irow, Icol)



# compute kron(A, B)[Irow, Icol]
function skron_sub!(out, A, B, Irow, Icol)
  @assert size(out, 1) == length(Irow)
  @assert size(out, 2) == length(Icol)

  m, n = size(A)
  p, q = size(B)
  for col=1:length(Icol)
    j = Icol[col]
    ac1 = div(j-1, q) + 1
    bc1 = mod(j-1, q) + 1
    ac2 = div(j-1, n) + 1
    bc2 = mod(j-1, n) + 1

    for row=1:length(Irow)
        i = Irow[row]
        ar1 = div(i-1, p) + 1
        br1 = mod(i-1, p) + 1
        ar2 = div(i-1, m) + 1
        br2 = mod(i-1, m) + 1

        out[row, col] = ( A[ar1, ac1] * B[br1, bc1] + B[ar2, ac2] * A[br2, bc2] ) / 2.
    end
  end
  out
end
skron_sub(A, B, Irow, Icol) = skron_sub!(Matrix{Float64}(undef, length(Irow), length(Icol)), A, B, Irow, Icol)



function _getElemKron(A::AbstractMatrix, B::AbstractMatrix, row, col)
    m, n = size(A)
    p, q = size(B)

    ac = div(col-1, p) + 1
    bc = mod(col-1, p) + 1
    ar = div(row-1, p) + 1
    br = mod(row-1, p) + 1

    @inbounds A[ar, ac] * B[br, bc]
end

function _getElemKron(S::AbstractMatrix, xi::AbstractVector, row, col)
    p = size(S, 1)

    ac = div(col-1, p) + 1
    bc = mod(col-1, p) + 1
    ar = div(row-1, p) + 1
    br = mod(row-1, p) + 1

    @inbounds S[ar, ac] * xi[br] * xi[bc]
end

function _getElemKron(xi::AbstractVector, S::AbstractMatrix, row, col)
    p = size(S, 1)

    ac = div(col-1, p) + 1
    bc = mod(col-1, p) + 1
    ar = div(row-1, p) + 1
    br = mod(row-1, p) + 1

    @inbounds S[br, bc] * xi[ar] * xi[ac]
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


function getSupport(x::SymmetricSparseIterate, τ::Float64=1e-3)
  p = size(x, 1)

  S = falses(p, p)
  for ci=1:p
      for ri=ci:p
          if abs(x[ri, ci] > τ)
              S[ri, ci] = true
              S[ci, ri] = true
              # S[ri, ri] = true
              # S[ci, ci] = true
          end
      end
  end
  S
end

function getSupport(x::SparseIterate, p::Int, τ::Float64=1e-3)
  S = falses(p, p)
  ind = 0
  for ci=1:p
      for ri=ci:p
          ind += 1
          if abs(x[ind] > τ)
              S[ri, ci] = true
              S[ci, ri] = true
              # S[ri, ri] = true
              # S[ci, ci] = true
          end
      end
  end
  S
end

function getLinearSupport(row::Int, col::Int, S1::BitArray{2}, S2::BitArray{2})
  p = size(S1, 1)

  supp = Vector{Int64}()
  for i=1:p
    for j=1:p
      if S1[i, j] | S2[i,j]
        push!(supp, (j-1)*p + i)
      end
    end
  end
  makeFirst!(supp, p, row, col)
end

function getLinearSupport(row::Int, col::Int, S::BitArray{2})
  p = size(S, 1)

  supp = Vector{Int64}()
  for i=1:p
    for j=1:p
      if S[i, j]
        push!(supp, (j-1)*p + i)
      end
    end
  end
  makeFirst!(supp, p, row, col)
end

function makeFirst!(supp::Vector{Int64}, p, row, col)
  ind = (col - 1) * p + row
  pos = findfirst(isequal(ind), supp)
  if  pos === nothing
      pushfirst!(supp, ind)
  else
      supp[pos], supp[1] = supp[1], supp[pos]
  end

  if row != col
    ind = (row - 1) * p + col
    pos = findfirst(isequal(ind), supp)
    if  pos === nothing
        pushfirst!(supp, ind)
    else
        supp[pos], supp[2] = supp[2], supp[pos]
    end
  end
  supp
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
