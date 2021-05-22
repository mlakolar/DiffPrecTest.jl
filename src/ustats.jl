abstract type UStatistics end

########################################

struct OneSampleUstatistics{N} <: UStatistics
  h::Function
end

struct TwoSampleUstatistics{N, M} <: UStatistics
  h::Function
end

######################################

function compute(s::OneSampleUstatistics{1}, X::AbstractMatrix)
  h = s.h
  n = size(X, 1)
  v = 0.
  @inbounds for i=1:n
    v += h(view(X, i, :))
  end

  v / n
end

function compute(s::OneSampleUstatistics{2}, X::AbstractMatrix)
  h = s.h
  n = size(X, 1)
  v = 0.
  @inbounds for i1=1:n
    for i2=i1+1:n
      v += h( view(X, i1, :), 
              view(X, i2, :) )
    end
  end

  2. * v / n / (n-1)
end


function compute(s::TwoSampleUstatistics{1, 1}, X::AbstractMatrix, Y::AbstractMatrix)
  h = s.h
  nx = size(X, 1)
  ny = size(Y, 1)

  v = 0.
  for i=1:nx
    for j=1:ny
      v += h(view(X, i, :), view(Y, j, :))
    end
  end
  v / nx / ny
end

######################################

function variance(s::TwoSampleUstatistics{1, 1}, X::AbstractMatrix, Y::AbstractMatrix)
  h = s.h
  nx = size(X, 1)
  ny = size(Y, 1)

  q = zeros(nx)  
  @inbounds for k=1:nx
    vx = view(X, k, :)
    for j=1:ny
      q[k] += h(vx, view(Y, j, :))
    end
    q[k] /= ny
  end

  r = zeros(ny)
  @inbounds for k=1:ny
    vy = view(Y, k, :)
    for j=1:nx
      r[k] += h(view(X, j, :), vy)
    end
    r[k] /= nx
  end

  t = mean(q)

  σ1 = sum(abs2, q) / (nx - 1) - nx / (nx - 1) * t^2
  σ2 = sum(abs2, r) / (ny - 1) - ny / (ny - 1) * t^2

  return σ1/nx + σ2/ny  
end


######################################

### examples 

function createCovarianceStatistic(row::Int, col::Int)
    @inbounds h(X, Y) = (X[row] - Y[row]) * (X[col] - Y[col]) / 2.    
    OneSampleUstatistics{2}(h)
end

function createSecondMomentStatistic(row::Int, col::Int)
    @inbounds h(X) = X[row] * X[col]
    OneSampleUstatistics{1}(h)
end


function createVarianceReducedEstim(ω, I, Δ, J)
  function h(x, y)
    dot( (skron(x, y, I, J) * Δ + svec(x, x, I) - svec(y, y, I)), ω )
  end
  TwoSampleUstatistics{1, 1}(h)
end


function createVarianceGrad1(Δ::SparseIterate, rowIndex::Int)
  function h(x, y)
    # skron(vx_j, vy_k)[ri, :] * Δ
    out = 0.
    for inz = 1:nnz(Δ)
      indColumn = Δ.nzval2ind[inz]          
      out += skron(x, y, rowIndex, indColumn) * Δ.nzval[inz]
    end  
    out + svec(x, x, rowIndex) - svec(y, y, rowIndex)
  end
  TwoSampleUstatistics{1, 1}(h)
end

function createVarianceGrad2(Δ::SparseIterate, rowIndex::Int, indElem::Int)
  function h(x, y)
    # skron(vx_j, vy_k)[ri, :] * Δ
    out = 0.
    for inz = 1:nnz(Δ)
      indColumn = Δ.nzval2ind[inz]          
      out += skron(x, y, rowIndex, indColumn) * Δ.nzval[inz]
    end  
    out + (rowIndex == indElem ? 1. : 0.)
  end
  TwoSampleUstatistics{1, 1}(h)
end