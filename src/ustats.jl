abstract type UStatistics end

########################################

struct OneSampleUstatistics{N} <: UStatistics
  h
end

struct TwoSampleUstatistics{N, M} <: UStatistics
  h
end

######################################

compute(s::UStatistics, X::AbstractArray) = compute(s, s.h, X)
compute(s::UStatistics, X::AbstractArray, Y::AbstractArray) = compute(s, s.h, X, Y)


function compute(::OneSampleUstatistics{1}, h, X::AbstractVector)
  v = zero(eltype(X))
  n = 0
  for x in X
    n += 1
    v += h(x)
  end

  v / n
end

function compute(::OneSampleUstatistics{1}, h, X::AbstractMatrix)
  n = size(X, 1)
  v = 0.
  @inbounds for i=1:n
    v += h(view(X, i, :))
  end

  v / n
end

function compute(::OneSampleUstatistics{2}, h, X::AbstractVector)
  n = length(X)
  v = 0.
  @inbounds for i1=1:n
    for i2=i1+1:n
      v += h( X[i1], X[i2] )::Float64
    end
  end

  2. * v / (n * (n-1))
end


function compute(::OneSampleUstatistics{2}, h, X::AbstractMatrix)
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

function compute(::TwoSampleUstatistics{1, 1}, h, X::AbstractVector, Y::AbstractVector)
  nx = length(X)
  ny = length(Y)

  v = 0.
  for i=1:nx
    for j=1:ny
      v += h(X[i], Y[j])
    end
  end
  v / nx / ny
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

variance(s::UStatistics, X::AbstractArray) = variance(s, s.h, X)
variance(s::UStatistics, X::AbstractArray, Y::AbstractArray) = variance(s, s.h, X, Y)

function variance(::OneSampleUstatistics{1}, h, X::AbstractVector)
  n = length(X)
  q = map(h, X)
  t = mean(q)
  (sum(abs2, q) - n * t^2) / (n - 1)
end

function variance(::OneSampleUstatistics{1}, h, X::AbstractMatrix)
  n = size(X, 1)
  q = mapslices(h, X; dims = 2)
  t = mean(q)
  (sum(abs2, q) - n * t^2) / (n - 1)
end

function variance(::TwoSampleUstatistics{1, 1}, h, X::AbstractMatrix, Y::AbstractMatrix)
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

function createVarianceStatistic()
  @inline function h(x, y) 
    (x - y)^2 / 2.    
  end
  OneSampleUstatistics{2}(h)
end

function createCovarianceStatistic(row::Int, col::Int)
  @inline function h(X, Y) 
      @inbounds (X[row] - Y[row]) * (X[col] - Y[col]) / 2.    
  end
  OneSampleUstatistics{2}(h)
end

function createSecondMomentStatistic()
  h(X) = X*X
  OneSampleUstatistics{1}(h)
end

function createSecondMomentStatistic(row::Int, col::Int)
  @inbounds h(X) = X[row] * X[col]
  OneSampleUstatistics{1}(h)
end