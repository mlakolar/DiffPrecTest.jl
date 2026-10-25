abstract type UStatistics end

########################################

# U-statistic of order N of one sample with kernel h
struct OneSampleUstatistics{N, F} <: UStatistics
  h::F
end
OneSampleUstatistics{N}(h::F) where {N, F} = OneSampleUstatistics{N, F}(h)

# U-statistic of orders N and M of two samples with kernel h
struct TwoSampleUstatistics{N, M, F} <: UStatistics
  h::F
end
TwoSampleUstatistics{N, M}(h::F) where {N, M, F} = TwoSampleUstatistics{N, M, F}(h)

######################################

"""
    compute(s, X)
    compute(s, X, Y)

Value of the one sample U-statistic `s` on the observations `X` (the elements of
a vector or the rows of a matrix), or of the two sample U-statistic `s` on `X`
and `Y`.
"""
compute(s::UStatistics, X::AbstractArray) = _compute(s, s.h, X)
compute(s::UStatistics, X::AbstractArray, Y::AbstractArray) = _compute(s, s.h, X, Y)


function _compute(::OneSampleUstatistics{1}, h, X::AbstractVector)
  v = zero(eltype(X))
  n = 0
  for x in X
    n += 1
    v += h(x)
  end

  v / n
end

function _compute(::OneSampleUstatistics{1}, h, X::AbstractMatrix)
  n = size(X, 1)
  v = 0.
  @inbounds for i=1:n
    v += h(view(X, i, :))
  end

  v / n
end

function _compute(::OneSampleUstatistics{2}, h, X::AbstractVector)
  n = length(X)
  v = 0.
  @inbounds for i1=1:n
    for i2=i1+1:n
      v += h( X[i1], X[i2] )::Float64
    end
  end

  2. * v / (n * (n-1))
end


function _compute(::OneSampleUstatistics{2}, h, X::AbstractMatrix)
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

function _compute(::TwoSampleUstatistics{1, 1}, h, X::AbstractVector, Y::AbstractVector)
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

function _compute(::TwoSampleUstatistics{1, 1}, h, X::AbstractMatrix, Y::AbstractMatrix)
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

"""
    variance(s, X)
    variance(s, X, Y)

For a one sample U-statistic `s` of order one, the sample variance of the kernel
evaluated at the observations `X`. For a two sample U-statistic `s` of orders
one and one, the estimated variance of `compute(s, X, Y)`.

Methods for the estimators of `Δ` compute the variance of the estimating
equation for one entry; see [`estimate`](@ref).
"""
variance(s::UStatistics, X::AbstractArray) = _variance(s, s.h, X)
variance(s::UStatistics, X::AbstractArray, Y::AbstractArray) = _variance(s, s.h, X, Y)

function _variance(::OneSampleUstatistics{1}, h, X::AbstractVector)
  n = length(X)
  q = map(h, X)
  t = mean(q)
  (sum(abs2, q) - n * t^2) / (n - 1)
end

function _variance(::OneSampleUstatistics{1}, h, X::AbstractMatrix)
  n = size(X, 1)
  q = mapslices(h, X; dims = 2)
  t = mean(q)
  (sum(abs2, q) - n * t^2) / (n - 1)
end

function _variance(::TwoSampleUstatistics{1, 1}, h, X::AbstractMatrix, Y::AbstractMatrix)
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

function create_variance_statistic()
  @inline function h(x, y)
    (x - y)^2 / 2.
  end
  OneSampleUstatistics{2}(h)
end

function create_covariance_statistic(row::Int, col::Int)
  @inline function h(X, Y)
      @inbounds (X[row] - Y[row]) * (X[col] - Y[col]) / 2.
  end
  OneSampleUstatistics{2}(h)
end

function create_second_moment_statistic()
  h(X) = X*X
  OneSampleUstatistics{1}(h)
end

function create_second_moment_statistic(row::Int, col::Int)
  @inbounds h(X) = X[row] * X[col]
  OneSampleUstatistics{1}(h)
end
