# Two Gaussian samples whose precision matrices Ωx and Ωy = Ωx - Δ differ by a
# tridiagonal Δ. Returns X, Y and the true difference.
function generateData(p, n)
  # generate model
  Ωx = Matrix{Float64}(I, p, p)
  for l=1:p-1
    Ωx[l  , l+1] = 0.6
    Ωx[l+1, l  ] = 0.6
  end
  for l=1:p-2
    Ωx[l  , l+2] = 0.3
    Ωx[l+2, l  ] = 0.3
  end
  Δ = zeros(p, p)
  # generate Delta
  for j=1:p
    Δ[j, j] = 0.1 + 0.1 * rand()
  end
  for j=1:p-1
    v = 0.2 + 0.3 * rand()
    Δ[j  , j+1] = v
    Δ[j+1, j  ] = v
  end
  Ωy = Ωx - Δ
  D = Diagonal(sqrt.(0.5 .+ 2. .* rand(p)))
  Ωx = Symmetric(D * Ωx * D)
  Ωy = Symmetric(D * Ωy * D)

  # rows with covariance inv(Ω) = inv(U) inv(U)'
  X = randn(n, p) / cholesky(Ωx).U'
  Y = randn(n, p) / cholesky(Ωy).U'

  X, Y, Matrix(Ωx - Ωy)
end
