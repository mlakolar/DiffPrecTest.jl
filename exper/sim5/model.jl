# Model of sim5: Ωx is block diagonal with banded blocks of sizes p/2, p/4 and
# p/4 scaled by 1, 2 and 4, and Ωy is the diagonal of Ωx, so that Δ = Ωx - Ωy
# is the off-diagonal part of Ωx. p must be divisible by 4.
function sim5model(p)
  Ωx = Matrix{Float64}(I, p, p)
  mf = [1., 0.5, 0.4]
  for (α, bp, ep) in ((1., 1, div(p, 2)), (2., div(p, 2) + 1, div(3p, 4)), (4., div(3p, 4) + 1, p))
    for k=0:2
      v = α * mf[k+1]
      for l=bp:ep-k
        Ωx[l  , l+k] = v
        Ωx[l+k, l  ] = v
      end
    end
  end
  Ωy = Matrix(Diagonal(Ωx))
  Ωx, Ωy
end
