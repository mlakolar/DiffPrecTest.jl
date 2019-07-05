using JLD
using DiffPrecTest


pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (21, 20), (30, 30)]

for ip=1:2
  for iElem=1:5
    p = pArr[ip]
    Random.seed!(1234)

    # generate model
    Ωx = Matrix{Float64}(I, p, p)
    ρ = 0.7
    for k=1:p-1
      for l=1:p-k
        Ωx[l  , l+k] = ρ^k
        Ωx[l+k, l  ] = ρ^k
      end
    end
    Δ = zeros(p, p)
    # generate Delta
    for j=1:5
      i = 5 + (j-1)
      Δ[i, i] = rand(Uniform(0.1, 0.2))
    end
    for j=1:5
      i = 5 + (j-1)
      v = rand(Uniform(0.2, 0.5))
      Δ[i  , i+1] = v
      Δ[i+1, i  ] = v
    end
    Ωy = Ωx - Δ
    d = Vector{Float64}(undef, p)
    rand!(Uniform(0.5, 2.5), d)
    d .= sqrt.(d)
    D = Diagonal(d)
    Σx = inv(Symmetric(D * Ωx * D))
    Σy = inv(Symmetric(D * Ωy * D))

    tΔ = D * Δ * D

    res = load("../sim4_res_$(ip)_$(iElem).jld", "results")

    @show ip, iElem
    for j in 1:5
      @show computeSimulationResult([res[j, i] for i=1:1000], tΔ[indE])
    end
  end
end
