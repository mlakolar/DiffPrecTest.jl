using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD


@show gethostname()

rep   = parse(Int,ARGS[1])
ip    = parse(Int,ARGS[2])
dir = ARGS[3]

pArr = [50, 100, 150]
n = 300

p = pArr[ip]
Random.seed!(54298)

# generate model
KK=2
Σx = Matrix{Float64}(I, p, p)
for k = 1:div(p, KK)
  for i = ((k-1)*KK+1):(k*KK-1)
    for j = (i+1):(k*KK)
      Σx[i, j] = 0.8
      Σx[j, i] = 0.8
    end
  end
end
de = abs(minimum(eigvals(Σx))) + 0.05
Σx = (Σx + de*I)/(1+de)
Ωx = inv( Σx )

# generate Delta
d = Vector{Float64}(undef, p)
rand!(Uniform(0.5, 2.5), d)
d .= sqrt.(d)
D = Diagonal(d)
Σ = inv(Symmetric(D * Ωx * D))

dist_X = MvNormal(convert(Matrix, Σ))

# generate data
Random.seed!(12346 + rep)
X = rand(dist_X, n)'
Y = rand(dist_X, n)'

Sx = Symmetric( X'X / n )
Sy = Symmetric( Y'Y / n )


@time boot_res, eS = bootstrap(X, Y)


@save "$(dir)/res_$(ip)_$(rep).jld" boot_res eS
