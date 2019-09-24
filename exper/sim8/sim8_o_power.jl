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
Ωx = Matrix{Float64}(I, p, p)
for l=1:p-1
  Ωx[l  , l+1] = 0.6
  Ωx[l+1, l  ] = 0.6
end
for l=1:p-2
  Ωx[l  , l+2] = 0.3
  Ωx[l+2, l  ] = 0.3
end







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









eS_init = Array{BitArray}(undef, div((p + 1)*p, 2))
for j=1:div((p + 1)*p, 2)
    eS_init[j] = falses(p, p)
end

@time boot_res, eS = bootstrap(X, Y; estimSupport=eS_init)

@save "$(dir)/res_$(ip)_$(rep).jld" boot_res
