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
for k=1:div(p, 10)
    for j=10*(k-1)+2:10*(k-1)+10
        Ωx[10*(k-1)+1, j] = 0.5
        Ωx[j, 10*(k-1)+1] = 0.5
    end
end
de = abs(minimum(eigvals(Ωx)))+0.05
Ωx = (Ωx + de*I)/(1+de)

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
