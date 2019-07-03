using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25)]
n = 300
est      = Array{DiffPrecResultNormal}(undef, 2)   # number of methods


rep   = parse(Int,ARGS[1])
ip    = parse(Int,ARGS[2])
iElem = parse(Int,ARGS[3])
dir = ARGS[4]

Random.seed!(134)

p = pArr[ip]
# generate model
Ω = Matrix{Float64}(I, p, p)
for l=1:p-1
    Ω[l  , l+1] = 0.6
    Ω[l+1, l  ] = 0.6
end
for l=1:p-2
    Ω[l  , l+2] = 0.3
    Ω[l+2, l  ] = 0.3
end
d = Vector{Float64}(undef, p)
rand!(Uniform(0.5, 2.5), d)
d .= sqrt.(d)
D = Diagonal(d)
Σ = inv(Symmetric(D * Ω * D))

dist_X = MvNormal(convert(Matrix, Σ))
dist_Y = MvNormal(convert(Matrix, Σ))

# generate data
Random.seed!(134 + rep)
X = rand(dist_X, n)'
Y = rand(dist_Y, n)'

@show (ip, iElem)

ri, ci = elemArr[iElem]
indE = (ci - 1) * p + ri

@time est[1], _, _, _ = DiffPrecTest.estimate(SymmetricNormal(), X, Y, indE)
@time est[2] = DiffPrecTest.estimate(SeparateNormal(), X, Y, indE)


@save "$(dir)/res_$(ip)_$(iElem)_$(rep).jld" est
