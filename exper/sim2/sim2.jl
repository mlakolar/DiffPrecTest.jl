using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25)]
n = 300
est      = Array{Any}(undef, 3)   # number of methods


rep   = parse(Int,ARGS[1])
ip    = parse(Int,ARGS[2])
iElem = parse(Int,ARGS[3])
dir = ARGS[4]

Random.seed!(134)

p = pArr[ip]
# generate model
Ω = Matrix{Float64}(I, p, p)
ρ = 0.7
for k=1:p-1
    for l=1:p-k
        Ω[l  , l+k] = ρ^k
        Ω[l+k, l  ] = ρ^k
    end
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

@time est[1], _, _, indS = DiffPrecTest.estimate(SymmetricNormal(), X, Y, indE)
@time est[2] = DiffPrecTest.estimate(SeparateNormal(), X, Y, indE)
@time est[3] = DiffPrecTest.estimate(SymmetricOracleBoot(), X, Y, indS)


@save "$(dir)/res_$(ip)_$(iElem)_$(rep).jld" est
