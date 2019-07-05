using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (21, 20), (30, 30)]
n = 300
est      = Array{Any}(undef, 5)   # number of methods

rep   = parse(Int,ARGS[1])
ip    = parse(Int,ARGS[2])
iElem = parse(Int,ARGS[3])
dir = ARGS[4]

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

dist_X = MvNormal(convert(Matrix, Σx))
dist_Y = MvNormal(convert(Matrix, Σy))

# generate data
Random.seed!(1234 + rep)
X = rand(dist_X, n)'
Y = rand(dist_Y, n)'

@show (ip, iElem)

ri, ci = elemArr[iElem]
indE = (ci - 1) * p + ri
indOracle =  LinearIndices(tΔ)[findall(!iszero, tΔ)]

@time est[1], _, _, indS = DiffPrecTest.estimate(SymmetricNormal(), X, Y, indE)
@time est[2] = DiffPrecTest.estimate(SeparateNormal(), X, Y, indE)
@time est[3] = DiffPrecTest.estimate(SymmetricOracleBoot(), X, Y, indS)
@time est[4] = DiffPrecTest.estimate(SymmetricOracleNormal(), Symmetric(cov(X)), n, Symmetric(cov(Y)), n, indOracle)
@time est[5] = DiffPrecTest.estimate(SymmetricOracleBoot(), X, Y, indOracle)



@save "$(dir)/res_$(ip)_$(iElem)_$(rep).jld" est
