using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD


@show gethostname()

pArr = [100, 200, 500]
elemArr = [(5,5), (8, 7), (50, 25), (21, 20), (30, 30)]
n = 300
est      = Array{Any}(undef, 9)   # number of methods

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

Sx = Symmetric( X'X / n )
Sy = Symmetric( Y'Y / n )

@show (ip, iElem)
supportOracle =  LinearIndices(tΔ)[findall(!iszero, tΔ)]
ri, ci = elemArr[iElem]

@time est[1], eΔ, esuppΔ, eω, esuppω, eSupport = DiffPrecTest.estimate(SymmetricNormal(), X, Y, ri, ci; Sx=Sx, Sy=Sy)
@time est[2], _, _, _, _, _ = DiffPrecTest.estimate(AsymmetricNormal(), X, Y, ri, ci; Sx=Sx, Sy=Sy, Δ=eΔ, suppΔ=esuppΔ)
@time est[3] = DiffPrecTest.estimate(SeparateNormal(), X, Y, ri, ci; Sx=Sx, Sy=Sy)
@time est[4] = DiffPrecTest.estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, ri, ci, eSupport)
@time est[5] = DiffPrecTest.estimate(AsymmetricOracleNormal(), Sx, Sy, X, Y, ri, ci, eSupport)

@time est[6] = DiffPrecTest.estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, ri, ci, supportOracle)
@time est[7] = DiffPrecTest.estimate(AsymmetricOracleNormal(), Sx, Sy, X, Y, ri, ci, supportOracle)
@time est[8] = DiffPrecTest.estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, ri, ci, supportOracle)
@time est[9] = DiffPrecTest.estimate(AsymmetricOracleBoot(), Sx, Sy, X, Y, ri, ci, supportOracle)

@save "$(dir)/res_$(ip)_$(iElem)_$(rep).jld" est eΔ esuppΔ eω esuppω eSupport
