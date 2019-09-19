using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD


@show gethostname()

rep   = parse(Int,ARGS[1])
ip    = parse(Int,ARGS[2])
iElem = parse(Int,ARGS[3])
dir = ARGS[4]

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25), (22, 20), (32, 30)]
n = 300
est      = Array{Any}(undef, 9)   # number of methods

p = pArr[ip]
Random.seed!(7689)

# generate model
Ωx = Matrix{Float64}(I, p, p)
for l=1:p-1
  Ωx[l  , l+1] = 0.3
  Ωx[l+1, l  ] = 0.3
end
for l=1:p-2
  Ωx[l  , l+2] = 0.2
  Ωx[l+2, l  ] = 0.2
end
Ωy = Matrix{Float64}(I, p, p) * 1.1
for l=1:p-1
  Ωy[l  , l+1] = 0.3
  Ωy[l+1, l  ] = 0.3
end
for l=1:p-2
  Ωy[l  , l+2] = -0.1
  Ωy[l+2, l  ] = -0.1
end
Δ = Ωx - Ωy
# generate Delta
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
@time est[2], _, _, eωa, esuppωa, eSupporta = DiffPrecTest.estimate(AsymmetricNormal(), X, Y, ri, ci; Sx=Sx, Sy=Sy, Δ=eΔ, suppΔ=esuppΔ)
@time est[3] = DiffPrecTest.estimate(SeparateNormal(), X, Y, ri, ci; Sx=Sx, Sy=Sy)
@time est[4] = DiffPrecTest.estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, ri, ci, eSupport)
@time est[5] = DiffPrecTest.estimate(AsymmetricOracleBoot(), Sx, Sy, X, Y, ri, ci, eSupporta)

@time est[6] = DiffPrecTest.estimate(SymmetricOracleNormal(), Sx, Sy, X, Y, ri, ci, supportOracle)
@time est[7] = DiffPrecTest.estimate(AsymmetricOracleNormal(), Sx, Sy, X, Y, ri, ci, supportOracle)
@time est[8] = DiffPrecTest.estimate(SymmetricOracleBoot(), Sx, Sy, X, Y, ri, ci, supportOracle)
@time est[9] = DiffPrecTest.estimate(AsymmetricOracleBoot(), Sx, Sy, X, Y, ri, ci, supportOracle)

@save "$(dir)/res_$(ip)_$(iElem)_$(rep).jld" est eΔ esuppΔ eω esuppω eSupport
