using Revise
using DiffPrecTest
using Statistics, StatsBase, LinearAlgebra
using SparseArrays
using ProximalBase, CoordinateDescent, CovSel
using Random, Distributions
using JLD

#using Plots

pArr = [5, 200]
elemArr = [(5,5), (3, 2), (50, 25), (21, 20), (30, 30)]
n = 300
est      = Array{Any}(undef, 5)   # number of methods

out = Vector{DiffPrecResultNormal}(undef, 1000)

ip = 1
iElem = 2

p = pArr[ip]
Random.seed!(1234)

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
Δ = zeros(p, p)
# generate Delta
for j=1:p
    #i = 5 + (j-1)
    i = j
    Δ[i, i] = rand(Uniform(0.1, 0.2))
end
for j=1:p-1
    #i = 5 + (j-1)
    i = j
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

ri, ci = elemArr[iElem]
indE = sub2indLowerTriangular(p, ri, ci)

indOracle = Vector{Int64}()
for ci=1:p
  for ri=ci:p    
    global indOracle    
    
    if !iszero(tΔ[ri, ci])
      push!(indOracle, sub2indLowerTriangular(p, ri, ci))
    end
  end
end


Random.seed!(1234)
X = rand(dist_X, n)'
Y = rand(dist_Y, n)'

@time res1, _ = DiffPrecTest.estimate(ReducedNormal(), X, Y, indE)

@time res2, _ = DiffPrecTest.estimate(SymmetricNormal(), X, Y, 3, 2)

for rep = 1:1000
  global out
  global indOracle
  global indE

  # generate data
  Random.seed!(1234 + rep)
  X = rand(dist_X, n)'
  Y = rand(dist_Y, n)'

  #@time out[rep], _, _, _, _, indS = DiffPrecTest.estimate(SymmetricNormal(), X, Y, ri, ci)
  
  @time out[rep] = DiffPrecTest.estimate(ReducedOracleNormal(), Symmetric(cov(X)), Symmetric(cov(Y)), X, Y, indE, indOracle)
end

computeSimulationResult(out, tΔ[elemArr[iElem]...])

true_v = tΔ[elemArr[iElem]...]
histogram( [(out[i].p - true_v) / out[i].std  for i = 1:1000] )



# @time est[2] = DiffPrecTest.estimate(SeparateNormal(), X, Y, indE)
# @time est[3] = DiffPrecTest.estimate(SymmetricOracleBoot(), X, Y, indS)
# @time est[4] = DiffPrecTest.estimate(SymmetricOracleNormal(), Symmetric(cov(X)), n, Symmetric(cov(Y)), n, indOracle)
# @time est[5] = DiffPrecTest.estimate(SymmetricOracleBoot(), X, Y, indOracle)




######################
# test variance

Random.seed!(1234+5)
X = rand(dist_X, n)'
Y = rand(dist_Y, n)'

Sx = Symmetric(cov(X))
Sy = Symmetric(cov(Y))

indS = indOracle

C = DiffPrecTest.skron(Sx, Sy, indS)
b = DiffPrecTest.svec(Sy, indS) - DiffPrecTest.svec(Sx, indS)

tmp = zeros(length(indS))
tmp[1] = 1.
ω = C \ tmp
Δ = C \ b

@time v = variance(ReducedOracleNormal(), Sx, Sy, X, Y, ω, indS, Δ, indS)

@time variance(DiffPrecTest.createVarianceReducedEstim(ω, indS, Δ, indS), X, Y)

#########################
# test gradient variance

A = DiffPrecTest.skron(Sx, Sy)
b = DiffPrecTest.svec(Sx) - DiffPrecTest.svec(Sy)

f = CDQuadraticLoss(A, b)
x = SparseIterate(size(A, 1))

x[1] = 0.5
x[10] = -0.3
x[4] = 2.

@time v = variance(DiffPrecTest.createVarianceGrad1(x, 2), X, Y)

ω = Array{eltype(Sx)}(undef, length(x))
DiffPrecTest._computeVarStep1!(ω, X, Y, x)

ω[2] - sqrt(v)


#########################
# test gradient variance

indElem = 2
A = DiffPrecTest.skron(Sx, Sy)
b = zeros(size(A, 1))
b[indElem] = 1.

f = CDQuadraticLoss(A, b)
x = SparseIterate(size(A, 1))

x[1] = 0.5
x[10] = -0.3
x[4] = 2.

ω = Array{eltype(Sx)}(undef, length(x))
DiffPrecTest._computeVarStep2!(ω, indElem, X, Y, x)

ω[1] - sqrt( variance(DiffPrecTest.createVarianceGrad2(x, 1, indElem), X, Y) )
ω[2] - sqrt( variance(DiffPrecTest.createVarianceGrad2(x, 2, indElem), X, Y) )


