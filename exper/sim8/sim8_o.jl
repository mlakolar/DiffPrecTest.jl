# Simultaneous confidence intervals for all entries of Δ by the bootstrap, with
# oracle supports, when Ωx = Ωy (banded Ωx).
#
#   julia --project=.. sim8_o.jl rep ip dir
#
# saves the bootstrap estimates to dir/res_ip_rep.jld2
using DiffPrecTest
using LinearAlgebra, Random
using Distributions
using JLD2


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
X = permutedims(rand(dist_X, n))
Y = permutedims(rand(dist_X, n))

Sx = Symmetric( X'X / n )
Sy = Symmetric( Y'Y / n )

# oracle supports: Δ = 0, so every entry is refitted on its own
eS_init = [Int64[] for j=1:div((p + 1)*p, 2)]

@time boot_res, eS = bootstrap(X, Y; estimSupport=eS_init)

@save "$(dir)/res_$(ip)_$(rep).jld2" boot_res
