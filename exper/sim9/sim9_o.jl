# Simultaneous confidence intervals for all entries of Δ by the bootstrap, with
# oracle supports, when Ωx = Ωy (hub structured Ωx).
#
#   julia --project=.. sim9_o.jl rep ip dir
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
X = permutedims(rand(dist_X, n))
Y = permutedims(rand(dist_X, n))

Sx = Symmetric( X'X / n )
Sy = Symmetric( Y'Y / n )


# oracle supports: Δ = 0, so every entry is refitted on its own
eS_init = [Int64[] for j=1:div((p + 1)*p, 2)]

@time boot_res, e_s = bootstrap(X, Y; estim_support=eS_init)


@save "$(dir)/res_$(ip)_$(rep).jld2" boot_res
