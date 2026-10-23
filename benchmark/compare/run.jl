# The estimators of DiffPrecTest before and after the move onto ProximalSolvers,
# run by the version in the active environment: old/ (DiffPrecTest and CovSel
# master with their own coordinate descent losses, on CoordinateDescent 0.6.0
# and ProximalBase 0.4.1) or new/ (this checkout, on ProximalSolvers 0.1 and
# ProximalBase 0.5). The data come from Random and LinearAlgebra only, so that
# both sides solve the same problems. The settings of a case (old: the default
# options of CoordinateDescent 0.6, and for the reduced estimators, whose
# CDQuadraticLoss it accelerates by momentum, also momentum = false; new: the
# default CDSolver) run in turn, after one run each to compile, until each has
# at least 5 runs and half a second of them, each from the same seed. Writes
# the median times and the solutions to results/<side>_<stamp>.jls;
# compare.jl compares the two sides.
#
#     julia --project=benchmark/compare/old benchmark/compare/run.jl old [regex]
#     julia --project=benchmark/compare/new benchmark/compare/run.jl new [regex]

using DiffPrecTest, ProximalBase, LinearAlgebra, Printf, Random, Serialization, SparseArrays, Statistics
BLAS.set_num_threads(1)

const SIDE = ARGS[1]
const OLD = SIDE == "old"
const ONLY = Regex(length(ARGS) > 1 ? ARGS[2] : ".")
const SEED = 20261023

if OLD
  using CoordinateDescent
  const DEFAULT = [("old", CDOptions())]
  const MOMENTUM = [("old", CDOptions()), ("old, momentum = false", CDOptions(; momentum=false))]
else
  using ProximalSolvers
  const DEFAULT = [("new", CDSolver())]
  const MOMENTUM = DEFAULT
end
# the functions with no solver options run with the defaults
const NONE = [(OLD ? "old" : "new", nothing)]

# Two samples of n observations, X from N(0, I) and Y from N(0, Σy) with the
# AR(1) covariance Σy, ρ = 0.7, and their second moments, as DiffPrecTest's
# estimators compute them; Δ = Ωx - Ωy is tridiagonal.
function difference_problem(p; n = 500, seed = p)
  rng = Xoshiro(seed)
  Σy = [0.7^abs(i - j) for i=1:p, j=1:p]
  X = randn(rng, n, p)
  Y = randn(rng, n, p) * cholesky(Σy).U
  (X = X, Y = Y, Sx = Symmetric(X'X / n), Sy = Symmetric(Y'Y / n))
end

const results = Dict{Tuple{String, String}, Any}()

function run_case(name, settings, f)
  occursin(ONLY, name) || return
  runs = [() -> (Random.seed!(SEED); f(o)) for (_, o) in settings]
  for r in runs
    r()
  end
  times = [Float64[] for _ in settings]
  sols = Vector{Any}(undef, length(settings))
  enough(j) = length(times[j]) >= 5 && sum(times[j]) >= 0.5
  while !all(enough, eachindex(settings))
    for (j, r) in enumerate(runs)
      enough(j) && continue
      t = @elapsed sols[j] = r()
      push!(times[j], t)
    end
  end
  for (j, (label, _)) in enumerate(settings)
    results[(name, label)] = (order = length(results), time = median(times[j]), solution = sols[j])
    @printf("%-50s %-24s %10.2f ms\n", name, label, 1e3 * median(times[j]))
  end
end

# the first and second stages: two solves each, before and after recomputing
# the weights of the penalty
function stage_cases()
  for p in (30, 50)
    P = difference_problem(p)
    X, Y, Sx, Sy = P.X, P.Y, P.Sx, P.Sy
    λ = DiffPrecTest._penaltyLevel(p)
    run_case("diffEstimation p=$p", DEFAULT, o -> vec(Matrix(diffEstimation(Sx, Sy, X, Y, λ, o))))
    run_case("invAsymHessianEstimation p=$p (2, 1)", DEFAULT, o -> Vector(invAsymHessianEstimation(Sx, Sy, 2, 1, X, Y, λ, o)))
    run_case("invQSymHessian p=$p (2, 1)", DEFAULT, o -> Vector(invQSymHessian(Sx, Sy, 2, X, Y, λ, o)))
  end
  for p in (20, 30)
    P = difference_problem(p)
    X, Y, Sx, Sy = P.X, P.Y, P.Sx, P.Sy
    λ = DiffPrecTest._penaltyLevel(p)
    H = Symmetric(DiffPrecTest.skron(Sx, Sy))
    b = DiffPrecTest.svec(Sx) - DiffPrecTest.svec(Sy)
    run_case("reducedDiffEstimation p=$p", MOMENTUM, o -> Vector(DiffPrecTest.reducedDiffEstimation(H, b, Sx, Sy, X, Y, λ, o)))
    run_case("invHessianReduced p=$p (2, 1)", MOMENTUM, o -> Vector(invHessianReduced(H, 2, Sx, Sy, X, Y, λ, o)))
  end
end

# confidence intervals for Δ[2, 1]: the three stages, and the node-wise lasso
function estimate_cases()
  P = difference_problem(30)
  result(r) = [r.p, r.std]
  run_case("estimate SymmetricNormal p=30 (2, 1)", NONE, o -> result(estimate(SymmetricNormal(), P.X, P.Y, 2, 1)[1]))
  run_case("estimate AsymmetricNormal p=30 (2, 1)", NONE, o -> result(estimate(AsymmetricNormal(), P.X, P.Y, 2, 1)[1]))
  P = difference_problem(20)
  run_case("estimate ReducedNormal p=20 (2, 1)", NONE, o -> result(estimate(ReducedNormal(), P.X, P.Y, 2)[1]))
  P = difference_problem(100)
  run_case("estimate SeparateNormal p=100 (2, 1)", NONE, o -> result(estimate(SeparateNormal(), P.X, P.Y, 2, 1)))
end

function support_cases()
  p = 50
  P, V = difference_problem(p), difference_problem(p; seed = 1000 + p)
  λs = maximum(abs, P.Sy - P.Sx) .* 10 .^ range(0, -1.3; length=20)
  run_case("supportEstimate DTraceValidationSupport p=50 20λ", DEFAULT,
           o -> reduce(vcat, vec.(Matrix.(supportEstimate(DTraceValidationSupport(), P.Sx, P.Sy, V.Sx, V.Sy, λs, o)[1]))))
  P = difference_problem(20; n = 200)
  run_case("supportEstimate ANTSupport p=20 n=200", NONE,
           o -> (Δpoint = supportEstimate(ANTSupport(), P.X, P.Y, 2.0)[2]; [[d.p for d in Δpoint]; [d.std for d in Δpoint]]))
end

stage_cases()
estimate_cases()
support_cases()

mkpath(joinpath(@__DIR__, "results"))
file = joinpath(@__DIR__, "results", "$(SIDE)_$(round(Int, time())).jls")
serialize(file, results)
println("wrote ", file)
