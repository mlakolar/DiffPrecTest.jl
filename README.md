# DiffPrecTest.jl

[![CI](https://github.com/mlakolar/DiffPrecTest.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/DiffPrecTest.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/DiffPrecTest.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/DiffPrecTest.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Inference for the difference `Δ = Ωx - Ωy` of the precision matrices of two
independent Gaussian samples.

* Confidence intervals for one entry of `Δ`: `estimate` with
  * `SymmetricNormal`: estimate `Δ` by the weighted ℓ1 penalized direct
    difference estimator, estimate the corresponding row of the inverse Hessian,
    and refit on the union of the two supports; `AsymmetricNormal` and
    `ReducedNormal` are variants
  * `SeparateNormal`: estimate `Ωx` and `Ωy` separately by node-wise lasso
    regressions (Xia, Cai and Cai, 2015)
  * refits on a given support: `SymmetricOracleNormal`, `AsymmetricOracleNormal`,
    `ReducedOracleNormal`, and with bootstrap standard errors
    `SymmetricOracleBoot`, `AsymmetricOracleBoot`
* Simultaneous confidence intervals for all entries by the bootstrap:
  `bootstrap`, `simulCI`, `simulCIstudentized`
* Support recovery: `supportEstimate` with `ANTSupport`, `BootStdSupport` or
  `DTraceValidationSupport`

```julia
using DiffPrecTest, LinearAlgebra

p, n = 10, 500
Ωx = Matrix(SymTridiagonal(fill(1., p), fill(0.4, p - 1)))
Ωy = Matrix(SymTridiagonal(fill(1., p), fill(0.2, p - 1)))
X = randn(n, p) / cholesky(Ωx).U'     # rows from N(0, inv(Ωx))
Y = randn(n, p) / cholesky(Ωy).U'

# 95% confidence interval for Δ[2, 1] = 0.2
res, = estimate(SymmetricNormal(), X, Y, 2, 1)
res.p .+ [-1.96, 1.96] .* res.std

# simultaneous 95% confidence intervals for the entries of the lower triangle of Δ;
# the off-diagonal entries are multiplied by √2
straps, = bootstrap(X, Y; bootSamples=500)
CI = simulCIstudentized(straps, 0.95)
```

`Distributions` also exports `estimate`; when both packages are loaded, call
`DiffPrecTest.estimate`.

The `exper/` folder holds the simulation scripts, with their own environment;
see [`exper/README.md`](exper/README.md).

Package depends on [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl),
[CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl) and
[CovSel.jl](https://github.com/mlakolar/CovSel.jl).
