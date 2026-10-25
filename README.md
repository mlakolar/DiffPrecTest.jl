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
  `bootstrap`, `simul_ci`, `simul_ci_studentized`
* Support recovery: `support_estimate` with `ANTSupport`, `BootStdSupport` or
  `DTraceValidationSupport`

DiffPrecTest is one of five Julia packages for high-dimensional statistics,
three for estimation and inference and two for the optimization beneath them:

| Package | |
|---|---|
| [HighDimensionalModels.jl](https://github.com/mlakolar/HighDimensionalModels.jl) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| **DiffPrecTest.jl** (this package) | confidence intervals and tests for the difference of two precision matrices |
| [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.

## Usage

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
straps, = bootstrap(X, Y; boot_samples=500)
CI = simul_ci_studentized(straps, 0.95)
```

`Distributions` also exports `estimate`; when both packages are loaded, call
`DiffPrecTest.estimate`.

The `exper/` folder holds the simulation scripts, with their own environment;
see [`exper/README.md`](exper/README.md).

The penalized estimators of the first and second stages are solved by the
coordinate descent of [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl),
on losses of [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl):
the direct difference loss of [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl)
and the inverse Hessian losses are its `KronQuadratic`. The functions of the
two stages (`diff_estimation`, `inv_q_sym_hessian`, `inv_asym_hessian_estimation`,
`reduced_diff_estimation`, `inv_hessian_reduced`) take the solver options as a
ProximalSolvers `CDSolver`; so are the node-wise lasso regressions of
`SeparateNormal`.

## Naming migration

Types retain `UpperCamelCase`; descriptive functions, fields and keyword
arguments use `snake_case`. Mathematical identifiers remain available.

| Previous spelling | Canonical spelling |
|---|---|
| `diffEstimation` | `diff_estimation` |
| `invAsymHessianEstimation`, `invQSymHessian`, `invHessianReduced` | `inv_asym_hessian_estimation`, `inv_q_sym_hessian`, `inv_hessian_reduced` |
| `supportEstimate` | `support_estimate` |
| `simulCI`, `simulCIstudentized` | `simul_ci`, `simul_ci_studentized` |
| `computeSimulationResult` | `compute_simulation_result` |
| `bootSamples`, `estimSupport` | `boot_samples`, `estim_support` |
| `SimulationResult.lenCoverage` | `SimulationResult.len_coverage` |

Previous public spellings are deprecated aliases for a transition period.
Supplying both spellings of a keyword raises an error. Package versions are
unchanged. Active environments use sibling master checkouts of
CovarianceSelection, ProximalBase and ProximalSolvers through local sources.
The pinned `benchmark/compare/old` environment remains a historical baseline.

## Development tests

From the repository root, run:

```sh
julia scripts/test_master.jl
```

The script prepares an isolated test environment using dependency master
checkouts and test extras on Julia 1.10 and newer. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the naming policy and environment setup.
