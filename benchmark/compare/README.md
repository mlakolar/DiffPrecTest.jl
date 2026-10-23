# Against the code before ProximalOPT

`run.jl` times the estimators of DiffPrecTest master before the move onto
ProximalOPT (its own coordinate descent losses, with CovSel master, on
CoordinateDescent 0.6.0 and ProximalBase 0.4.1; `old/`) against this version
(`new/`), on the same problems and seeds. `run.jl` runs one side in a fresh process;
`ab.sh` alternates the sides over a number of rounds, since load on a shared
machine changes over minutes, and `compare.jl` keeps the fastest median of each
case and compares the solutions. From the root of the repository:

```sh
julia --project=benchmark/compare/old -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/compare/new -e 'using Pkg; Pkg.instantiate()'
benchmark/compare/ab.sh 4
```

The reduced estimators (`reducedDiffEstimation`, `invHessianReduced` and
`estimate(ReducedNormal(), ...)`) run on the old side twice: with the default
options of CoordinateDescent 0.6, which accelerate their `CDQuadraticLoss` by
momentum, and with `momentum = false`. The new side uses the default `CDSolver`,
which has the same momentum.

## Results

23 October 2026: four rounds on one core of an AMD EPYC 7513 (USC CARC
Discovery, job 13326859), Julia 1.13.2, one BLAS thread. This version is at
6fe3cf5, on ProximalOPT 1f82ae5 and ProximalBase 0.5.0; times are the fastest
medians over the rounds, in ms.

| case | old | old, momentum = false | new | new / old | largest difference |
|---|---:|---:|---:|---:|---:|
| diffEstimation p=30 | 22.1 | | 21.5 | 0.97 | 3e-08 |
| invAsymHessianEstimation p=30 (2, 1) | 25.3 | | 25.3 | 1.00 | 3e-08 |
| invQSymHessian p=30 (2, 1) | 1.2 | | 0.8 | 0.66 | 3e-09 |
| diffEstimation p=50 | 99.4 | | 97.8 | 0.98 | 3e-08 |
| invAsymHessianEstimation p=50 (2, 1) | 140.1 | | 140.7 | 1.00 | 8e-10 |
| invQSymHessian p=50 (2, 1) | 3.5 | | 2.3 | 0.65 | 7e-09 |
| reducedDiffEstimation p=20 | 198.3 | 200.6 | 91.3 | 0.46 | 1e-07 |
| invHessianReduced p=20 (2, 1) | 30.0 | 30.0 | 14.8 | 0.49 | 1e-07 |
| reducedDiffEstimation p=30 | 637.2 | 639.9 | 302.9 | 0.48 | 1e-07 |
| invHessianReduced p=30 (2, 1) | 67.0 | 67.1 | 33.1 | 0.49 | 3e-08 |
| estimate SymmetricNormal p=30 (2, 1) | 113.7 | | 114.0 | 1.00 | 0 |
| estimate AsymmetricNormal p=30 (2, 1) | 110.0 | | 108.9 | 0.99 | 0 |
| estimate ReducedNormal p=20 (2, 1) | 266.7 | | 123.3 | 0.46 | 0 |
| estimate SeparateNormal p=100 (2, 1) | 0.5 | | 0.5 | 1.02 | 0 |
| supportEstimate DTraceValidationSupport p=50 20λ | 485.9 | | 179.6 | 0.37 | 2e-07 |
| supportEstimate ANTSupport p=20 n=200 | 6350.5 | | 4348.6 | 0.68 | 0 |

No case is slower. Over the 16 cases the geometric mean of new / old is 0.69.
Within each round the ratio is at most 1.04, apart from the 0.5 ms
`SeparateNormal` case, whose ratio varies from 0.90 to 1.15 between rounds.

- **`diffEstimation` and `invAsymHessianEstimation`**: the same time. Their
  weights take most of it, at O(n p³) each.
- **`invQSymHessian`**: a third less, from `KronQuadratic` on a symmetric `Θ`,
  which keeps dense copies of the covariances where the old loss indexed
  `Symmetric` matrices.
- **The reduced estimators**: half the time.
  - Their coordinate descent is about four times faster (0.12 against 0.46 ms
    at p = 20, M3 Pro), with the same momentum as CoordinateDescent 0.6. That
    momentum makes no difference here; see the old code without it.
  - Most of their time goes to the weights. These now compute the entry
    `(row, col)` of each coordinate once instead of in every term.
  - Without that change they were 1.33 to 1.39 times slower than master
    (job 13326410), because ProximalBase 0.5.0's `tril_cartesian` takes 3.4
    instead of 1.2 ns, and `skron` called it twice in every term.
- **`DTraceValidationSupport`**: almost three times faster, mostly from its
  validation losses (CovSel's `diffLoss` now multiplies dense matrices), and
  from the faster solves along the path.
- **The solutions** agree to 2e-7, the tolerance of coordinate descent; the
  refitted estimates agree exactly.

The pull request now pins ProximalOPT to aeed600 and takes ProximalBase 0.5.1
from the registry. Since this run, both have only changed in ways that save
time:
- aeed600 skips the stopping scale of coordinates that did not move;
- 0.5.1 brings back the faster `tril_cartesian`.
