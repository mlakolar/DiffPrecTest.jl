# Simulations

Simulation scripts for DiffPrecTest. Each `simN` folder holds the scripts of one
simulation; one run of a script computes one repetition, and the Slurm jobs run
1000 repetitions (100 for sim5).

| Simulation | Question | Model | Scripts |
|---|---|---|---|
| sim1, sim2 | coverage of the confidence interval for one entry, `Δ = 0` | `Ω` banded (sim1), `Ω[i, j] = 0.9^\|i-j\|` (sim2) | `simN.jl`, `summarize.jl`, `plot_results.jl` (sim1) |
| sim3, sim4 | coverage for one entry, sparse `Δ` | `Ωx` banded (sim3), `Ωx[i, j] = 0.7^\|i-j\|` (sim4) | `simN.jl`, `summarize.jl` |
| sim6, sim7 | coverage for one entry, banded `Δ` | banded `Ωx` and `Ωy` | `simN.jl`, `summarize.jl` |
| sim5 | support recovery | block diagonal `Ωx`, diagonal `Ωy` (`model.jl`), `p = 60` (a) and `100` (b) | `sim5a.jl`, `sim5b.jl`, `summarize.jl` |
| sim8–sim11 | simultaneous bootstrap intervals for all entries, `Δ = 0` | banded, hub, random sparse and block `Ωx` | `simN.jl` (estimated supports), `simN_o.jl` (oracle supports), `simN_YinXia.jl` |
| sim8–sim12 | power of the bootstrap test along `Ωy = (1 - α) Ωx + α Ωy₁` | as above; banded `Ωx`, `Ωy` (sim12) | `simN_o_power.jl`, `simN_YinXia_power.jl` |

The coverage simulations compare nine methods: the three stage procedure
(`Sym-N`, `Asym-N`), Xia, Cai and Cai (2015) (`YinXia`), bootstrap standard
errors on the estimated support (`Sym-B`, `Asym-B`), and the same procedures on
the true support (`O-...`).

## Setup

From the repository root, prepare the experiment environment on Julia 1.10
or newer:

```sh
julia scripts/instantiate.jl exper
```

The bootstrap loads DiffPrecTest from this repository and CovarianceSelection,
ProximalSolvers and ProximalBase from sibling master checkouts, including their
transitive workspace dependencies. Existing checkouts are never reset or
silently updated.

## Running

One repetition, from a simulation folder:

```sh
cd sim1
julia --project=.. sim1.jl 1 1 1 /tmp/sim1    # rep ip iElem dir
```

All repetitions on a Slurm cluster, with GNU parallel running one repetition per
task (`./start_all.sh` submits sims 1-7):

```sh
export SCRATCH=/path/to/scratch JULIA=/path/to/julia
cd sim1
./start_sbatch.sh -p <partition>               # arguments go to sbatch
```

Results go to `$SCRATCH/diffTest/<simulation>`. The `simN_YinXia*.jl` scripts
run all repetitions in one process and write to the current folder; set
`NUM_REP` for fewer.

## Summaries

```sh
./collect.sh                                   # sims 1-4, 6, 7 into sim*/results
julia --project=. sim1/summarize.jl            # sim1.csv, sim1_table.tex
julia --project=. sim1/plot_results.jl         # Q-Q plots and histograms
julia --project=. sim5/summarize.jl $SCRATCH/diffTest/sim5a a
julia --project=. sim8/collect_our.jl          # bootstrap coverage, sims 8-11
```

The scripts were first run with Julia 1.5 in 2019–2020. Julia's default random
number generator changed in Julia 1.7, so the same seeds give different data.
