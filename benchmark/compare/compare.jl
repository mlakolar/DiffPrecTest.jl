# Compares the runs of run.jl on the sides: for every case, the fastest median
# time over the runs of each side (load on the machine only adds time) and the
# largest difference of the solutions; for the reduced estimators also the old
# code without momentum, and the side newfix when it was run.
#
#     julia benchmark/compare/compare.jl

using Printf, Serialization

dir = joinpath(@__DIR__, "results")
load(side) = [deserialize(joinpath(dir, f)) for f in readdir(dir) if startswith(f, side * "_")]
olds, news, fixes = load("old"), load("new"), load("newfix")
fastest(runs, key) = minimum(r[key].time for r in runs if haskey(r, key))
fix = !isempty(fixes)

cases = sort!(unique(first.(collect(keys(first(news))))); by = c -> first(news)[(c, "new")].order)
print("| case | old (ms) | old, momentum = false (ms) | new (ms) | new / old | new / old without momentum |")
fix && print(" newfix (ms) | newfix / old |")
println(" largest difference |")
println("|---|---:|---:|---:|---:|---:|", fix ? "---:|---:|" : "", "---:|")
ratios, fixratios = Float64[], Float64[]
for c in cases
  haskey(first(olds), (c, "old")) || continue
  told, tnew = fastest(olds, (c, "old")), fastest(news, (c, "new"))
  nomom = haskey(first(olds), (c, "old, momentum = false"))
  tnm = nomom ? fastest(olds, (c, "old, momentum = false")) : NaN
  o, nw = first(olds)[(c, nomom ? "old, momentum = false" : "old")].solution, first(news)[(c, "new")].solution
  d = length(o) == length(nw) ? maximum(abs, o - nw; init = 0.0) : NaN
  push!(ratios, tnew / told)
  @printf("| %s | %.1f | %s | %.1f | %.2f | %s |", c, 1e3told, nomom ? @sprintf("%.1f", 1e3tnm) : "", 1e3tnew,
          tnew / told, nomom ? @sprintf("%.2f", tnew / tnm) : "")
  if fix
    tfix = fastest(fixes, (c, "new"))
    push!(fixratios, tfix / told)
    @printf(" %.1f | %.2f |", 1e3tfix, tfix / told)
  end
  @printf(" %.0e |\n", d)
end
summary(r) = @sprintf("geometric mean %.2f, range %.2f to %.2f", exp(sum(log, r) / length(r)), minimum(r), maximum(r))
@printf("\n%d cases, %d rounds of each side; new / old: %s\n", length(ratios), min(length(olds), length(news)), summary(ratios))
fix && println("newfix / old: ", summary(fixratios))
