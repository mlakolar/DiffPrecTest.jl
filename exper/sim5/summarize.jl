# Support recovery in sim5: false and true positive rates, averaged over the
# repetitions, of the thresholded refits along the thresholds τ and of the
# D-trace estimator along its penalty path, and the rates of the D-trace
# estimator at the penalties chosen on validation data.
#
#   julia --project=.. summarize.jl dir variant
#
# where dir holds the files res_rep.jld2 written by sim5a.jl (variant a) or
# sim5b.jl (variant b). Writes sim5<variant>_roc.csv, sim5<variant>_dtrace.csv
# and sim5<variant>_roc.pdf.
using LinearAlgebra, Statistics
using CovarianceSelection: getConfusionMatrix, tpr, fpr, precision
using DataFrames, CSV
using JLD2
using Plots

include("model.jl")

dir = ARGS[1]
variant = ARGS[2]
p = Dict("a" => 60, "b" => 100)[variant]

Ωx, Ωy = sim5model(p)
Δ = Ωx - Ωy

files = filter(f -> occursin(r"^res_\d+\.jld2$", f), readdir(dir))
NUM_REP = length(files)
@show NUM_REP

numτ = 500
numΛ = 50
FPR_normal = zeros(NUM_REP, numτ)
TPR_normal = zeros(NUM_REP, numτ)
FPR_boot = zeros(NUM_REP, numτ)
TPR_boot = zeros(NUM_REP, numτ)
FPR_tr = zeros(NUM_REP, numΛ)
TPR_tr = zeros(NUM_REP, numΛ)
dtrace = DataFrame(selection=String[], rep=Int[], tpr=Float64[], fpr=Float64[], precision=Float64[])

for (rep, f) in enumerate(files)
    eΔNormal, eΔBoot, eΔDTr, i2, iInf = jldopen(joinpath(dir, f), "r") do file
        read(file, "eΔNormal"), read(file, "eΔBoot"), read(file, "eΔDTr"), read(file, "i2"), read(file, "iInf")
    end

    for j=1:numτ
        cm = getConfusionMatrix(Δ, eΔNormal[j])
        FPR_normal[rep, j] = fpr(cm)
        TPR_normal[rep, j] = tpr(cm)

        cm = getConfusionMatrix(Δ, eΔBoot[j])
        FPR_boot[rep, j] = fpr(cm)
        TPR_boot[rep, j] = tpr(cm)
    end

    for j=1:numΛ
        cm = getConfusionMatrix(Δ, eΔDTr[j])
        FPR_tr[rep, j] = fpr(cm)
        TPR_tr[rep, j] = tpr(cm)
    end

    for (selection, i) in (("frobenius", i2), ("max", iInf))
        cm = getConfusionMatrix(Δ, eΔDTr[i])
        push!(dtrace, (selection, rep, tpr(cm), fpr(cm), precision(cm)))
    end
end

roc = vcat(
    DataFrame(method="Normal", fpr=vec(mean(FPR_normal; dims=1)), tpr=vec(mean(TPR_normal; dims=1))),
    DataFrame(method="Boot", fpr=vec(mean(FPR_boot; dims=1)), tpr=vec(mean(TPR_boot; dims=1))),
    DataFrame(method="DTrace", fpr=vec(mean(FPR_tr; dims=1)), tpr=vec(mean(TPR_tr; dims=1))),
)
CSV.write("sim5$(variant)_roc.csv", roc)
CSV.write("sim5$(variant)_dtrace.csv", combine(groupby(dtrace, :selection), [:tpr, :fpr, :precision] .=> mean))

plt = plot(xlabel="false positive rate", ylabel="true positive rate", legend=:bottomright, title="p = $(p)")
for g in groupby(roc, :method; sort=false)
    plot!(plt, g.fpr, g.tpr; label=first(g.method))
end
savefig(plt, "sim5$(variant)_roc.pdf")
