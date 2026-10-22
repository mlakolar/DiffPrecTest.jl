# Normal Q-Q plots and histograms of the standardized estimates p / std of Sym-N
# (and of O-Sym-N in the Q-Q plots) in sim1, from the files
# results/sim1_res_ip_iElem.jld2 written by collect.sh. Writes qq_plot_sim1.pdf
# and hist_plot_sim1.pdf.
#
#   julia --project=.. plot_results.jl
using DiffPrecTest
using JLD2
using LaTeXStrings

include("../qq_plot.jl")

cd(@__DIR__)

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25)]

standardized(res, j) = [res[j, i].p / res[j, i].std for i=1:size(res, 2)]
panelTitle(ip, iElem) = latexstring("p = $(pArr[ip]), \\Delta_{$(elemArr[iElem][1]),$(elemArr[iElem][2])} = 0")

## qq-plots

panels = []
for ip=1:2
  for iElem=1:3
    res = load("results/sim1_res_$(ip)_$(iElem).jld2", "results")

    plt = plot(title=panelTitle(ip, iElem), titlefontsize=8, tickfontsize=5)
    qqplot!(plt, standardized(res, 6), color=:grey)
    qqplot!(plt, standardized(res, 1))
    push!(panels, plt)
  end
end
savefig(plot(panels...; layout=(2, 3), size=(709, 472)), "qq_plot_sim1.pdf")

## hist-plots

x = range(-3.290, stop=3.290, length=100)
y = pdf.(Normal(), x)

panels = []
for ip=1:2
  for iElem=1:3
    res = load("results/sim1_res_$(ip)_$(iElem).jld2", "results")

    plt = histogram(standardized(res, 1); bins=100, normalize=:pdf, label="",
                    title=panelTitle(ip, iElem), titlefontsize=8, tickfontsize=5)
    plot!(plt, x, y; label="")
    xlims!(plt, -3.290, 3.290)
    push!(panels, plt)
  end
end
savefig(plot(panels...; layout=(2, 3), size=(709, 472)), "hist_plot_sim1.pdf")
