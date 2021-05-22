using JLD
using DiffPrecTest
using LaTeXStrings

include("../qq_plot.jl")

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25)]
methodArr = ["Sym-N", "Asym-N", "YinXia", "Sym-B", "Asym-B", "O-Sym-N", "O-Asym-N", "O-Sym-B", "O-Asym-B"]

## qq-plots

fig_num = 0

fig = figure(figsize=(7.08661, 4.72441), dpi=1000)
for ip=1:2
  for iElem=1:3     
    global fig_num

    res = load("results/sim1_res_$(ip)_$(iElem).jld", "results")
    fig_num += 1

    qq_my     = [res[1, i].p / res[1, i].std for i=1:1000]
    qq_oracle = [res[6, i].p / res[6, i].std for i=1:1000]

    ax = subplot(2, 3, fig_num)
    qqplot(qq_oracle, color="grey")
    qqplot(qq_my)
    title(
     latexstring("p = $(pArr[ip]), \\Delta_{$(elemArr[iElem][1]),$(elemArr[iElem][2])} = 0"),
     size="small"
    )
    ax[:tick_params]("both", labelsize="xx-small", length=2, pad=2)    
  end
end
tight_layout()
savefig("qq_plot_sim1.pdf")
close(fig)

## hist-plots

fig_num = 0

x = range(-3.290, stop=3.290, length=100)
y = pdf.(Normal(), x)

fig = figure(figsize=(7.08661, 4.72441), dpi=1000)
for ip=1:2
  for iElem=1:3     
    global fig_num
    
    res = load("results/sim1_res_$(ip)_$(iElem).jld", "results")
    fig_num += 1   
    
    ax = subplot(2, 3, fig_num)
    plt[:hist]([res[1, i].p / res[1, i].std for i=1:1000], 100, density=true)
    plot(x,y)
    xlim(-3.290,3.290)
    title(
     latexstring("p = $(pArr[ip]), \\Delta_{$(elemArr[iElem][1]),$(elemArr[iElem][2])} = 0"),
     size="small"
    )
    ax[:tick_params]("both", labelsize="xx-small", length=2, pad=2)
  end
end
tight_layout()
savefig("hist_plot_sim1.pdf")
close(fig)


## irina plots