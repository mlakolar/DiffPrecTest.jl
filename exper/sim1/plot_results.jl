using JLD
using DiffPrecTest

include("../qq_plot.jl")

pArr = [100, 200]
elemArr = [(5,5), (8, 7), (50, 25)]
methodArr = ["Sym-N", "Asym-N", "YinXia", "Sym-B", "Asym-B", "O-Sym-N", "O-Asym-N", "O-Sym-B", "O-Asym-B"]

for ip=1
  for iElem=1
    res = load("../sim1_res_$(ip)_$(iElem).jld", "results")

    qq_my     = [res[1, i].p / res[1, i].std for i=1:1000]
    qq_oracle = [res[6, i].p / res[6, i].std for i=1:1000]

    x = range(-3.290, stop=3.290, length=100)
    y = pdf.(Normal(), x)

    fig = figure(figsize=(7.08661, 4.72441), dpi=1000)

    #ax = subplot(2, 3, 1)
    qqplot(qq_oracle, color="grey")
    qqplot(qq_my)
    title("naive re-fitting", size="small")
#    ax[:tick_params]("both", labelsize="xx-small", length=2, pad=2)

    tight_layout()
    savefig("qq_plot_sim1.pdf")
    close(fig)
  end
end
