using Distributions, StatsBase
using PyPlot


function qqplot(z; color="black")
    n = length(z)

    grid = [(1 / (n + 1)):(1 / (n + 1)):(1.0 - (1 / (n + 1)));]

    qz = quantile(z, grid)
    qd = quantile.(Ref(Distributions.Normal()), grid)

    lims = 3.290
    x = range(-lims, stop=lims)

    plot(x, x, color="grey", linestyle=":", linewidth=.25)
    scatter(qz, qd, s=.75, color=color)

    xlim([-lims, lims])
    ylim([-lims, lims])

    nothing
end