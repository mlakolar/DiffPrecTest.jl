using Distributions
using Plots

# normal Q-Q plot of the sample z added to plt
function qqplot!(plt, z; color=:black)
    n = length(z)

    grid = (1:n) ./ (n + 1)

    qz = quantile(z, grid)
    qd = quantile.(Normal(), grid)

    lims = 3.290

    plot!(plt, [-lims, lims], [-lims, lims]; color=:grey, linestyle=:dot, linewidth=0.5, label="")
    scatter!(plt, qz, qd; markersize=1, markerstrokewidth=0, color=color, label="")

    xlims!(plt, -lims, lims)
    ylims!(plt, -lims, lims)

    plt
end
